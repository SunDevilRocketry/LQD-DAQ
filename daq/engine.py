"""
daq/engine.py

500 Hz background acquisition engine for the Liquids DAQ system.

Responsibilities:
  - Drives the hardware stream loop in a daemon thread.
  - Applies real-time calibration and calculations per scan.
  - Publishes atomic snapshots (no partial reads) of latest values.
  - Executes fire and abort autosequences on a dedicated thread.
  - Controls CSV logging lifecycle.
  - Exposes thread-safe data to the API and GUI.

Design constraints (from SR 3.4.1):
  - File I/O is decoupled from the stream thread to prevent buffer overflows.
  - Every 500 Hz scan is logged; consumer threads poll snapshots at lower rates.
  - Sequence timing uses perf_counter absolute baselines to prevent timing drift.

Threading model:
  - _stream_thread (Producer):     Acquires data and publishes EngineState snapshots.
  - _sequence_thread:              Executes autosequence steps and hardware outputs.
  - Main / API Thread (Consumer):  Reads the latest snapshot with minimal lock contention.
  - self._lock:                    Protects all shared mutable state.
"""

from __future__ import annotations

import json
import os
import time
import threading
import traceback
from collections import deque
from typing import Optional, Any

import yaml

from daq.hardware import Device, USING_MOCK
from daq.calculations import (
    lm34_voltage_to_celsius,
    software_seebeck_type_k,
    psi_to_pa,
    pt_voltage_to_pa,
    load_cell_voltage_to_force,
    lox_mass_flow_rate,
    fuel_mass_flow_rate,
    mixture_ratio,
    impulse_step_load_cell,
    impulse_step_estimate,
    lox_below_saturation,
)


# Fallback calibrations (overridden by calibration.json at startup).
# PT slope/intercept remain in psi/V, psi
_DEFAULT_CAL: dict[str, dict[str, float]] = {
    "POT": {"slope": 252.0, "intercept": -106.0},
    "PFT": {"slope": 252.0, "intercept": -121.0},
    "POI": {"slope": 252.0, "intercept": -119.5},
    "PFI": {"slope": 252.0, "intercept": -119.5},
    "PFO": {"slope": 252.0, "intercept": -119.5},
    "PC":  {"slope": 128.0, "intercept": -62.8},
    "PNS": {"slope": 252.0, "intercept": -116.5},
    "PNP": {"slope": 252.0, "intercept": -104.5},
    "LC_1": {"slope": 100.0, "intercept": 0.0},
    "LC_2": {"slope": 100.0, "intercept": 0.0},
}

# Cold Junction Compensation (CJC) sensor polling interval in seconds.
_CJC_INTERVAL_S = 0.5

# FC.NLFS.LQDDAQ.1: data older than this is considered stale and should
# not be presented to an operator/dashboard as current.
_DATA_STALE_THRESHOLD_S = 1.0

# Specific impulse (seconds) used for impulse estimation if load cells are absent.
_ISP_ESTIMATE_S = 220.0

# Standard sea-level atmospheric pressure (Pa) used to convert gauge -> absolute
_ATMO_PA = 101_325.0

# Threshold config (config.yaml) is authored in psi for these tags
_PRESSURE_THRESHOLD_TAGS = frozenset({
    "POT", "PFT", "POI", "PFI", "PFO", "PC", "PNS", "PNP",
})

# Reconnection tuning (FC.NLFS.LJ.2 / FC.NLFS.LQDDAQ.1 mitigation): after this
# many consecutive stream_read() failures, assume the connection itself is
# gone and attempt a full close/open/restart cycle.
_MAX_CONSECUTIVE_READ_ERRORS = 3
_RECONNECT_BACKOFF_S         = 1.0    # multiplied by attempt number, capped below
_RECONNECT_BACKOFF_CAP_S     = 10.0


class EngineState:
    """
    Read-only snapshot of calculated sensor data and system states.

    All scalar fields are primitive types or None (Json serialization ease).
    Instance fields are restricted using __slots__ to prevent dynamic 
    attribute assignment.
    """
    __slots__ = (
        "timestamp",
        # Calibrated sensor readings
        "POT", "PFT", "POI", "PFI", "PFO", "PC", "PNS", "PNP",
        "TOI", "TFI", "TFO",
        "LC_1", "LC_2",
        # Calculated physical values
        "lox_mdot", "fuel_mdot", "mixture_ratio",
        "impulse_lbfs", "impulse_ns",
        "lox_below_sat",
        # Hardware & loop state
        "actuators",
        "streaming", "sequence_active", "sequence_name",
        "using_mock", "stream_hz",
        "cjc_celsius",
    )

    def __init__(self) -> None:
        for slot in self.__slots__:
            object.__setattr__(self, slot, None)
        object.__setattr__(self, "actuators", {})
        object.__setattr__(self, "streaming", False)
        object.__setattr__(self, "sequence_active", False)
        object.__setattr__(self, "using_mock", USING_MOCK)


class Engine:
    """
    Central acquisition and control engine.

    Instantiate once, call start(), then read .snapshot from any thread.

    Args:
        cal_path:      Path to calibration.json (optional).
                       Also used by save_calibration() to persist runtime changes.
        sequence_dir:  Directory containing fire.yaml / abort.yaml.
        logger:        A daq.logger.Logger instance (optional).
                       If None, no CSV logging occurs.
                       The caller is responsible for logger.open() / logger.close();
                       the engine only calls start_recording() / stop_recording().
        thresholds:    Warning and abort thresholds dictionary (optional).
    """

    # Public alias so callers can reference the threshold
    # w/o reaching into the private module-level constant.
    DATA_STALE_THRESHOLD_S = _DATA_STALE_THRESHOLD_S

    def __init__(
        self,
        cal_path:     Optional[str] = None,
        sequence_dir: str = "sequences",
        logger=None,
        thresholds:   Optional[dict] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._event_log: deque[str] = deque(maxlen=500)  # Rolling debug log console (SR 3.5)
        self._snapshot = EngineState()

        self._device        = Device()
        self._logger        = logger
        self._sequence_dir  = sequence_dir
        self._thresholds    = self._convert_thresholds_to_pa(thresholds or {})

        # Calibration state (re-loaded from disk if path exists)        
        self._cal_path = cal_path
        self._cal = dict(_DEFAULT_CAL)
        if cal_path and os.path.exists(cal_path):
            self._load_calibration(cal_path)

        # Tare offsets for load cells (set by caller via tare())
        self._lc_tare: dict[str, float] = {"LC_1": 0.0, "LC_2": 0.0}

        # CJC polling state (thermocouple cold junction reference)
        self._cjc_celsius: float = 25.0
        self._last_cjc_read: float = 0.0

        # Monotonic timestamp of the last successfully processed batch.
        self._last_batch_perf_time: float = 0.0

        # Impulse integration accumulators
        self._impulse_lbfs:     float = 0.0
        self._impulse_ns:       float = 0.0
        self._prev_force_lbf:   Optional[float] = None
        self._prev_scan_time:   Optional[float] = None

        # Autosequence thread handles & event flags
        self._sequence_thread:   Optional[threading.Thread] = None
        self._abort_flag         = threading.Event()
        self._sequence_active    = False
        self._sequence_name      = ""

        # Hardware streaming thread state
        self._stream_thread: Optional[threading.Thread] = None
        self._running = False

    # --------------------------------------------------------
    # Public API (thread-safe)
    # --------------------------------------------------------

    def start(self) -> None:
        """Connects to hardware and starts the acquisition loop on a daemon thread."""
        self._log(f"Engine starting (mock={USING_MOCK})")
        self._device.open()
        self._running = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="daq-stream",
            daemon=True,
        )
        self._stream_thread.start()

    def stop(self) -> None:
        """Stops background acquisition and disconnects hardware safely."""
        self._log("Engine stopping")
        self._running = False
        self.abort()

        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)

        try:
            self._device.stop_stream()
            self._device.close()
        except Exception as exc:
            self._log(f"Warning during stop: {exc}")

        self._log("Engine stopped")

    @property
    def snapshot(self) -> EngineState:
        """Returns the latest thread-safe sensor snapshot."""
        with self._lock:
            return self._snapshot

    @property
    def event_log(self) -> list[str]:
        """Returns a copy of the rolling debug event log."""
        with self._lock:
            return list(self._event_log)

    @property
    def thresholds(self) -> dict:
        """Returns configured safety thresholds."""
        with self._lock:
            return dict(self._thresholds)

    @staticmethod
    def _convert_thresholds_to_pa(raw: dict) -> dict:
        """
        Converts psi-authored pressure threshold bounds to Pa
        """
        converted: dict = {}
        for tag, bands in raw.items():
            if tag in _PRESSURE_THRESHOLD_TAGS and isinstance(bands, dict):
                converted[tag] = {
                    band: [psi_to_pa(lo), psi_to_pa(hi)]
                    for band, (lo, hi) in bands.items()
                }
            else:
                converted[tag] = bands
        return converted

    @property
    def data_age_seconds(self) -> float:
        """
        Seconds since the last successfully processed stream batch.

        Returns float("inf") if no batch has ever been processed. Snaps back 
	near-zero when fresh batch is processed.
        """
        with self._lock:
            last = self._last_batch_perf_time
        if last == 0.0:
            return float("inf")
        return time.perf_counter() - last

    @property
    def is_data_stale(self) -> bool:
        """True if the most recent snapshot is older than the FC.NLFS.LQDDAQ.1 threshold."""
        return self.data_age_seconds > _DATA_STALE_THRESHOLD_S

    def write_actuator(self, name: str, state: int) -> None:
        """
        Commands an individual actuator if no autosequence is active.
        
        Args:
            name:  Actuator name.
            state: 1 = open, 0 = safe/closed.

        Raises:
            RuntimeError: If a sequence is currently active.
        """
        with self._lock:
            if self._sequence_active:
                raise RuntimeError(
                    "Cannot send manual commands while a sequence is running. "
                    "Send ABORT first."
                )
        self._device.write_actuator(name, state)
        self._log(f"Manual: {name} -> {'OPEN' if state else 'CLOSED'}")

    def all_safe(self) -> None:
        """De-energizes all hardware outputs immediately."""
        self.abort()
        self._device.all_safe()
        self._log("ALL SAFE commanded")

    def tare(self) -> None:
        """Tares the load cells using the latest baseline sensor readings."""
        with self._lock:
            snap = self._snapshot
            for tag in ("LC_1", "LC_2"):
                raw = getattr(snap, tag, None)
                if raw is not None:
                    self._lc_tare[tag] = raw
        self._log("Load cells tared")

    def reset_impulse(self) -> None:
        """Resets the total impulse integration accumulators."""
        with self._lock:
            self._impulse_lbfs   = 0.0
            self._impulse_ns     = 0.0
            self._prev_force_lbf = None
            self._prev_scan_time = None

    def fire(self) -> None:
        """Spawns the fire autosequence thread if the system is idle."""
        seq_path = os.path.join(self._sequence_dir, "fire.yaml")
        if not self._start_sequence("fire", seq_path, is_fire=True):
            self._log("Fire command ignored: autosequence already active")

    def abort(self) -> None:
        """
        Terminates any active sequence and runs the abort sequence.

        Guarantees that the abort thread is spawned safely without 
        double-activation conflicts.
        """
        self._abort_flag.set()

        # Wait for the running sequence thread to notice the flag and exit.
        old_thread = self._sequence_thread
        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=0.5)
            if old_thread.is_alive():
                # It didn't exit in time (don't clear flag yet)
		# Give it a bit longer.
                self._log(
                    f"WARNING: '{old_thread.name}' did not exit within 0.5s "
                    f"of abort - waiting longer before starting abort.yaml"
                )
                old_thread.join(timeout=2.0)
                if old_thread.is_alive():
                    self._log(
                        f"WARNING: '{old_thread.name}' still alive after 2.5s "
                        f"total - force-starting abort sequence anyway. "
                        f"Actuator states may be contested by both threads."
                    )

        self._abort_flag.clear()

        seq_path = os.path.join(self._sequence_dir, "abort.yaml")
        if os.path.exists(seq_path):
            # force=True: abort must always be able to preempt, even if the
            # previous sequence thread is (unexpectedly) still marked alive.
            self._start_sequence("abort", seq_path, is_fire=False, force=True)
        else:
            self._device.all_safe()
            self._log("ABORT: no abort.yaml found, hardware -> all safe")

    def update_calibration(self, tag: str, slope: float, intercept: float) -> None:
        """
        Update a sensor calibration coefficient at runtime.

        Args:
            tag:       Sensor tag (e.g. "PC", "POT").
            slope:     New slope in engineering-units per volt.
            intercept: New intercept in engineering units.
        """
        with self._lock:
            self._cal[tag] = {"slope": slope, "intercept": intercept}
        self._log(f"Cal updated: {tag} slope={slope} intercept={intercept}")

    def save_calibration(self) -> None:
        """
        Saves current in-memory calibrations to calibration.json.

        Raises:
            AttributeError: If no cal_path was provided at construction time.
            OSError: If the file cannot be written.
        """
        if not self._cal_path:
            raise AttributeError(
                "No cal_path was set at Engine construction time. "
                "Pass cal_path=<path> to enable persistence."
            )

        pt_tags = {"POT", "PFT", "POI", "PFI", "PFO", "PC", "PNS", "PNP"}
        lc_tags = {"LC_1", "LC_2"}
        output: dict[str, dict] = {"PT": {}, "LC": {}}

        with self._lock:
            cal_snapshot = dict(self._cal)

        for tag, coeffs in cal_snapshot.items():
            if tag in pt_tags:
                output["PT"][tag] = dict(coeffs)
            elif tag in lc_tags:
                output["LC"][tag] = dict(coeffs)

        with open(self._cal_path, "w") as f:
            json.dump(output, f, indent=2)

        self._log(f"Calibration saved to {self._cal_path}")

    # --------------------------------------------------------
    # Stream loop (runs in _stream_thread)
    # --------------------------------------------------------

    def _stream_loop(self) -> None:
        """
        Main acquisition loop.

        1. Starts the hardware stream.
        2. Reads streaming data in batches.
        3. Calibrates and processes each batch.
        4. Atomically updates the state snapshot.
        5. Writes raw and calibrated data rows to the logger.
        """
        try:
            actual_hz = self._device.start_stream()
            self._log(f"Stream running at {actual_hz} Hz")

            with self._lock:
                s = self._snapshot
                object.__setattr__(s, "streaming",   True)
                object.__setattr__(s, "stream_hz",   actual_hz)
                object.__setattr__(s, "using_mock",  USING_MOCK)

            consecutive_errors = 0
            while self._running:
                try:
                    batch = self._device.stream_read()
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    self._log(
                        f"Stream read error ({consecutive_errors}/"
                        f"{_MAX_CONSECUTIVE_READ_ERRORS}): {exc}"
                    )
                    with self._lock:
                        object.__setattr__(self._snapshot, "streaming", False)

                    if consecutive_errors >= _MAX_CONSECUTIVE_READ_ERRORS:
                        consecutive_errors = 0
                        if self._running:
                            self._reconnect_device()
                    elif self._running:
                        time.sleep(0.1)
                    continue

                # Periodically poll CJC temperature outside the hardware stream
                now = time.perf_counter()
                if now - self._last_cjc_read >= _CJC_INTERVAL_S:
                    try:
                        cjc_v = self._device.read_cjc()
                        self._cjc_celsius = lm34_voltage_to_celsius(cjc_v)
                    except Exception:
                        pass
                    self._last_cjc_read = now

                try:
                    self._process_batch(batch)
                except Exception as exc:
                    self._log(
                        f"Batch processing error (scan dropped, stream continues): "
                        f"{exc}\n{traceback.format_exc()}"
                    )
                    continue

        except Exception as exc:
            self._log(f"Stream loop fatal: {exc}\n{traceback.format_exc()}")
        finally:
            with self._lock:
                object.__setattr__(self._snapshot, "streaming", False)

    def _sleep_interruptible(self, duration: float) -> None:
        """
        Sleep in small increments, rechecking self._running throughout,
        so a stop() request is noticed promptly instead of being blocked 
        behind a single long time.sleep() call.
        """
        deadline = time.perf_counter() + duration
        while self._running and time.perf_counter() < deadline:
            time.sleep(min(0.1, deadline - time.perf_counter()))

    def _reconnect_device(self) -> None:
        """
        Attempts to fully re-establish the hardware connection after repeated
        stream_read() failures.

        Loops with linear backoff (capped) until self._running goes False or
        a reconnect attempt succeeds. The job is only to get software back 
        in sync once comms return.
        """
        self._log("Device connection lost - attempting to reconnect...")
        attempt = 0
        while self._running:
            attempt += 1
            try:
                try:
                    self._device.stop_stream()
                except Exception:
                    pass
                try:
                    self._device.close()
                except Exception:
                    pass

                backoff = min(_RECONNECT_BACKOFF_S * attempt, _RECONNECT_BACKOFF_CAP_S)
                self._sleep_interruptible(backoff)
                if not self._running:
                    break   # stop() was called mid-backoff - don't touch the device again

                self._device.open()
                actual_hz = self._device.start_stream()

                with self._lock:
                    s = self._snapshot
                    object.__setattr__(s, "streaming", True)
                    object.__setattr__(s, "stream_hz", actual_hz)

                self._log(
                    f"Reconnected on attempt {attempt}: stream running at "
                    f"{actual_hz} Hz"
                )
                return
            except Exception as exc:
                self._log(f"Reconnect attempt {attempt} failed: {exc}")

        self._log("Reconnect loop exiting - engine is stopping")

    def _process_batch(self, batch: dict[str, list[float]]) -> None:
        """Calibrates, computes derived values for a raw data batch, and updates the snapshot."""
        scan_times = batch.get("scan_times", [])
        n_scans    = len(scan_times)
        if n_scans == 0:
            return

        cal   = self._cal
        cjc_c = self._cjc_celsius

        # Accumulate high-frequency TC readings for batch-averaging to mitigate noise
        tc_accum: dict[str, list[float]] = {tag: [] for tag in ("TOI", "TFI", "TFO")}

        last: dict[str, Any] = {}

        # Capture intermediate impulse step values per scan for accurate row logging
        impulse_ns_per_scan: list[float] = []

        for i in range(n_scans):
            t = scan_times[i]
            row: dict[str, Any] = {"t": t}

            # Scale pressure transducers
            for tag in ("POT", "PFT", "POI", "PFI", "PFO", "PC", "PNS", "PNP"):
                v = batch[tag][i]
                if v == -9999.0:
                    row[tag] = None
                    continue
                c = cal.get(tag, {"slope": 252.0, "intercept": -119.5})
                row[tag] = pt_voltage_to_pa(v, c["slope"], c["intercept"])

            # Accumulate thermocouple raw voltages
            for tag in ("TOI", "TFI", "TFO"):
                v = batch[tag][i]
                if v != -9999.0:
                    tc_accum[tag].append(v)
                row[tag] = None

            # Scale and tare load cells
            for tag in ("LC_1", "LC_2"):
                v = batch[tag][i]
                if v == -9999.0:
                    row[tag] = None
                    continue
                c    = cal.get(tag, {"slope": 100.0, "intercept": 0.0})
                tare = self._lc_tare.get(tag, 0.0)
                row[tag] = load_cell_voltage_to_force(v, c["slope"], c["intercept"], tare)

            # Integrate force over time to accumulate total impulse (trapezoidal step)
            lc1 = row.get("LC_1")
            lc2 = row.get("LC_2")
            if lc1 is not None and lc2 is not None:
                total_force = lc1 + lc2
                if self._prev_force_lbf is not None and self._prev_scan_time is not None:
                    dt = t - self._prev_scan_time
                    if 0 < dt < 1.0:
                        step = impulse_step_load_cell(
                            self._prev_force_lbf, total_force, dt
                        )
                        self._impulse_lbfs += step
                        self._impulse_ns   += step * 4.44822
                self._prev_force_lbf = total_force
                self._prev_scan_time = t

            impulse_ns_per_scan.append(self._impulse_ns)
            last = row

        # Compute batch-averaged engineering values for thermocouples
        tc_eng: dict[str, Optional[float]] = {}
        for tag, vals in tc_accum.items():
            if vals:
                avg_v = sum(vals) / len(vals)
                tc_eng[tag] = software_seebeck_type_k(avg_v, cjc_c)
            else:
                tc_eng[tag] = None

        # Calculate derived mass flow and mixture ratios
        toi  = tc_eng.get("TOI")
        poi  = last.get("POI")
        pfo  = last.get("PFO")
        pc   = last.get("PC")
        pot  = last.get("POT")

        # Guard against None values (zero is a valid physical value).
        lox_mdot  = (
            lox_mass_flow_rate(toi, poi, pc)
            if (toi is not None and poi is not None and pc is not None)
            else None
        )
        fuel_mdot = (
            fuel_mass_flow_rate(pfo, pc)
            if (pfo is not None and pc is not None)
            else None
        )
        of_ratio  = mixture_ratio(lox_mdot, fuel_mdot)

        # Convert gauge Pa to absolute Pa for the Antoine saturation calculation
        below_sat: Optional[bool] = None
        if toi is not None and pot is not None:
            pot_pa_abs = pot + _ATMO_PA
            below_sat = lox_below_saturation(pot_pa_abs, toi)

        # Apply mathematical fallback calculation for impulse if load cells are missing
        lc1_v = last.get("LC_1")
        lc2_v = last.get("LC_2")
        if (lc1_v is None or lc2_v is None) and (lox_mdot or fuel_mdot):
            dt_est = (1.0 / self._device.stream_rate_hz) * n_scans
            est_step = impulse_step_estimate(lox_mdot, fuel_mdot, dt_est, _ISP_ESTIMATE_S)
            if est_step is not None:
                self._impulse_ns += est_step

        # Build and swap the new snapshot
        new_snap = EngineState()
        object.__setattr__(new_snap, "timestamp",       last.get("t"))
        object.__setattr__(new_snap, "streaming",       True)
        object.__setattr__(new_snap, "stream_hz",       self._device.stream_rate_hz)
        object.__setattr__(new_snap, "using_mock",      USING_MOCK)
        object.__setattr__(new_snap, "cjc_celsius",     cjc_c)
        object.__setattr__(new_snap, "actuators",       dict(self._device.actuator_states()))
        object.__setattr__(new_snap, "sequence_active", self._sequence_active)
        object.__setattr__(new_snap, "sequence_name",   self._sequence_name)

        for tag in ("POT", "PFT", "POI", "PFI", "PFO", "PC", "PNS", "PNP",
                    "LC_1", "LC_2"):
            object.__setattr__(new_snap, tag, last.get(tag))

        for tag in ("TOI", "TFI", "TFO"):
            object.__setattr__(new_snap, tag, tc_eng.get(tag))

        object.__setattr__(new_snap, "lox_mdot",      lox_mdot)
        object.__setattr__(new_snap, "fuel_mdot",     fuel_mdot)
        object.__setattr__(new_snap, "mixture_ratio", of_ratio)
        object.__setattr__(new_snap, "impulse_lbfs",  self._impulse_lbfs)
        object.__setattr__(new_snap, "impulse_ns",    self._impulse_ns)
        object.__setattr__(new_snap, "lox_below_sat", below_sat)

        with self._lock:
            self._snapshot = new_snap
            self._last_batch_perf_time = time.perf_counter()

        # Export calibrated values and raw hardware rows to CSV
        if self._logger:
            for i in range(n_scans):
                t = scan_times[i]
                row_vals: list = [t]
                for tag in ("POT", "PFT", "POI", "PFI", "PFO",
                            "PC", "PNS", "PNP"):
                    v = batch[tag][i]
                    if v == -9999.0:
                        row_vals.extend(["", ""])
                    else:
                        c   = cal.get(tag, {"slope": 252.0, "intercept": -119.5})
                        eng = pt_voltage_to_pa(v, c["slope"], c["intercept"])
                        row_vals.extend([f"{v:.6f}", f"{eng:.4f}"])
                for tag in ("TOI", "TFI", "TFO"):
                    v = batch[tag][i]
                    if v == -9999.0:
                        row_vals.extend(["", ""])
                    else:
                        eng = software_seebeck_type_k(v, cjc_c)
                        row_vals.extend([f"{v:.6f}", f"{eng:.4f}"])
                for tag in ("LC_1", "LC_2"):
                    v = batch[tag][i]
                    if v == -9999.0:
                        row_vals.extend(["", ""])
                    else:
                        c    = cal.get(tag, {"slope": 100.0, "intercept": 0.0})
                        tare = self._lc_tare.get(tag, 0.0)
                        eng  = load_cell_voltage_to_force(v, c["slope"], c["intercept"], tare)
                        row_vals.extend([f"{v:.6f}", f"{eng:.4f}"])
                # Per-scan impulse - not batch-final
                row_vals.append(f"{impulse_ns_per_scan[i]:.4f}")
                row_vals.append(f"{lox_mdot:.6f}"  if lox_mdot  is not None else "")
                row_vals.append(f"{fuel_mdot:.6f}" if fuel_mdot is not None else "")
                self._logger.write_row(row_vals)

    # --------------------------------------------------------
    # Sequence control
    # --------------------------------------------------------

    def _start_sequence(
        self, name: str, path: str, is_fire: bool, force: bool = False
    ) -> bool:
        """
        Spawns a new background thread to execute an autosequence.

        Args:
            force: If True, starts even if a sequence thread object is still
                   marked alive (used by abort(), which must always be able
                   to preempt). If False (default), refuses to start a second
                   sequence on top of a live one - callers should check the
                   return value rather than assuming success.

        Returns:
            True if the sequence thread was started, False if refused
            because another sequence is already active.
        """
        with self._lock:
            thread_alive = (
                self._sequence_thread is not None
                and self._sequence_thread.is_alive()
            )
            if thread_alive and not force:
                return False
            if thread_alive:
                self._log(
                    f"WARNING: force-starting '{name}' while previous thread "
                    f"'{self._sequence_thread.name}' is still alive."
                )
            self._sequence_active = True
            self._sequence_name   = name

        self._abort_flag.clear()
        self._sequence_thread = threading.Thread(
            target=self._run_sequence,
            args=(name, path, is_fire),
            name=f"daq-seq-{name}",
            daemon=True,
        )
        self._sequence_thread.start()
        return True

    def _run_sequence(self, name: str, path: str, is_fire: bool) -> None:
        """Executes a YAML-defined autosequence step-by-step."""
        self._log(f"Sequence '{name}' starting")

        try:
            steps, post_s = self._load_sequence(path)
        except Exception as exc:
            self._log(f"Sequence load failed: {exc}")
            self._sequence_done()
            return

        if is_fire and self._logger:
            self._logger.start_recording(prefix="hotfire")

        t0 = time.perf_counter()
        idx = 0

        while idx < len(steps):
            # Exit thread immediately on abort flag (applicable to fire sequences only)
            if self._abort_flag.is_set() and is_fire:
                self._log(
                    "Sequence aborted - exiting. "
                    "abort() will handle the abort sequence launch."
                )
                self._sequence_done()
                return

            elapsed   = time.perf_counter() - t0
            target_t, act_name, state = steps[idx]

            if elapsed >= target_t:
                try:
                    self._device.write_actuator(act_name, state)
                    self._log(
                        f"[{name}] T+{target_t:.2f}s  "
                        f"{act_name} -> {'OPEN' if state else 'CLOSED'}"
                    )
                except Exception as exc:
                    self._log(f"Actuator write error: {exc}")
                idx += 1
            else:
                # Sleep briefly to prevent high CPU utilization while awaiting timing targets
                time.sleep(0.0005)

        # Keep recording post-sequence telemetry until timeout or abort
        if self._logger and post_s > 0:
            self._log(f"Sequence '{name}' complete, recording {post_s:.1f} s post-data")
            deadline = time.perf_counter() + post_s
            while time.perf_counter() < deadline:
                if self._abort_flag.is_set():
                    self._log("Post-record window interrupted by abort")
                    break
                time.sleep(0.05)

        if self._logger:
            self._logger.stop_recording()

        self._log(f"Sequence '{name}' finished")
        self._sequence_done()

    def _sequence_done(self) -> None:
        """Resets the active sequence flags in the shared state."""
        with self._lock:
            self._sequence_active = False
            self._sequence_name   = ""

    def _load_sequence(
        self, path: str
    ) -> tuple[list[tuple[float, str, int]], float]:
        """
        Parse a YAML sequence file.

        Returns:
            (steps, post_record_seconds)
            steps: sorted list of (elapsed_s, actuator_name, state)
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        post_s = float(data.get("post_record_seconds", 5.0))
        raw    = data.get("steps", [])
        steps  = [(float(s[0]), str(s[1]), int(s[2])) for s in raw]
        steps.sort(key=lambda s: s[0])
        return steps, post_s

    # --------------------------------------------------------
    # Calibration persistence
    # --------------------------------------------------------

    def _load_calibration(self, path: str) -> None:
        """Loads and merges coefficients from calibration.json."""
        try:
            with open(path) as f:
                saved = json.load(f)
            for tag, coeffs in saved.get("PT", {}).items():
                if tag in self._cal:
                    self._cal[tag].update(coeffs)
            for tag, coeffs in saved.get("LC", {}).items():
                if tag in self._cal:
                    self._cal[tag].update(coeffs)
            self._log(f"Calibration loaded from {path}")
        except Exception as exc:
            self._log(f"Calibration load warning: {exc} - using defaults")

    # --------------------------------------------------------
    # Internal logging (SR 3.5)
    # --------------------------------------------------------

    def _log(self, message: str) -> None:
        """Appends a timestamped string to the rolling debug event log."""
        entry = f"{time.strftime('%H:%M:%S')} {message}"
        with self._lock:
            self._event_log.append(entry)
        print(f"[ENGINE] {entry}")
