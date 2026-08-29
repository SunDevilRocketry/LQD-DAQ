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
from typing import Optional

import yaml

from daq.hardware import Device, USING_MOCK
from daq.manifest import (
    ChannelReading,
    ChannelSpec,
    ActuatorSpec,
    load_channels,
    load_actuators,
    PT_DIRECT,
    TC_DIFFERENTIAL,
    LC_DIRECT,
    PHOTOGATE_COUNTER,
)
from daq.calculations import (
    lm34_voltage_to_celsius,
    software_seebeck_type_k,
    psi_to_pa,
    pt_voltage_to_pa,
    load_cell_voltage_to_force,
)


# Manifests live at the repo root next to config.yaml. Resolved off this
# file so the defaults hold regardless of the process working directory.
_PKG_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)

DEFAULT_CHANNELS_PATH  = os.path.join(_REPO_DIR, "channels.yaml")
DEFAULT_ACTUATORS_PATH = os.path.join(_REPO_DIR, "actuators.yaml")

# Fallback calibrations by channel type, applied to every channel of that
# type that calibration.json doesn't override. Slope/intercept stay in the
# psi calibration domain for PTs (psi/V, psi), lbf for load cells.
#
#   pt_direct - Omega PX309-2K5V nominal transfer function: 0-2500 psig
#               over a 0-5 V output => 500 psi/V, 0 psi offset. Nominal
#               datasheet values.
#   lc_direct - placeholder. The PUSHTON S1 bridge is conditioned by an
#               external amplifier whose gain isn't recorded yet, so this
#               slope is a stand-in until the module is identified.
_DEFAULT_CAL_BY_TYPE: dict[str, dict[str, float]] = {
    PT_DIRECT: {"slope": 500.0, "intercept": 0.0},
    LC_DIRECT: {"slope": 100.0, "intercept": 0.0},
}

# Cold Junction Compensation (CJC) sensor polling interval in seconds.
# Photogate counters are polled on the same out-of-band tick.
_CJC_INTERVAL_S = 0.5

# LJM's missing-sample sentinel inside a stream batch.
_MISSING_SAMPLE = -9999.0

# During a comms outage no batch is processed, so state pushes come from
# the error path instead. The first failure fires immediately; subsequent
# ones are rate-limited to this interval.
_ERROR_NOTIFY_INTERVAL_S = 0.5

# FC.NLFS.LQDDAQ.1: data older than this is considered stale and should
# not be presented to an operator/dashboard as current.
_DATA_STALE_THRESHOLD_S = 1.0

# Reconnection tuning (FC.NLFS.LJ.2 / FC.NLFS.LQDDAQ.1 mitigation): after this
# many consecutive stream_read() failures, assume the connection itself is
# gone and attempt a full close/open/restart cycle.
_MAX_CONSECUTIVE_READ_ERRORS = 3
_RECONNECT_BACKOFF_S         = 1.0    # multiplied by attempt number, capped below
_RECONNECT_BACKOFF_CAP_S     = 10.0


class EngineState:
    """
    Read-only snapshot of calculated sensor data and system states.

    Split into two halves:

      Structural fields  - fixed for every cart, kept as named __slots__.
      Channel-set fields - whatever channels.yaml declares, carried in the
                           `channels` dict as ChannelReading objects.

    The channel set is manifest-driven precisely so this class does *not*
    have to be edited when the cart's sensor inventory changes.

    Instance fields are restricted using __slots__ to prevent dynamic
    attribute assignment.
    """
    __slots__ = (
        "timestamp",
        # Manifest-driven readings
        "channels",     # dict[str, ChannelReading]
        "actuators",    # dict[str, ActuatorReading]
        # Hardware & loop state
        "streaming", "sequence_active", "sequence_name",
        "using_mock", "stream_hz",
        "cjc_celsius",
    )

    def __init__(self) -> None:
        for slot in self.__slots__:
            object.__setattr__(self, slot, None)
        object.__setattr__(self, "channels", {})
        object.__setattr__(self, "actuators", {})
        object.__setattr__(self, "streaming", False)
        object.__setattr__(self, "sequence_active", False)
        object.__setattr__(self, "using_mock", USING_MOCK)


class Engine:
    """
    Central acquisition and control engine.

    Instantiate once, call start(), then read .snapshot from any thread.

    Args:
        channels_path:  Path to channels.yaml (the cart's sensor manifest).
        actuators_path: Path to actuators.yaml (the cart's actuator manifest).
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
        cal_path:       Optional[str] = None,
        sequence_dir:   str = "sequences",
        logger=None,
        thresholds:     Optional[dict] = None,
        channels_path:  str = DEFAULT_CHANNELS_PATH,
        actuators_path: str = DEFAULT_ACTUATORS_PATH,
    ) -> None:
        self._lock = threading.RLock()
        self._event_log: deque[str] = deque(maxlen=500)  # Rolling debug log console (SR 3.5)
        self._snapshot = EngineState()

        # Cart inventory - everything downstream iterates these instead of
        # a literal tag tuple.
        self._channels:  list[ChannelSpec]  = load_channels(channels_path)
        self._actuators: list[ActuatorSpec] = load_actuators(actuators_path)

        self._device        = Device(self._channels, self._actuators)
        self._logger        = logger
        self._sequence_dir  = sequence_dir
        self._thresholds    = self._convert_thresholds_to_pa(thresholds or {})

        # Calibration state (re-loaded from disk if path exists)
        self._cal_path = cal_path
        self._cal = self._default_calibration()
        if cal_path and os.path.exists(cal_path):
            self._load_calibration(cal_path)

        # Tare offsets, keyed by load cell channel id (set by caller via tare())
        self._lc_tare: dict[str, float] = {
            spec.id: 0.0 for spec in self._channels if spec.type == LC_DIRECT
        }

        # CJC polling state (thermocouple cold junction reference)
        self._cjc_celsius: float = 25.0
        self._last_cjc_read: float = 0.0

        # Latest out-of-band photogate counter values, keyed by channel id.
        self._counters: dict[str, float] = {}

        # Monotonic timestamp of the last successfully processed batch.
        self._last_batch_perf_time: float = 0.0

        # Autosequence thread handles & event flags
        self._sequence_thread:   Optional[threading.Thread] = None
        self._abort_flag         = threading.Event()
        self._sequence_active    = False
        self._sequence_name      = ""
        self._sequence_t0:       Optional[float] = None

        # Optional callback fired whenever there is new state worth pushing.
        # Kept as a plain callable so the engine stays free of any asyncio
        # or transport concern | api.py is what turns this into SSE.
        self._state_listener = None

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

    def set_state_listener(self, listener) -> None:
        """
        Registers a callback fired whenever there is new state worth pushing.

        Called with no arguments from the stream thread, both after a batch
        is processed and on an error tick where no batch arrived. The
        listener is responsible for reading the snapshot itself and for
        being quick and non-blocking - it runs on the acquisition thread.
        Pass None to detach.
        """
        self._state_listener = listener

    def _notify_state(self) -> None:
        """
        Fires the state listener, swallowing anything it raises.
        """
        listener = self._state_listener
        if listener is None:
            return
        try:
            listener()
        except Exception as exc:
            self._log(f"State listener error (ignored): {exc}")

    @property
    def sequence_elapsed_s(self) -> Optional[float]:
        """Seconds since the active sequence started, or None if idle."""
        with self._lock:
            t0 = self._sequence_t0 if self._sequence_active else None
        return None if t0 is None else time.perf_counter() - t0

    def set_logger(self, logger) -> None:
        """
        Attaches a CSV logger after construction.

        The logger's column set is built from this engine's channel
        manifest, so it can't be constructed until the engine has parsed
        channels.yaml - hence the post-construction hand-off.
        """
        self._logger = logger

    @property
    def channel_specs(self) -> list[ChannelSpec]:
        """The cart's channel manifest, as loaded from channels.yaml."""
        return list(self._channels)

    @property
    def actuator_specs(self) -> list[ActuatorSpec]:
        """The cart's actuator manifest, as loaded from actuators.yaml."""
        return list(self._actuators)

    @property
    def unwired_actuators(self) -> list[str]:
        """
        Ids of actuators the manifest declares but names no pins for.

        These exist in the inventory and are reported in every snapshot,
        but cannot be driven: LabJackT7.write_actuator() raises on them.
        """
        return [spec.id for spec in self._actuators if not spec.is_wired]

    def _default_calibration(self) -> dict[str, dict[str, float]]:
        """Builds the per-channel fallback calibration table from the manifest."""
        return {
            spec.cal_ref or spec.id: dict(_DEFAULT_CAL_BY_TYPE[spec.type])
            for spec in self._channels
            if spec.type in _DEFAULT_CAL_BY_TYPE
        }

    def _convert_thresholds_to_pa(self, raw: dict) -> dict:
        """
        Converts psi-authored pressure threshold bounds to Pa.

        Which tags count as pressures comes from the manifest (every
        pt_direct channel).
        """
        pressure_ids = {
            spec.id for spec in self._channels if spec.type == PT_DIRECT
        }
        converted: dict = {}
        for tag, bands in raw.items():
            if tag in pressure_ids and isinstance(bands, dict):
                converted[tag] = {
                    band: [psi_to_pa(lo), psi_to_pa(hi)]
                    for band, (lo, hi) in bands.items()
                }
            else:
                converted[tag] = bands
        return converted

    def _channel_status(self, tag: str, value: Optional[float]) -> Optional[str]:
        """
        Classify a reading against its config.yaml threshold bands.

        Returns:
            NOMINAL    - inside the normal band
            CAUTION    - outside normal but inside warning
            WARNING    - outside warning (interlock hazard)
            UNASSIGNED - the channel has no threshold bands configured
            None       - bands exist but there is no reading to classify

        The first three are Dashboard's existing enum (readingStatus.ts).
        UNASSIGNED extends it, and is checked first because it describes
        the *config* rather than the reading: A channel w/o set bounds is 
        an open gap. Flagging it as such.

        None survives only for the narrow case of a monitored channel with
        no data this batch - there is genuinely nothing to classify there.
        """
        bands = self._thresholds.get(tag)
        if not isinstance(bands, dict):
            return "UNASSIGNED"
        if value is None:
            return None

        normal  = bands.get("normal")
        warning = bands.get("warning")

        if normal and normal[0] <= value <= normal[1]:
            return "NOMINAL"
        if warning and warning[0] <= value <= warning[1]:
            return "CAUTION"
        return "WARNING"

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
            channels = self._snapshot.channels or {}
            for tag in self._lc_tare:
                reading = channels.get(tag)
                if reading is not None and reading.value is not None:
                    self._lc_tare[tag] += reading.value
        self._log("Load cells tared")

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
            tag:       Calibration reference tag (e.g. "pt0", "lc0").
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

        pt_tags = {
            spec.cal_ref or spec.id
            for spec in self._channels if spec.type == PT_DIRECT
        }
        lc_tags = {
            spec.cal_ref or spec.id
            for spec in self._channels if spec.type == LC_DIRECT
        }
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
            last_error_notify = 0.0
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

                    # The moment comms drop is the one moment no batch gets
                    # processed. Push the first failure immediately then throttle.
                    now_err = time.perf_counter()
                    if (consecutive_errors == 1
                            or now_err - last_error_notify >= _ERROR_NOTIFY_INTERVAL_S):
                        last_error_notify = now_err
                        self._notify_state()

                    if consecutive_errors >= _MAX_CONSECUTIVE_READ_ERRORS:
                        consecutive_errors = 0
                        if self._running:
                            self._reconnect_device()
                    elif self._running:
                        time.sleep(0.1)
                    continue

                # Periodically poll the out-of-band channels (CJC reference
                # and photogate counters) outside the hardware stream.
                now = time.perf_counter()
                if now - self._last_cjc_read >= _CJC_INTERVAL_S:
                    try:
                        cjc_v = self._device.read_cjc()
                        self._cjc_celsius = lm34_voltage_to_celsius(cjc_v)
                    except Exception:
                        pass
                    try:
                        self._counters = dict(self._device.read_counters())
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
                    now_err = time.perf_counter()
                    if now_err - last_error_notify >= _ERROR_NOTIFY_INTERVAL_S:
                        last_error_notify = now_err
                        self._notify_state()
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

    def _channel_calibration(self, spec: ChannelSpec) -> dict[str, float]:
        """Calibration coefficients for one channel, falling back to its type default."""
        return self._cal.get(
            spec.cal_ref or spec.id,
            _DEFAULT_CAL_BY_TYPE.get(spec.type, {"slope": 1.0, "intercept": 0.0}),
        )

    def _scale_channel(
        self, spec: ChannelSpec, raw: list[float], cjc_c: float
    ) -> list[Optional[float]]:
        """
        Convert one channel's raw scan voltages to engineering units.

        Dispatches on the manifest's channel type to the matching
        calculations.py function.
        """
        c = self._channel_calibration(spec)

        if spec.type == PT_DIRECT:
            return [
                None if v == _MISSING_SAMPLE
                else pt_voltage_to_pa(v, c["slope"], c["intercept"])
                for v in raw
            ]

        if spec.type == LC_DIRECT:
            tare = self._lc_tare.get(spec.id, 0.0)
            return [
                None if v == _MISSING_SAMPLE
                else load_cell_voltage_to_force(v, c["slope"], c["intercept"], tare)
                for v in raw
            ]

        if spec.type == TC_DIFFERENTIAL:
            return [
                None if v == _MISSING_SAMPLE
                else software_seebeck_type_k(v, cjc_c)
                for v in raw
            ]

        return [None] * len(raw)

    def _process_batch(self, batch: dict[str, list[float]]) -> None:
        """
        Calibrates a raw data batch and atomically swaps in a new snapshot.

        Iterates the channels.yaml manifest so the cart's sensor inventory 
        is config. Channels the manifest marks inactive & channels the hardware
        layer had no wired pin for simply carry no value. They stay in the 
        reported channel set so a consumer's key set matches the manifest.
        """
        scan_times = batch.get("scan_times", [])
        n_scans    = len(scan_times)
        if n_scans == 0:
            return

        cjc_c = self._cjc_celsius
        now_epoch = time.time()

        # Per-scan engineering values, keyed by channel id (feeds the CSV).
        eng_rows: dict[str, list[Optional[float]]] = {}
        # Snapshot value per channel - the batch's final scan, except TCs.
        latest: dict[str, Optional[float]] = {}

        for spec in self._channels:
            if not spec.active or not spec.is_streamed:
                continue
            raw = batch.get(spec.id)
            if raw is None:
                continue

            eng_rows[spec.id] = self._scale_channel(spec, raw, cjc_c)

            if spec.type == TC_DIFFERENTIAL:
                # Batch-average the raw voltages before conversion; the TC
                # signal is small enough that per-scan noise dominates.
                valid = [v for v in raw if v != _MISSING_SAMPLE]
                latest[spec.id] = (
                    software_seebeck_type_k(sum(valid) / len(valid), cjc_c)
                    if valid else None
                )
            else:
                latest[spec.id] = eng_rows[spec.id][-1]

        # Photogate counters arrive from the out-of-band poll. Informational 
        # telemetry only.
        for spec in self._channels:
            if spec.active and spec.type == PHOTOGATE_COUNTER:
                latest[spec.id] = self._counters.get(spec.id)

        channels = {
            spec.id: ChannelReading(
                value        = latest.get(spec.id),
                unit         = spec.unit,
                status       = self._channel_status(spec.id, latest.get(spec.id)),
                last_updated = now_epoch if spec.id in latest else None,
            )
            for spec in self._channels
        }

        # Build and swap the new snapshot
        new_snap = EngineState()
        object.__setattr__(new_snap, "timestamp",       scan_times[-1])
        object.__setattr__(new_snap, "streaming",       True)
        object.__setattr__(new_snap, "stream_hz",       self._device.stream_rate_hz)
        object.__setattr__(new_snap, "using_mock",      USING_MOCK)
        object.__setattr__(new_snap, "cjc_celsius",     cjc_c)
        object.__setattr__(new_snap, "channels",        channels)
        object.__setattr__(new_snap, "actuators",       dict(self._device.actuator_states()))
        object.__setattr__(new_snap, "sequence_active", self._sequence_active)
        object.__setattr__(new_snap, "sequence_name",   self._sequence_name)

        with self._lock:
            self._snapshot = new_snap
            self._last_batch_perf_time = time.perf_counter()

        # Push before the CSV work so consumers see the new state at the
        # earliest possible moment; logging is the slower, non-urgent half.
        self._notify_state()

        # Export raw hardware and calibrated values to CSV, one row per scan.
        # Column order follows logger.py's manifest-driven header.
        if self._logger:
            csv_specs = [
                spec for spec in self._channels
                if spec.active and spec.is_streamed
            ]
            for i in range(n_scans):
                row_vals: list = [scan_times[i]]
                for spec in csv_specs:
                    raw = batch.get(spec.id)
                    eng = eng_rows.get(spec.id)
                    if raw is None or eng is None or eng[i] is None:
                        row_vals.extend(["", ""])
                    else:
                        row_vals.extend([f"{raw[i]:.6f}", f"{eng[i]:.4f}"])
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

        undrivable = self._undrivable_steps(steps)
        if undrivable:
            self._log(
                f"Sequence '{name}' REFUSED: cannot drive "
                f"{', '.join(undrivable)} - not wired in actuators.yaml. "
                f"Every step for these would have failed and the sequence "
                f"would have reported completion having moved nothing."
            )
            if not is_fire:
                # An ordered closure is off the table, so fall back to the
                # same de-energise abort() uses when there is no abort.yaml
                # at all.
                self._device.all_safe()
                self._log("ABORT: refused sequence, hardware -> all safe")
            self._sequence_done()
            return

        if is_fire and self._logger:
            self._logger.start_recording(prefix="hotfire")

        t0 = time.perf_counter()
        with self._lock:
            self._sequence_t0 = t0
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
            self._sequence_t0     = None

    def _undrivable_steps(
        self, steps: list[tuple[float, str, int]]
    ) -> list[str]:
        """
        Actuator ids a sequence commands that this cart cannot drive.

        Covers both an id absent from actuators.yaml and one present but
        with null pin fields. Either way the device write raises, and
        _run_sequence's per-step handler logs it and moves on - so without
        this check a sequence runs to "completion" having moved nothing.
        """
        drivable = {spec.id for spec in self._actuators if spec.is_wired}
        return sorted({name for _, name, _ in steps if name not in drivable})

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
