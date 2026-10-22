"""
daq/engine.py

Acquisition and sequencing engine. The stream thread calibrates each batch
and swaps in a new snapshot; a sequence thread runs fire/abort; API threads
read the snapshot. self._lock guards shared state.
"""

from __future__ import annotations

import json
import os
import time
import threading
import traceback
from collections import deque
from dataclasses import dataclass, replace
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
    PT_DIFFERENTIAL,
    PRESSURE_TYPES,
    TC_DIFFERENTIAL,
    LC_DIRECT,
    PHOTOGATE_COUNTER,
    PULSE_STEPPER,
    BINARY_DIO,
)
from daq.calculations import (
    thermocouple_type_t,
    psi_to_pa,
    pt_voltage_to_pa,
    load_cell_voltage_to_force,
    orifice_area_m2,
    lox_mass_flow_rate,
    fuel_mass_flow_rate,
    mixture_ratio,
)


# Repo-root manifests, resolved off this file so the cwd doesn't matter.
_PKG_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)

DEFAULT_CHANNELS_PATH  = os.path.join(_REPO_DIR, "channels.yaml")
DEFAULT_ACTUATORS_PATH = os.path.join(_REPO_DIR, "actuators.yaml")

# Nominal fallbacks where calibration.json has no entry (psi/V, lbf/V).
# The dPT slope is UNVERIFIED.
_DEFAULT_CAL_BY_TYPE: dict[str, dict[str, float]] = {
    PT_DIRECT: {"slope": 400.0, "intercept": 0.0},
    PT_DIFFERENTIAL: {"slope": 5.80151, "intercept": 0.0},
    LC_DIRECT: {"slope": 440.92, "intercept": 0.0},
}

# CJC + counter poll. Also the watchdog keep-alive (stream data doesn't
# reset it): stop it and the T7 safes every output after 5 s.
_CJC_INTERVAL_S = 0.5

# --mock only: stand-ins for stepper step counts/rate that aren't set yet,
# so sequences can be exercised without hardware numbers.
_MOCK_STEPS     = 400
_MOCK_PULSE_HZ  = 2000.0

# SysRD 2.5.4 minimum sample rate.
_MIN_STREAM_HZ = 200

# Warn when the host falls this far behind the stream (default; tune on the
# cart).
_BACKLOG_WARN_S = 0.25

# LJM's missing-sample sentinel.
_MISSING_SAMPLE = -9999.0

# State pushes during an outage: first failure at once, then this often.
_ERROR_NOTIFY_INTERVAL_S = 0.5

# FC.NLFS.LQDDAQ.1: older data is stale, not live.
_DATA_STALE_THRESHOLD_S = 1.0

# FC.NLFS.LJ.2: this many failed reads in a row -> full reconnect.
_MAX_CONSECUTIVE_READ_ERRORS = 3
_RECONNECT_BACKOFF_S         = 1.0    # x attempt number, capped
_RECONNECT_BACKOFF_CAP_S     = 10.0

# De-energised on pause; fire first so the e-match circuit opens first.
_IGNITER_IDS = ("ignition_fire", "ignition_arm")

# Fire lifecycle, so a dashboard can tell why sequence_active is false.
FIRE_IDLE      = "idle"
FIRE_RUNNING   = "running"
FIRE_PAUSED    = "paused"
FIRE_COMPLETED = "completed"
FIRE_ABORTED   = "aborted"


class EngineState:
    """
    One published snapshot. Fixed fields are slots; per-channel readings
    live in `channels`, so a manifest change needs no edit here.
    """
    __slots__ = (
        "timestamp",
        # Manifest-driven readings
        "channels",     # dict[str, ChannelReading]
        "derived",      # dict[str, float | None], calculated
        "actuators",    # dict[str, ActuatorReading]
        # Hardware & loop state
        "streaming", "sequence_active", "sequence_name", "sequence_step",
        "sequence_state",
        "using_mock", "stream_hz",
        "cjc_celsius",
        "backlog",      # {"device": scans, "ljm": scans} still unread
    )

    def __init__(self) -> None:
        for slot in self.__slots__:
            object.__setattr__(self, slot, None)
        object.__setattr__(self, "channels", {})
        object.__setattr__(self, "derived", {})
        object.__setattr__(self, "actuators", {})
        object.__setattr__(self, "streaming", False)
        object.__setattr__(self, "sequence_active", False)
        object.__setattr__(self, "sequence_state", FIRE_IDLE)
        object.__setattr__(self, "using_mock", USING_MOCK)


def _parse_action(raw) -> tuple[float, str, int]:
    """
    One sequence action, [elapsed_s, actuator_id, state].

    Raises:
        ValueError: If it isn't a 3-item list or state isn't 0 or 1.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"action {raw!r} is not [elapsed_s, actuator, state]")
    elapsed, name, state = raw
    if isinstance(state, bool) or state not in (0, 1):
        raise ValueError(f"action {raw!r}: state must be 0 or 1")
    return float(elapsed), str(name), int(state)


class SequenceRefused(RuntimeError):
    """A sequence can't run; raised on the caller's thread before it starts."""


@dataclass(frozen=True)
class FireStep:
    """One named phase of fire.yaml; start_t (earliest action) is the jump target."""
    name:    str
    start_t: float
    actions: tuple[tuple[float, str, int], ...]


class Engine:
    """
    Acquisition and control engine. Create once, start(), read .snapshot.

    Args:
        cal_path:     calibration.json; save_calibration() writes it back.
        sequence_dir: holds fire.yaml / abort.yaml.
        logger:       optional; the caller opens/closes it, the engine only
                      starts/stops recordings.
        thresholds:   config.yaml display bands (nothing acts on them).
        derived:      config.yaml `derived` block (mass-flow inputs).
    """

    DATA_STALE_THRESHOLD_S = _DATA_STALE_THRESHOLD_S

    def __init__(
        self,
        cal_path:       Optional[str] = None,
        sequence_dir:   str = "sequences",
        logger=None,
        thresholds:     Optional[dict] = None,
        derived:        Optional[dict] = None,
        channels_path:  str = DEFAULT_CHANNELS_PATH,
        actuators_path: str = DEFAULT_ACTUATORS_PATH,
        stream_hz:      int = 500,
        mock_stepper_defaults: bool = False,
    ) -> None:
        if stream_hz < _MIN_STREAM_HZ:
            raise ValueError(
                f"stream.target_hz {stream_hz} is below the "
                f"{_MIN_STREAM_HZ} Hz minimum (SysRD 2.5.4)"
            )
        self._lock = threading.RLock()
        self._event_log: deque[str] = deque(maxlen=500)  # SR 3.5
        self._logger = logger
        self._snapshot = EngineState()

        self._channels:  list[ChannelSpec]  = load_channels(channels_path)
        self._actuators: list[ActuatorSpec] = load_actuators(actuators_path)
        if mock_stepper_defaults:
            self._actuators = [self._mock_stepper(s) for s in self._actuators]

        self._device        = Device(
            self._channels, self._actuators, stream_hz=stream_hz
        )
        self._backlog_warned = False
        self._sequence_dir  = sequence_dir
        self._thresholds    = self._convert_thresholds_to_pa(thresholds or {})
        self._derived_cfg   = derived or {}

        self._cal_path = cal_path
        self._cal = self._default_calibration()
        if cal_path and os.path.exists(cal_path):
            self._load_calibration(cal_path)

        # lbf, by load cell id.
        self._lc_tare: dict[str, float] = {
            spec.id: 0.0 for spec in self._channels if spec.type == LC_DIRECT
        }
        # Mean of each load cell's last batch (lbf, tared), for the next tare.
        self._lc_batch_mean: dict[str, float] = {}

        # None until a good read and after a failed one, so TCs report nothing.
        self._cjc_celsius: Optional[float] = None
        self._last_cjc_read: float = 0.0
        self._cjc_failing: bool = False
        self._counters_failing: bool = False

        self._counters: dict[str, float] = {}

        self._last_batch_perf_time: float = 0.0

        self._sequence_thread:   Optional[threading.Thread] = None
        self._abort_flag         = threading.Event()
        self._sequence_active    = False
        self._sequence_name      = ""
        self._sequence_t0:       Optional[float] = None

        # Fire-only pause/seek state.
        self._pause_flag         = threading.Event()
        self._fire_position_s:   float = 0.0
        self._fire_next_idx:     int = 0
        self._fire_steps:        Optional[list[FireStep]] = None
        # Fire clock anchor: kept after completion (clock runs), cleared to freeze.
        self._fire_t0:           Optional[float] = None
        # Set by abort(), cleared by fire(); blocks start_sequence() resumes.
        self._fire_invalidated:  bool = False
        self._fire_state:        str = FIRE_IDLE
        # Bumped by every halt; a start or seek begun before one won't go on.
        self._abort_gen:         int = 0
        # Path of the recording a fire started. Only that one is ever stopped
        # automatically; a manual recording never is.
        self._seq_recording:     Optional[str] = None
        self._fire_post_s:       float = 0.0

        self._state_listener = None

        self._stream_thread: Optional[threading.Thread] = None
        self._running = False

    # --------------------------------------------------------
    # Public API (thread-safe)
    # --------------------------------------------------------

    def start(self) -> None:
        """Open the device and start the stream thread."""
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
        """Abort, stop the stream thread, and close the device (which safes it)."""
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
        """Latest snapshot (replaced whole, never mutated after publish)."""
        with self._lock:
            return self._snapshot

    @property
    def event_log(self) -> list[str]:
        """Copy of the event log."""
        with self._lock:
            return list(self._event_log)

    @property
    def thresholds(self) -> dict:
        """Threshold bands in Pa / C / lbf (display only)."""
        with self._lock:
            return dict(self._thresholds)

    def set_state_listener(self, listener) -> None:
        """
        Set a no-arg callback run on the stream thread after each batch and
        error tick. It must be quick; None detaches.
        """
        self._state_listener = listener

    def _notify_state(self) -> None:
        """Run the listener; log and swallow its errors."""
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
        """Attach the logger (it needs channel_specs, so it's built after us)."""
        self._logger = logger

    @property
    def channel_specs(self) -> list[ChannelSpec]:
        """Channel manifest (channels.yaml)."""
        return list(self._channels)

    @property
    def actuator_specs(self) -> list[ActuatorSpec]:
        """Actuator manifest (actuators.yaml)."""
        return list(self._actuators)

    @property
    def unwired_actuators(self) -> list[str]:
        """Ids of actuators with missing pins or step settings; they can't be driven."""
        return [spec.id for spec in self._actuators if not spec.is_wired]

    def _mock_stepper(self, spec: ActuatorSpec) -> ActuatorSpec:
        """Fill a pinned stepper's missing step counts/rate (mock only)."""
        if (spec.type != PULSE_STEPPER or spec.step_dio is None
                or spec.dir_dio is None or spec.is_wired):
            return spec
        self._log(
            f"MOCK: {spec.id} has no step counts/rate; using placeholders "
            f"({_MOCK_STEPS} steps at {_MOCK_PULSE_HZ:.0f} Hz)"
        )
        return replace(
            spec,
            steps_open=spec.steps_open or _MOCK_STEPS,
            steps_close=spec.steps_close or _MOCK_STEPS,
            pulse_freq_hz=spec.pulse_freq_hz or _MOCK_PULSE_HZ,
        )

    def _default_calibration(self) -> dict[str, dict[str, float]]:
        """Per-channel calibration from the type defaults."""
        return {
            spec.cal_ref or spec.id: dict(_DEFAULT_CAL_BY_TYPE[spec.type])
            for spec in self._channels
            if spec.type in _DEFAULT_CAL_BY_TYPE
        }

    def _convert_thresholds_to_pa(self, raw: dict) -> dict:
        """Convert psi bands to Pa for pressure channels (PRESSURE_TYPES)."""
        pressure_ids = {
            spec.id for spec in self._channels
            if spec.type in PRESSURE_TYPES
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

    def _compute_derived(self, latest: dict) -> dict:
        """
        Injector mass flows (kg/s) and O/F from this batch's values.

        Any missing input or non-positive dP gives None for that output.
        """
        def _mdot(side: str):
            cfg = self._derived_cfg.get(side) or {}
            diameter = cfg.get("orifice_diameter_in")
            cd       = cfg.get("discharge_coefficient")
            dp       = latest.get(cfg.get("dp_channel"))
            if diameter is None or cd is None or dp is None:
                return None

            area = orifice_area_m2(diameter, cfg.get("orifice_count", 1))
            if side == "lox":
                return lox_mass_flow_rate(
                    latest.get(cfg.get("inlet_temp_channel")), dp, area, cd
                )
            density = cfg.get("density_kg_m3")
            if density is None:
                return None
            return fuel_mass_flow_rate(dp, area, cd, density)

        lox_mdot  = _mdot("lox")
        fuel_mdot = _mdot("fuel")
        return {
            "lox_mass_flow_rate_kg_s":  lox_mdot,
            "fuel_mass_flow_rate_kg_s": fuel_mdot,
            "mixture_ratio":            mixture_ratio(lox_mdot, fuel_mdot),
        }

    def _channel_status(self, tag: str, value: Optional[float]) -> Optional[str]:
        """
        NOMINAL (in normal), CAUTION (in warning), WARNING (outside),
        UNCONFIGURED (no bands), or None (bands, no value). Display only;
        the strings match Dashboard's readingStatus.ts.
        """
        bands = self._thresholds.get(tag)
        if not isinstance(bands, dict):
            return "UNCONFIGURED"
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
        """Seconds since the last processed batch; inf if none yet."""
        with self._lock:
            last = self._last_batch_perf_time
        if last == 0.0:
            return float("inf")
        return time.perf_counter() - last

    @property
    def is_data_stale(self) -> bool:
        """True if the snapshot is older than _DATA_STALE_THRESHOLD_S."""
        return self.data_age_seconds > _DATA_STALE_THRESHOLD_S

    def write_actuator(self, name: str, state: int) -> None:
        """
        Manual command: 1 = energise, 0 = de-energise. Refused mid-sequence.

        Raises:
            RuntimeError: If a sequence is currently active, or if the
                          actuator is a stepper still executing a move.
        """
        with self._lock:
            if self._sequence_active:
                raise RuntimeError(
                    "Cannot send manual commands while a sequence is running. "
                    "Send ABORT first."
                )

            if self._is_stepper(name):
                reading = self._device.actuator_states().get(name)
                if reading is not None:
                    if reading.state == state:
                        self._log(
                            f"Manual: {name} already "
                            f"{'OPEN' if state else 'CLOSED'} - command ignored "
                            f"rather than re-running the step burst"
                        )
                        return
                    if reading.moving:
                        raise RuntimeError(
                            f"'{name}' is still executing a move - wait for it to "
                            f"finish, or send ABORT to reverse it now"
                        )

            self._device.write_actuator(name, state)
            self._log(f"Manual: {name} -> {'OPEN' if state else 'CLOSED'}",
                      source="manual", actuator=name, state=state)

    def _is_stepper(self, name: str) -> bool:
        """True if `name` is a pulse_stepper."""
        return any(
            spec.id == name and spec.type == PULSE_STEPPER
            for spec in self._actuators
        )

    def deenergize(self) -> None:
        """
        Halt any sequence (no abort.yaml, recording untouched) and de-energise
        every output. Steppers stay where they are.
        """
        self._device.all_safe()
        self._halt_sequence()
        self._device.all_safe()   # again, in case a step landed while halting
        self._log("DEENERGIZE: every output de-energised")

    def tare(self) -> None:
        """
        Tare the load cells on the mean of their last batch.

        Raises:
            RuntimeError: a fire is running (it would zero the thrust).
        """
        with self._lock:
            if self._sequence_active and self._sequence_name == "fire":
                raise RuntimeError("Can't tare during a fire")
            for tag in self._lc_tare:
                mean = self._lc_batch_mean.get(tag)
                if mean is not None:
                    self._lc_tare[tag] += mean
        self._log("Load cells tared")

    def fire(self) -> None:
        """
        Start fire.yaml from T=0 (start_sequence() resumes instead).

        Raises:
            SequenceRefused: fire.yaml can't be loaded or driven.
            RuntimeError: a sequence is already running.
        """
        with self._lock:
            gen = self._abort_gen
        steps, post_s, fire_steps = self._validate_fire()
        if not self._start_sequence(
            "fire", steps, post_s, is_fire=True, reset_fire=fire_steps,
            abort_gen=gen,
        ):
            raise RuntimeError(
                "A sequence is already running, or an abort came in"
            )

    def dump_fire_sequence(self) -> list[dict]:
        """
        fire.yaml's phases for GET /sequence; unwired actuators don't block it.

        Raises:
            SequenceRefused: If the file can't be parsed.
        """
        seq_path = os.path.join(self._sequence_dir, "fire.yaml")
        _steps, _post_s, fire_steps = self._load_or_refuse(
            "fire", seq_path, self._load_fire_sequence
        )
        return [
            {
                "name":     step.name,
                "start_s":  step.start_t,
                "actions":  [list(a) for a in step.actions],
            }
            for step in fire_steps
        ]

    def sequence_status(self) -> dict:
        """
        Fire position for every /sequence/* response: step, state, server
        epoch ms and sequence ms. The clock runs on after completion and
        freezes on pause or abort.
        """
        with self._lock:
            if self._fire_t0 is not None:
                elapsed = time.perf_counter() - self._fire_t0
            else:
                elapsed = self._fire_position_s
            fire_steps = self._fire_steps
            state = self._fire_state
        step = self._step_name_for_position(fire_steps, elapsed) if fire_steps else None
        return {
            "step":              step,
            "state":             state,
            "server_time_ms":    int(time.time() * 1000),
            "sequence_time_ms":  int(round(elapsed * 1000)),
        }

    def start_sequence(self) -> dict:
        """
        Resume fire.yaml from the current position (like fire(), no reset).

        Raises:
            SequenceRefused: fire.yaml can't be loaded or driven.
            RuntimeError: a sequence (fire or abort) is already active, or
                          the position was invalidated by an abort.
        """
        with self._lock:
            if self._fire_invalidated:
                raise RuntimeError(
                    "Sequence was aborted - use /fire to restart from the top"
                )
            gen = self._abort_gen
        steps, post_s, fire_steps = self._validate_fire()
        with self._lock:
            self._fire_steps = fire_steps
            start_idx        = self._fire_next_idx
            start_elapsed    = self._fire_position_s
        started = self._start_sequence(
            "fire", steps, post_s, is_fire=True,
            start_idx=start_idx, start_elapsed=start_elapsed, abort_gen=gen,
        )
        if not started:
            raise RuntimeError("A sequence is already active")
        return self.sequence_status()

    def stop_sequence(self) -> dict:
        """
        Pause fire in place. Both igniter lines are de-energised; every other
        output and the recording are left as they are.

        Raises:
            RuntimeError: no fire is running, or it's past its last action.
        """
        with self._lock:
            if not (self._sequence_active and self._sequence_name == "fire"):
                raise RuntimeError("No fire sequence is active")
            self._refuse_if_actions_done()
        self._pause_flag.set()
        thread = self._sequence_thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._pause_flag.clear()
        wired = {s.id for s in self._actuators if s.is_wired}
        for name in _IGNITER_IDS:
            if name in wired:
                try:
                    self._device.write_actuator(name, 0)
                    self._log(f"Pause: {name} -> CLOSED",
                              source="pause", actuator=name, state=0)
                except Exception as exc:
                    self._log(f"Pause: {name} write failed: {exc}")
        return self.sequence_status()

    def set_sequence_time(self, seconds: float) -> dict:
        """
        Move the fire clock. Stopped: pointer only, no hardware writes.
        Running: forward only; skipped actions fire at once, then it ticks on.

        Raises:
            SequenceRefused: fire.yaml can't be loaded or driven.
            RuntimeError: while running, seconds is behind the current
                          position. While stopped, never raised.
        """
        seconds = max(0.0, float(seconds))
        with self._lock:
            active = self._sequence_active and self._sequence_name == "fire"
        if active:
            return self._jump_while_active(seconds)

        steps, _post_s, fire_steps = self._validate_fire()
        with self._lock:
            self._fire_steps      = fire_steps
            self._fire_position_s = seconds
            self._fire_next_idx   = sum(1 for t, _, _ in steps if t < seconds)
        return self.sequence_status()

    def _jump_while_active(self, target_seconds: float) -> dict:
        """
        Pause, fire the skipped actions, resume. The position fields are only
        safe to advance here once the old thread has exited.

        Raises:
            RuntimeError: target_seconds is behind the current live
                          position.
        """
        with self._lock:
            current = (
                time.perf_counter() - self._fire_t0
                if self._fire_t0 is not None else self._fire_position_s
            )
            if target_seconds < current:
                raise RuntimeError(
                    "Cannot jump backward while running - stop the sequence first"
                )
            self._refuse_if_actions_done()
            gen = self._abort_gen

        self._pause_flag.set()
        thread = self._sequence_thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._pause_flag.clear()

        steps, post_s, fire_steps = self._validate_fire()
        with self._lock:
            self._fire_steps = fire_steps
            idx = self._fire_next_idx

        while idx < len(steps) and steps[idx][0] <= target_seconds:
            if self._abort_gen != gen:
                raise RuntimeError("Aborted during the seek")
            target_t, act_name, state = steps[idx]
            try:
                self._device.write_actuator(act_name, state)
                self._log(
                    f"[fire] T+{target_t:.2f}s  "
                    f"{act_name} -> {'OPEN' if state else 'CLOSED'} (jump)",
                    source="fire", actuator=act_name, state=state,
                )
            except Exception as exc:
                self._log(f"Actuator write error: {exc}")
            idx += 1
            with self._lock:
                self._fire_next_idx   = idx
                self._fire_position_s = target_t

        started = self._start_sequence(
            "fire", steps, post_s, is_fire=True,
            start_idx=self._fire_next_idx, start_elapsed=target_seconds,
            abort_gen=gen,
        )
        if not started:
            raise RuntimeError("A sequence is already active")
        return self.sequence_status()

    def jump_to_step(self, step_name: str) -> dict:
        """
        set_sequence_time() to a named step's start.

        Raises:
            SequenceRefused: fire.yaml can't be loaded or driven.
            RuntimeError: a fire sequence is currently active - stop it
                          first.
            KeyError: step_name doesn't match any step in fire.yaml.
        """
        _steps, _post_s, fire_steps = self._validate_fire()
        match = next((s for s in fire_steps if s.name == step_name), None)
        if match is None:
            raise KeyError(step_name)
        return self.set_sequence_time(match.start_t)

    def _refuse_if_actions_done(self) -> None:
        """
        Raise if the fire has applied every action (only the post-record
        window is left, which can't be paused or seeked). Caller holds _lock.
        """
        total = sum(len(step.actions) for step in self._fire_steps or [])
        if self._fire_next_idx >= total:
            raise RuntimeError(
                "Fire has applied every action; only post-record remains"
            )

    @staticmethod
    def _step_name_for_position(
        fire_steps: list[FireStep], elapsed: float
    ) -> Optional[str]:
        """Last phase with start_t <= elapsed, or None before the first."""
        name = None
        for step in fire_steps:
            if step.start_t <= elapsed:
                name = step.name
            else:
                break
        return name

    def _current_fire_step_name(self) -> Optional[str]:
        """Current fire step name; caller holds self._lock."""
        if self._fire_steps is None:
            return None
        if self._fire_t0 is not None:
            elapsed = time.perf_counter() - self._fire_t0
        else:
            elapsed = self._fire_position_s
        return self._step_name_for_position(self._fire_steps, elapsed)

    def abort(self) -> None:
        """
        Halt any sequence and run abort.yaml, or, if it can't run, de-energise
        everything and close the mains if possible. Never raises; doesn't wait
        for the control lock. abort.yaml's T+0 off-lines (igniter, presses)
        drop before any wait. A no-op while abort.yaml is already running.
        """
        with self._lock:
            running = self._sequence_thread
            if (self._sequence_active and self._sequence_name == "abort"
                    and running is not None and running.is_alive()):
                self._log("ABORT: abort.yaml already in progress")
                return

        plan = self._abort_plan()
        if plan is None:
            self._device.all_safe()
            self._halt_sequence()
            self._device.all_safe()
            self._log("ABORT: abort.yaml can't run, hardware -> all safe")
            self._close_mains()
            self._finish_seq_recording()
            return

        steps, post_s = plan
        first_off = self._first_offs(steps)
        self._write_offs(first_off)
        self._halt_sequence()
        self._write_offs(first_off)   # again, in case a step landed while halting
        self._start_sequence("abort", steps, post_s, is_fire=False, force=True)

    def _abort_plan(self) -> Optional[tuple]:
        """abort.yaml's (steps, post_s), or None if it's missing or refused."""
        seq_path = os.path.join(self._sequence_dir, "abort.yaml")
        if not os.path.exists(seq_path):
            self._log("ABORT: no abort.yaml found")
            return None
        try:
            return self._validate_sequence("abort", seq_path)
        except SequenceRefused as exc:
            self._log(str(exc))
            return None

    def _first_offs(self, steps: list[tuple[float, str, int]]) -> list[str]:
        """binary_dio ids abort.yaml de-energises at T+0."""
        binary = {
            s.id for s in self._actuators
            if s.type == BINARY_DIO and s.is_wired
        }
        return [name for t, name, state in steps
                if t == 0.0 and state == 0 and name in binary]

    def _write_offs(self, names: list[str]) -> None:
        for name in names:
            try:
                self._device.write_actuator(name, 0)
                self._log(f"ABORT: {name} -> CLOSED",
                          source="abort", actuator=name, state=0)
            except Exception as exc:
                self._log(f"ABORT: {name} write failed: {exc}")

    def _close_mains(self) -> None:
        """Best effort: command each stepper main closed unless it already is."""
        states = self._device.actuator_states()
        for spec in self._actuators:
            if spec.type != PULSE_STEPPER:
                continue
            if not spec.is_wired:
                self._log(f"ABORT: can't close {spec.id}: step settings not set")
                continue
            reading = states.get(spec.id)
            if reading is not None and reading.state == 0 and not reading.moving:
                continue
            try:
                self._device.write_actuator(spec.id, 0)
                self._log(f"ABORT: {spec.id} -> CLOSED",
                          source="abort", actuator=spec.id, state=0)
            except Exception as exc:
                self._log(f"ABORT: {spec.id} close failed: {exc}")

    def _stop_seq_recording(self) -> None:
        """Stop the recording a fire started, if it's still the current one."""
        path, self._seq_recording = self._seq_recording, None
        if (path and self._logger and self._logger.is_recording
                and self._logger.current_path == path):
            self._logger.stop_recording()

    def _finish_seq_recording(self) -> None:
        """Stop the fire's recording after its post-record time, off-thread."""
        if self._seq_recording and self._logger:
            timer = threading.Timer(self._fire_post_s, self._stop_seq_recording)
            timer.daemon = True
            timer.start()

    def _halt_sequence(self) -> None:
        """Stop any sequence thread; blocks fire resumes until the next fire()."""
        with self._lock:
            self._abort_gen += 1
            self._fire_invalidated = True
            # A paused fire has no thread to record the abort itself.
            if self._fire_state == FIRE_PAUSED:
                self._fire_state = FIRE_ABORTED
        self._abort_flag.set()

        old_thread = self._sequence_thread
        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=0.5)
            if old_thread.is_alive():
                self._log(
                    f"WARNING: '{old_thread.name}' did not exit within 0.5s "
                    f"of a halt - waiting longer"
                )
                old_thread.join(timeout=2.0)
                if old_thread.is_alive():
                    self._log(
                        f"WARNING: '{old_thread.name}' still alive after 2.5s "
                        f"total - going on anyway. Actuator states may be "
                        f"contested by both threads."
                    )

        self._abort_flag.clear()

    def update_calibration(self, tag: str, slope: float, intercept: float) -> None:
        """
        Set a channel's slope/intercept in memory (psi or lbf domain).

        Raises:
            KeyError: tag isn't a calibrated channel's cal_ref.
        """
        with self._lock:
            if tag not in self._cal:
                raise KeyError(tag)
            self._cal[tag] = {"slope": slope, "intercept": intercept}
        self._log(f"Cal updated: {tag} slope={slope} intercept={intercept}")

    def save_calibration(self) -> None:
        """
        Update calibration.json's PT/LC entries; other keys are kept.

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
            for spec in self._channels if spec.type in PRESSURE_TYPES
        }
        lc_tags = {
            spec.cal_ref or spec.id
            for spec in self._channels if spec.type == LC_DIRECT
        }
        try:
            with open(self._cal_path) as f:
                output = json.load(f)
        except (OSError, ValueError):
            output = {}
        output.setdefault("PT", {})
        output.setdefault("LC", {})

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
        """Read, poll, process; reconnect after repeated read failures."""
        try:
            actual_hz = self._device.start_stream()
            self._log(f"Stream running at {actual_hz} Hz")
            self._check_rate(actual_hz)

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

                    # No batch to push now, so push the failure.
                    now_err = time.perf_counter()
                    if (consecutive_errors == 1
                            or now_err - last_error_notify >= _ERROR_NOTIFY_INTERVAL_S):
                        last_error_notify = now_err
                        self._notify_state()

                    if consecutive_errors >= _MAX_CONSECUTIVE_READ_ERRORS:
                        consecutive_errors = 0
                        if self._running:
                            self._halt_on_comms_loss()
                            self._reconnect_device()
                    elif self._running:
                        time.sleep(0.1)
                    continue

                now = time.perf_counter()
                if now - self._last_cjc_read >= _CJC_INTERVAL_S:
                    self._poll_out_of_band()
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

    def _check_rate(self, actual_hz: int) -> None:
        """Warn if the T7 had to drop below the SysRD minimum."""
        if actual_hz < _MIN_STREAM_HZ:
            self._log(
                f"WARNING: stream at {actual_hz} Hz, below the "
                f"{_MIN_STREAM_HZ} Hz minimum (SysRD 2.5.4)"
            )

    def _check_backlog(self, backlog: dict[str, int]) -> None:
        """Log once when the host falls behind the stream, and once on recovery."""
        limit = _BACKLOG_WARN_S * self._device.stream_rate_hz
        behind = max(backlog.values()) > limit
        if behind and not self._backlog_warned:
            self._log(
                f"WARNING: host falling behind the stream "
                f"(backlog device {backlog['device']}, LJM {backlog['ljm']} scans)"
            )
        elif not behind and self._backlog_warned:
            self._log("Stream backlog back to normal")
        self._backlog_warned = behind

    def _halt_on_comms_loss(self) -> None:
        """
        Treat comms loss mid-sequence as an abort: halt it (no abort.yaml, the
        T7 is unreachable; the reconnect safes the outputs) and end the fire's
        recording after its post-record time.
        """
        with self._lock:
            name = self._sequence_name if self._sequence_active else None
        if name is None:
            return
        self._log(f"Comms lost during '{name}': treated as an abort")
        self._halt_sequence()
        self._finish_seq_recording()

    def _sleep_interruptible(self, duration: float) -> None:
        """Sleep in 0.1 s steps so stop() is noticed."""
        deadline = time.perf_counter() + duration
        while self._running and time.perf_counter() < deadline:
            time.sleep(min(0.1, deadline - time.perf_counter()))

    def _reconnect_device(self) -> None:
        """
        close/open/restart with backoff until it works or stop() is called.
        open() safes the outputs, so the reported states match afterwards.
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
                    break   # stop() during backoff

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
                self._check_rate(actual_hz)
                return
            except Exception as exc:
                self._log(f"Reconnect attempt {attempt} failed: {exc}")

        self._log("Reconnect loop exiting - engine is stopping")

    def _channel_calibration(self, spec: ChannelSpec) -> dict[str, float]:
        """A channel's calibration, or its type default."""
        return self._cal.get(
            spec.cal_ref or spec.id,
            _DEFAULT_CAL_BY_TYPE.get(spec.type, {"slope": 1.0, "intercept": 0.0}),
        )

    def _poll_out_of_band(self) -> None:
        """
        Read CJC and counters (the watchdog keep-alive). A failure clears the
        value and logs once per outage.
        """
        try:
            self._cjc_celsius = self._device.read_cjc_celsius()
            if self._cjc_failing:
                self._log("CJC read recovered")
            self._cjc_failing = False
        except Exception as exc:
            self._cjc_celsius = None
            if not self._cjc_failing:
                self._log(f"CJC read FAILED, thermocouples report no value: {exc}")
            self._cjc_failing = True

        try:
            self._counters = dict(self._device.read_counters())
            if self._counters_failing:
                self._log("Photogate counter read recovered")
            self._counters_failing = False
        except Exception as exc:
            self._counters = {}
            if not self._counters_failing:
                self._log(f"Photogate counter read FAILED: {exc}")
            self._counters_failing = True

    def _scale_channel(
        self, spec: ChannelSpec, raw: list[float], cjc_c: Optional[float]
    ) -> list[Optional[float]]:
        """Raw volts to engineering units, by channel type."""
        c = self._channel_calibration(spec)

        if spec.type in PRESSURE_TYPES:
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
                None if v == _MISSING_SAMPLE or cjc_c is None
                else thermocouple_type_t(v, cjc_c)
                for v in raw
            ]

        return [None] * len(raw)

    def _process_batch(self, batch: dict[str, list[float]]) -> None:
        """
        Calibrate a batch, publish a new snapshot, then queue CSV rows.

        Every manifest channel is reported; inactive or unwired ones carry no
        value, so the key set matches the manifest.
        """
        scan_times = batch.get("scan_times", [])
        n_scans    = len(scan_times)
        if n_scans == 0:
            return

        cjc_c = self._cjc_celsius
        now_epoch = time.time()
        backlog = {
            "device": int(batch.get("backlog_device", [0])[0]),
            "ljm":    int(batch.get("backlog_ljm", [0])[0]),
        }
        self._check_backlog(backlog)

        # Per-scan values for the CSV.
        eng_rows: dict[str, list[Optional[float]]] = {}
        # Snapshot values: the last scan (TCs: batch average).
        latest: dict[str, Optional[float]] = {}

        for spec in self._channels:
            if not spec.active or not spec.is_streamed:
                continue
            raw = batch.get(spec.id)
            if raw is None:
                continue

            eng_rows[spec.id] = self._scale_channel(spec, raw, cjc_c)
            if spec.type == LC_DIRECT:
                values = [v for v in eng_rows[spec.id] if v is not None]
                if values:
                    with self._lock:
                        self._lc_batch_mean[spec.id] = sum(values) / len(values)

            if spec.type == TC_DIFFERENTIAL:
                # Average first: per-scan noise dominates a mV signal.
                valid = [v for v in raw if v != _MISSING_SAMPLE]
                latest[spec.id] = (
                    thermocouple_type_t(sum(valid) / len(valid), cjc_c)
                    if valid and cjc_c is not None else None
                )
            else:
                latest[spec.id] = eng_rows[spec.id][-1]

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

        # One locked read, so a frame can't mix before/after a transition.
        with self._lock:
            seq_active = self._sequence_active
            seq_name   = self._sequence_name
            seq_step   = self._current_fire_step_name()
            seq_state  = self._fire_state

        new_snap = EngineState()
        object.__setattr__(new_snap, "timestamp",       scan_times[-1])
        object.__setattr__(new_snap, "streaming",       True)
        object.__setattr__(new_snap, "stream_hz",       self._device.stream_rate_hz)
        object.__setattr__(new_snap, "using_mock",      USING_MOCK)
        object.__setattr__(new_snap, "cjc_celsius",     cjc_c)
        object.__setattr__(new_snap, "backlog",         backlog)
        object.__setattr__(new_snap, "channels",        channels)
        object.__setattr__(new_snap, "derived",         self._compute_derived(latest))
        object.__setattr__(new_snap, "actuators",       dict(self._device.actuator_states()))
        object.__setattr__(new_snap, "sequence_active", seq_active)
        object.__setattr__(new_snap, "sequence_name",   seq_name)
        object.__setattr__(new_snap, "sequence_step",   seq_step)
        object.__setattr__(new_snap, "sequence_state",  seq_state)

        with self._lock:
            self._snapshot = new_snap
            self._last_batch_perf_time = time.perf_counter()

        # Push before the slower CSV work.
        self._notify_state()

        # One row per scan; order matches Logger._header().
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

    def _validate_fire(
        self,
    ) -> tuple[list[tuple[float, str, int]], float, list[FireStep]]:
        """Load and validate fire.yaml; re-read each call so edits apply live."""
        return self._validate_sequence(
            "fire",
            os.path.join(self._sequence_dir, "fire.yaml"),
            self._load_fire_sequence,
        )

    def _load_or_refuse(self, name: str, path: str, loader) -> tuple:
        """Run a loader, turning any parse failure into SequenceRefused."""
        try:
            return loader(path)
        except Exception as exc:
            raise SequenceRefused(
                f"Sequence '{name}' could not be loaded: {exc}"
            ) from exc

    def _validate_sequence(self, name: str, path: str, loader=None) -> tuple:
        """
        Load a sequence and check every actuator in it can be driven.

        Args:
            loader: _load_sequence (flat, default) or _load_fire_sequence.
                    Its tuple is returned; element 0 is the flat step list.

        Raises:
            SequenceRefused: If the file cannot be parsed, or any step
                             commands an actuator this cart cannot drive.
        """
        loaded = self._load_or_refuse(name, path, loader or self._load_sequence)

        undrivable = self._undrivable_steps(loaded[0])
        if undrivable:
            raise SequenceRefused(
                f"Sequence '{name}' REFUSED: cannot drive "
                f"{', '.join(undrivable)} - not wired in actuators.yaml. "
                f"Every step for these would have failed and the sequence "
                f"would have reported completion having moved nothing."
            )
        return loaded

    def _start_sequence(
        self,
        name: str,
        steps: list[tuple[float, str, int]],
        post_s: float,
        is_fire: bool,
        force: bool = False,
        start_idx: int = 0,
        start_elapsed: float = 0.0,
        reset_fire: Optional[list[FireStep]] = None,
        abort_gen: Optional[int] = None,
    ) -> bool:
        """
        Start a sequence thread. Returns False if one is alive, unless force
        (abort), or if a halt came in since abort_gen was read.
        start_idx/start_elapsed resume a fire mid-timeline; reset_fire
        restarts the fire position from 0 with these steps.
        """
        # A thread that has marked itself done can still read as alive for a
        # moment; let it exit rather than refuse.
        prev = self._sequence_thread
        if not force and prev is not None and prev.is_alive():
            with self._lock:
                finishing = not self._sequence_active
            if finishing:
                prev.join(timeout=0.5)

        with self._lock:
            if abort_gen is not None and abort_gen != self._abort_gen:
                return False
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
            # Set here, not in the thread, so sequence_status() can't race it.
            t0 = time.perf_counter() - start_elapsed
            self._sequence_t0 = t0
            if is_fire:
                self._fire_t0 = t0
                self._fire_state = FIRE_RUNNING
                if reset_fire is not None:
                    self._fire_steps       = reset_fire
                    self._fire_position_s  = 0.0
                    self._fire_next_idx    = 0
                    self._fire_invalidated = False

        self._abort_flag.clear()
        self._pause_flag.clear()
        self._sequence_thread = threading.Thread(
            target=self._run_sequence,
            args=(name, steps, post_s, is_fire, start_idx, t0),
            name=f"daq-seq-{name}",
            daemon=True,
        )
        self._sequence_thread.start()
        return True

    def _run_sequence(
        self,
        name: str,
        steps: list[tuple[float, str, int]],
        post_s: float,
        is_fire: bool,
        start_idx: int,
        t0: float,
    ) -> None:
        """Run validated steps against t0; fire checks abort/pause each tick."""
        self._log(f"Sequence '{name}' starting")

        if is_fire and self._logger and not self._logger.is_recording:
            self._seq_recording = self._logger.start_recording(prefix="hotfire")
        if is_fire:
            self._fire_post_s = post_s

        idx = start_idx

        while idx < len(steps):
            if self._abort_flag.is_set():
                self._log(f"Sequence '{name}' halted")
                if is_fire:
                    with self._lock:
                        self._fire_state = FIRE_ABORTED
                self._sequence_paused()
                return

            if self._pause_flag.is_set() and is_fire:
                self._log(f"Sequence '{name}' paused at T+{steps[idx][0]:.2f}s")
                with self._lock:
                    self._fire_state = FIRE_PAUSED
                self._sequence_paused()
                return

            elapsed   = time.perf_counter() - t0
            target_t, act_name, state = steps[idx]

            if elapsed >= target_t:
                try:
                    self._device.write_actuator(act_name, state)
                    self._log(
                        f"[{name}] T+{target_t:.2f}s  "
                        f"{act_name} -> {'OPEN' if state else 'CLOSED'}",
                        source=name, actuator=act_name, state=state,
                    )
                except Exception as exc:
                    self._log(f"Actuator write error: {exc}")
                idx += 1
                if is_fire:
                    with self._lock:
                        self._fire_next_idx   = idx
                        self._fire_position_s = target_t
            else:
                time.sleep(0.0005)

        if self._logger and post_s > 0:
            self._log(f"Sequence '{name}' complete, recording {post_s:.1f} s post-data")
            deadline = time.perf_counter() + post_s
            while time.perf_counter() < deadline:
                if self._abort_flag.is_set():
                    # Keep recording: abort.yaml runs next and stops it.
                    self._log("Post-record window interrupted by abort")
                    if is_fire:
                        with self._lock:
                            self._fire_state = FIRE_ABORTED
                    self._sequence_paused()
                    return
                time.sleep(0.05)

        self._stop_seq_recording()

        self._log(f"Sequence '{name}' finished")
        if is_fire:
            self._sequence_completed()
        else:
            self._sequence_paused()

    def _sequence_completed(self) -> None:
        """Mark fire completed; _fire_t0 stays so the clock keeps running."""
        with self._lock:
            self._sequence_active = False
            self._sequence_name   = ""
            self._sequence_t0     = None
            self._fire_state      = FIRE_COMPLETED

    def _sequence_paused(self) -> None:
        """End the sequence and freeze the fire clock at _fire_position_s."""
        with self._lock:
            self._sequence_active = False
            self._sequence_name   = ""
            self._sequence_t0     = None
            self._fire_t0         = None

    def _undrivable_steps(
        self, steps: list[tuple[float, str, int]]
    ) -> list[str]:
        """
        Ids a sequence commands that are unknown or unwired. Without this
        check the writes fail one by one and the sequence still "completes".
        """
        drivable = {spec.id for spec in self._actuators if spec.is_wired}
        return sorted({name for _, name, _ in steps if name not in drivable})

    def _load_sequence(
        self, path: str
    ) -> tuple[list[tuple[float, str, int]], float]:
        """Flat sequence file -> (time-sorted steps, post_record_seconds)."""
        with open(path) as f:
            data = yaml.safe_load(f)

        post_s = float(data.get("post_record_seconds", 5.0))
        raw    = data.get("steps", [])
        steps  = [_parse_action(s) for s in raw]
        steps.sort(key=lambda s: s[0])
        return steps, post_s

    def _load_fire_sequence(
        self, path: str
    ) -> tuple[list[tuple[float, str, int]], float, list[FireStep]]:
        """
        fire.yaml -> (flat time-sorted steps, post_record_seconds, phases).

        Phase start times must not decrease in file order, so each step name
        maps to one position.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        post_s    = float(data.get("post_record_seconds", 5.0))
        raw_steps = data.get("steps", [])

        flat: list[tuple[float, str, int]] = []
        fire_steps: list[FireStep] = []
        last_start_t = float("-inf")
        for phase in raw_steps:
            name = str(phase["name"])
            raw_actions = phase.get("actions", [])
            if not raw_actions:
                raise ValueError(f"step '{name}' has no actions")

            actions = tuple(_parse_action(a) for a in raw_actions)
            flat.extend(actions)

            start_t = min(t for t, _, _ in actions)
            if start_t < last_start_t:
                raise ValueError(
                    f"step '{name}' starts at {start_t}s, before the "
                    f"preceding step's start time {last_start_t}s"
                )
            last_start_t = start_t
            fire_steps.append(FireStep(name=name, start_t=start_t, actions=actions))

        flat.sort(key=lambda s: s[0])
        return flat, post_s, fire_steps

    # --------------------------------------------------------
    # Calibration persistence
    # --------------------------------------------------------

    def _load_calibration(self, path: str) -> None:
        """Merge calibration.json over the defaults; unknown tags are logged."""
        try:
            with open(path) as f:
                saved = json.load(f)
            unknown = []
            for section in ("PT", "LC"):
                for tag, coeffs in saved.get(section, {}).items():
                    if tag in self._cal:
                        self._cal[tag].update(coeffs)
                    else:
                        unknown.append(tag)
            self._log(f"Calibration loaded from {path}")
            if unknown:
                self._log(
                    f"Calibration tags not in the manifest, ignored: "
                    f"{', '.join(sorted(unknown))}"
                )
        except Exception as exc:
            self._log(f"Calibration load warning: {exc} - using defaults")

    # --------------------------------------------------------
    # Internal logging (SR 3.5)
    # --------------------------------------------------------

    def _log(
        self, message: str, source: str = "engine",
        actuator: str = "", state: object = "",
    ) -> None:
        """
        Timestamp, append to the event log, and print. While recording, also
        write it to the events CSV on the recording's time_s clock.
        """
        entry = f"{time.strftime('%H:%M:%S')} {message}"
        with self._lock:
            self._event_log.append(entry)
        print(f"[ENGINE] {entry}")
        logger = self._logger
        if logger is not None and logger.is_recording and hasattr(self, "_device"):
            logger.write_event([
                f"{self._device.stream_time_s:.6f}", source, actuator, state, message,
            ])
