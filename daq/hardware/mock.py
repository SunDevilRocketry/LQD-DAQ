"""
daq/hardware/mock.py

Simulated LabJack T7 for hardware-free runs and tests. Same public
interface as hardware/interface.py; keep the two in sync.

Signals: sine drift plus noise; load cells ramp once both mains are open;
counters advance while their stepper moves.
"""

from __future__ import annotations

import math
import time
import threading
from typing import Sequence

from daq.manifest import (
    ActuatorReading,
    ActuatorSpec,
    ChannelSpec,
    BINARY_DIO,
    PULSE_STEPPER,
    PT_DIRECT,
    PT_DIFFERENTIAL,
    TC_DIFFERENTIAL,
    LC_DIRECT,
    PHOTOGATE_COUNTER,
    VOLTAGE,
)


# -- Sensor Simulation Parameters -----------------------------
# V = base + amplitude * sin(2pi * t / period) + noise

# ~200 psi at 400 psi/V; each PT offset so the traces don't overlap.
_PT_BASE_V     = 0.50
_PT_OFFSET_V   = 0.04
_PT_AMP_V      = 0.020
_PT_PERIOD_S   = 12.0
_PT_NOISE_V    = 0.002

# ~0.6 psi at 5.80151 psi/V (0-0.4 MPa over 0-10 V).
_DPT_BASE_V    = 0.10
_DPT_OFFSET_V  = 0.02     # separates dpt0 from dpt1
_DPT_AMP_V     = 0.015
_DPT_PERIOD_S  = 9.0
_DPT_NOISE_V   = 0.001

# ~-177 C (Type T) against the mock's 25 C cold junction.
_TC_BASE_V     = -0.006200
_TC_AMP_V      = 0.000050
_TC_PERIOD_S   = 25.0
_TC_NOISE_V    = 0.000005

# Fire ramp to ~500 lbf at 440.92 lbf/V.
_LC_NOISE_V       = 0.003
_LC_FIRE_V        = 1.134
_LC_RAMP_S        = 0.5
_LC_OSC_V         = 0.1
_LC_OSC_PERIOD_S  = 0.8
_LC_FIRE_NOISE_V  = 0.005

# Igniter sense node, calculated from the igniter schematic: no e-match
# fitted (until testing), armed (~battery), firing.
_IGN_SAFE_V    = 4.4
_IGN_ARMED_V   = 9.0
_IGN_FIRING_V  = 2.0
_IGN_NOISE_V   = 0.01

_CJC_BASE_C  = 25.0
_CJC_DRIFT_C = 1.0               # slow +/-1 C swing, like a warming T7

# -- Actuator Simulation Parameters ---------------------------
_COUNTS_PER_SECOND = 200.0       # photogate edges while moving

# -- Timing Parameters ----------------------------------------
_STREAM_HZ         = 500


class MockLabJack:
    """Simulated LabJack T7; same public interface as LabJackT7."""

    def __init__(
        self,
        channels:  Sequence[ChannelSpec]  = (),
        actuators: Sequence[ActuatorSpec] = (),
        stream_hz: int = _STREAM_HZ,
    ) -> None:
        self._stream_hz      = stream_hz
        self._scans_per_read = max(1, stream_hz // 20)   # 20 batches/s, like the T7
        self._batch_duration = self._scans_per_read / stream_hz

        self._lock = threading.Lock()

        self._stream_channels = [
            spec for spec in channels if spec.active and spec.is_streamed
        ]
        self._counter_channels = [
            spec for spec in channels
            if spec.active and spec.type == PHOTOGATE_COUNTER
        ]
        # Per-PT voltage offset, assigned by manifest order.
        self._pt_index = {
            spec.id: i for i, spec in enumerate(
                s for s in self._stream_channels if s.type == PT_DIRECT
            )
        }
        self._dpt_index = {
            spec.id: i for i, spec in enumerate(
                s for s in self._stream_channels if s.type == PT_DIFFERENTIAL
            )
        }

        self._actuator_specs: dict[str, ActuatorSpec] = {
            spec.id: spec for spec in actuators
        }
        self._actuators: dict[str, int] = {spec.id: 0 for spec in actuators}
        self._steppers = [
            spec for spec in actuators if spec.type == PULSE_STEPPER
        ]

        # Per stepper: the move is in flight until perf_counter() passes this.
        self._move_deadline: dict[str, float] = {}
        # Accumulated simulated photogate counts, keyed by channel id.
        self._counts: dict[str, float] = {spec.id: 0.0 for spec in self._counter_channels}
        self._last_count_update: float = 0.0

        self._connected      = False
        self._streaming      = False
        self._stream_start: float = 0.0
        self._next_batch_time: float = 0.0
        # Perf-counter instant both mains last became open, or None at rest.
        self._fire_start: float | None = None

        # Fixed seed: reproducible noise.
        self._rng_state: int = 0xDEADBEEF

    # --------------------------------------------------------
    # Lifecycle Management
    # --------------------------------------------------------

    def open(self) -> None:
        """Simulate opening a connection; safes outputs like the T7."""
        self._safe_outputs()
        self._connected = True
        print("[MOCK] LabJack T7 connected (simulated)")

    def close(self) -> None:
        """Simulate closing the connection; safes outputs like the T7."""
        self._safe_outputs()
        self._connected  = False
        self._streaming  = False
        print("[MOCK] LabJack T7 disconnected (simulated)")

    def start_stream(self) -> int:
        """Start the simulated stream; returns the rate in Hz."""
        now = time.perf_counter()
        if not self._stream_start:   # once, so time runs on across restarts
            self._stream_start = now
        self._next_batch_time = now + self._batch_duration
        self._streaming       = True
        print(f"[MOCK] Stream started at {self._stream_hz} Hz "
              f"({self._scans_per_read} scans/read)")
        return self._stream_hz

    def stop_stream(self) -> None:
        """Simulate stopping the data stream."""
        self._streaming = False
        print("[MOCK] Stream stopped")

    # --------------------------------------------------------
    # Data acquisition
    # --------------------------------------------------------

    def stream_read(self) -> dict[str, list[float]]:
        """
        Block until the next batch is due, then return it (same shape as the T7).

        Raises:
            RuntimeError: If called before start_stream().
        """
        if not self._streaming:
            raise RuntimeError("stream_read() called before start_stream()")

        # Sleep until the next batch boundary
        now = time.perf_counter()
        sleep_duration = self._next_batch_time - now
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self._next_batch_time += self._batch_duration

        t_elapsed = time.perf_counter() - self._stream_start
        batch: dict[str, list[float]] = {tag: [] for tag in self.sensor_tags}
        batch["scan_times"] = []
        batch["backlog_device"] = [0]   # type: ignore[assignment]
        batch["backlog_ljm"]    = [0]   # type: ignore[assignment]

        fire_elapsed = self._fire_elapsed()

        for i in range(self._scans_per_read):
            t = t_elapsed - (self._scans_per_read - 1 - i) / self._stream_hz
            batch["scan_times"].append(t)

            for spec in self._stream_channels:
                batch[spec.id].append(self._sample(spec, t, fire_elapsed))

        return batch

    def read_cjc_celsius(self) -> float:
        """Cold junction C, near 25 C."""
        t = time.perf_counter() - self._stream_start
        drift = _CJC_DRIFT_C * math.sin(2 * math.pi * t / 60.0)
        noise = self._noise() * 0.02
        return _CJC_BASE_C + drift + noise

    def read_counters(self) -> dict[str, float]:
        """Photogate counts by channel id; each advances while its stepper moves."""
        now = time.perf_counter()
        with self._lock:
            dt = now - self._last_count_update if self._last_count_update else 0.0
            self._last_count_update = now

            for spec in self._steppers:
                if spec.position_feedback not in self._counts:
                    continue
                if self._move_deadline.get(spec.id, 0.0) > now:
                    self._counts[spec.position_feedback] += _COUNTS_PER_SECOND * dt

            return dict(self._counts)

    # --------------------------------------------------------
    # Actuator control
    # --------------------------------------------------------

    def write_actuator(self, name: str, state: int) -> None:
        """
        Energise (1) or de-energise (0); a stepper also starts a timed move.

        Raises:
            KeyError:     If name is not a recognised actuator.
            RuntimeError: If it isn't fully wired, like the T7.
        """
        spec = self._actuator_specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown actuator: '{name}'")
        if not spec.is_wired:
            raise RuntimeError(
                f"Actuator '{name}' is not fully specified in actuators.yaml "
                f"- fill in its pin/step fields before commanding it"
            )

        with self._lock:
            self._actuators[name] = state
            if spec.type == PULSE_STEPPER:
                self._move_deadline[name] = (
                    time.perf_counter() + self._move_duration(spec, state)
                )
        self._refresh_fire_state()
        print(f"[MOCK] {name} -> {'OPEN' if state else 'CLOSED'}")

    def read_actuator(self, name: str) -> int:
        """Commanded state: 1 = energised, 0 = de-energised."""
        with self._lock:
            return self._actuators[name]

    def all_safe(self) -> None:
        """Mirrors LabJackT7.all_safe(): binary_dio to 0, steppers keep their state."""
        self._safe_outputs()
        print("[MOCK] All actuators -> SAFE/CLOSED")

    def _safe_outputs(self) -> None:
        """binary_dio to 0 and moves cancelled; steppers keep their state."""
        with self._lock:
            for name, spec in self._actuator_specs.items():
                if spec.type == BINARY_DIO:
                    self._actuators[name] = 0
            self._move_deadline.clear()
        self._refresh_fire_state()

    def actuator_states(self) -> dict[str, ActuatorReading]:
        """Returns a copy of all current mock actuator states."""
        now = time.perf_counter()
        with self._lock:
            return {
                name: ActuatorReading(
                    state  = state,
                    moving = self._move_deadline.get(name, 0.0) > now,
                )
                for name, state in self._actuators.items()
            }

    # --------------------------------------------------------
    # Device info
    # --------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @property
    def serial_number(self) -> str:
        return "MOCK-000000"

    @property
    def stream_time_s(self) -> float:
        """Now on the time_s clock (seconds since the first stream start)."""
        return time.perf_counter() - self._stream_start

    @property
    def stream_rate_hz(self) -> int:
        return self._stream_hz

    @property
    def sensor_tags(self) -> list[str]:
        """All sensor tags produced by stream_read()."""
        return [spec.id for spec in self._stream_channels]

    # --------------------------------------------------------
    # Private Simulation Helpers
    # --------------------------------------------------------

    def _sample(self, spec: ChannelSpec, t: float, fire_elapsed: float | None) -> float:
        """Generate one raw voltage sample for a channel, by its manifest type."""
        if spec.type == PT_DIRECT:
            i = self._pt_index.get(spec.id, 0)
            return self._sine_sample(
                t,
                _PT_BASE_V + _PT_OFFSET_V * i,
                _PT_AMP_V,
                _PT_PERIOD_S + i,
                _PT_NOISE_V,
            )
        if spec.type == PT_DIFFERENTIAL:
            i = self._dpt_index.get(spec.id, 0)
            return self._sine_sample(
                t,
                _DPT_BASE_V + _DPT_OFFSET_V * i,
                _DPT_AMP_V,
                _DPT_PERIOD_S + i,
                _DPT_NOISE_V,
            )
        if spec.type == TC_DIFFERENTIAL:
            return self._sine_sample(
                t, _TC_BASE_V, _TC_AMP_V, _TC_PERIOD_S, _TC_NOISE_V
            )
        if spec.type == LC_DIRECT:
            return self._lc_sample(fire_elapsed)
        if spec.type == VOLTAGE:
            return self._igniter_sense_sample()
        return 0.0

    def _igniter_sense_sample(self) -> float:
        """The only voltage channel is the igniter sense node; follow its lines."""
        armed  = self._actuators.get("ignition_arm") == 1
        firing = armed and self._actuators.get("ignition_fire") == 1
        level = _IGN_FIRING_V if firing else _IGN_ARMED_V if armed else _IGN_SAFE_V
        return level + self._noise() * _IGN_NOISE_V

    def _sine_sample(
        self,
        t: float,
        base: float,
        amplitude: float,
        period: float,
        noise_scale: float,
    ) -> float:
        """Return base + sine drift + gaussian-ish noise."""
        drift = amplitude * math.sin(2 * math.pi * t / period)
        noise = self._noise() * noise_scale
        return base + drift + noise

    def _lc_sample(self, fire_elapsed: float | None) -> float:
        """Noise at rest; while firing, ramp over _LC_RAMP_S then oscillate."""
        if fire_elapsed is None:
            return self._noise() * _LC_NOISE_V

        ramp        = min(1.0, fire_elapsed / _LC_RAMP_S)
        thrust_v    = _LC_FIRE_V * ramp
        oscillation = _LC_OSC_V * math.sin(2 * math.pi * fire_elapsed / _LC_OSC_PERIOD_S)
        noise       = self._noise() * _LC_FIRE_NOISE_V
        return thrust_v + oscillation + noise

    def _move_duration(self, spec: ActuatorSpec, state: int) -> float:
        """Move time: steps / pulse rate."""
        steps = spec.steps_open if state == 1 else spec.steps_close
        return steps / spec.pulse_freq_hz

    def _refresh_fire_state(self) -> None:
        """Mark the fire start (every stepper main commanded open) for the LC ramp."""
        with self._lock:
            firing = bool(self._steppers) and all(
                self._actuators.get(spec.id, 0) == 1 for spec in self._steppers
            )
            if firing and self._fire_start is None:
                self._fire_start = time.perf_counter()
            elif not firing:
                self._fire_start = None

    def _fire_elapsed(self) -> float | None:
        """Seconds since firing began, or None if not firing."""
        with self._lock:
            start = self._fire_start
        return None if start is None else time.perf_counter() - start

    def _noise(self) -> float:
        """Roughly Gaussian noise in [-1, 1]: mean of 4 xorshift32 uniforms."""
        total = 0.0
        for _ in range(4):
            x = self._rng_state
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= (x >> 17) & 0xFFFFFFFF
            x ^= (x << 5)  & 0xFFFFFFFF
            self._rng_state = x & 0xFFFFFFFF
            total += (x / 0xFFFFFFFF) * 2.0 - 1.0
        return total / 4.0
