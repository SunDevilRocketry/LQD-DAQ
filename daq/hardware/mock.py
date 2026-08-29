"""
daq/hardware/mock.py

Simulated LabJack T7 for development and hardware-free testing.

Provides mathematical models of pressures, temperatures, load cells and
photogate counters to produce realistic, responsive signals matching real
test physics.

Sensor simulation:
  - pt_direct:         slow sine-wave drift around a realistic setpoint
  - tc_differential:   stable with small noise, near -160 C
  - lc_direct:         zero until both stepper mains are open, then ramps
  - photogate_counter: accumulates counts while its stepper is moving
  - CJC (LM34):        fixed room temperature with minor drift
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
    PULSE_STEPPER,
    PT_DIRECT,
    TC_DIFFERENTIAL,
    LC_DIRECT,
    PHOTOGATE_COUNTER,
)


# -- Sensor Simulation Parameters -----------------------------
# Voltage formula: V = base + amplitude * sin(2pi * t / period) + noise

# Pressure transducers. 0.50 V is ~250 psi at the PX309-2K5V nominal
# 500 psi/V slope; each successive PT is offset a little so channels are
# distinguishable in a dashboard rather than drawing on top of each other.
_PT_BASE_V     = 0.50
_PT_OFFSET_V   = 0.04
_PT_AMP_V      = 0.020
_PT_PERIOD_S   = 12.0
_PT_NOISE_V    = 0.002

# Thermocouples: millivolt-scale differential, LOX-ish (~-160 C).
_TC_BASE_V     = -0.006200
_TC_AMP_V      = 0.000050
_TC_PERIOD_S   = 25.0
_TC_NOISE_V    = 0.000005

# Load cells: at rest near zero, ramping to ~500 lbf (5.0 V at slope=100).
_LC_NOISE_V       = 0.003
_LC_FIRE_V        = 5.0
_LC_RAMP_S        = 0.5
_LC_OSC_V         = 0.1
_LC_OSC_PERIOD_S  = 0.8
_LC_FIRE_NOISE_V  = 0.005

_CJC_BASE_V  = 0.770
_CJC_DRIFT_V = 0.001

# -- Actuator Simulation Parameters ---------------------------
# How long a commanded stepper move stays in flight when the manifest has
# no steps/rate filled in yet. Real values override this once actuators.yaml
# carries steps_open/steps_close/pulse_freq_hz.
_DEFAULT_MOVE_DURATION_S = 1.5

# Photogate edges per second while a stepper is moving.
_COUNTS_PER_SECOND = 200.0

# -- Timing Parameters ----------------------------------------

_STREAM_HZ         = 500
_SCANS_PER_READ    = 50


class MockLabJack:
    """
    Simulated LabJack T7.

    Usage:
        device = MockLabJack(channels, actuators)
        device.open()
        device.start_stream()

        while running:
            batch = device.stream_read()   # blocks ~0.1 s
            cjc_v = device.read_cjc()
            device.write_actuator("lox_main", state=1)

        device.stop_stream()
        device.close()
    """

    def __init__(
        self,
        channels:  Sequence[ChannelSpec]  = (),
        actuators: Sequence[ActuatorSpec] = (),
        stream_hz: int = _STREAM_HZ,
    ) -> None:
        self._stream_hz      = stream_hz
        self._scans_per_read = _SCANS_PER_READ
        self._batch_duration = _SCANS_PER_READ / stream_hz

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

        self._actuator_specs: dict[str, ActuatorSpec] = {
            spec.id: spec for spec in actuators
        }
        self._actuators: dict[str, int] = {spec.id: 0 for spec in actuators}
        self._steppers = [
            spec for spec in actuators if spec.type == PULSE_STEPPER
        ]

        # Deadline per stepper; a commanded move is in flight
        # until perf_counter() passes it.
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

        # Uniform seed for reproducible noise generations
        self._rng_state: int = 0xDEADBEEF

    # --------------------------------------------------------
    # Lifecycle Management
    # --------------------------------------------------------

    def open(self) -> None:
        """Simulate opening a connection to the LabJack."""
        self._connected = True
        print("[MOCK] LabJack T7 connected (simulated)")

    def close(self) -> None:
        """Simulate closing the connection."""
        self._connected  = False
        self._streaming  = False
        print("[MOCK] LabJack T7 disconnected (simulated)")

    def start_stream(self) -> int:
        """
        Simulate starting the 500 Hz data stream.

        Returns:
            The actual (simulated) stream rate in Hz.
        """
        self._stream_start    = time.perf_counter()
        self._next_batch_time = self._stream_start + self._batch_duration
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
        Blocks to simulate 500 Hz clock, then generates mock scan values.

        Returns:
            Dict mapping channel id -> list of raw voltages, one per scan.
            Also includes "scan_times" -> list of elapsed times in seconds.

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

        fire_elapsed = self._fire_elapsed()

        for i in range(self._scans_per_read):
            t = t_elapsed - (self._scans_per_read - 1 - i) / self._stream_hz
            batch["scan_times"].append(t)

            for spec in self._stream_channels:
                batch[spec.id].append(self._sample(spec, t, fire_elapsed))

        return batch

    def read_cjc(self) -> float:
        """
        Simulates LM34 cold junction ambient reference voltage.

        Returns:
            Voltage in volts. At 25 °C room temp this is ~0.770 V.
        """
        t = time.perf_counter() - self._stream_start
        drift = _CJC_DRIFT_V * math.sin(2 * math.pi * t / 120.0)
        noise = self._noise() * 0.0002
        return _CJC_BASE_V + drift + noise

    def read_counters(self) -> dict[str, float]:
        """
        Simulates the photogate position-feedback counters.

        A counter only advances while the stepper it belongs to is actually
        moving, so the count tracks commanded motion. Reported telemetry only.

        Returns:
            Dict: channel id -> accumulated edge count.
        """
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
        Set an actuator to open (1) or closed (0).

        A pulse_stepper actuator additionally starts a simulated move that
        keeps it reporting moving=True for the burst's duration.

        Args:
            name:  Actuator id from actuators.yaml.
            state: 1 = energised/open, 0 = de-energised/closed.

        Raises:
            KeyError: If name is not a recognised actuator.
        """
        spec = self._actuator_specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown actuator: '{name}'")

        with self._lock:
            self._actuators[name] = state
            if spec.type == PULSE_STEPPER:
                self._move_deadline[name] = (
                    time.perf_counter() + self._move_duration(spec, state)
                )
        self._refresh_fire_state()
        print(f"[MOCK] {name} -> {'OPEN' if state else 'CLOSED'}")

    def read_actuator(self, name: str) -> int:
        """
        Return the current logical state of an actuator.

        Args:
            name: Actuator id.

        Returns:
            1 if energised/open, 0 if de-energised/closed.
        """
        with self._lock:
            return self._actuators[name]

    def all_safe(self) -> None:
        """De-energise all actuators (safe/closed position)."""
        with self._lock:
            for name in self._actuators:
                self._actuators[name] = 0
            self._move_deadline.clear()
        self._refresh_fire_state()
        print("[MOCK] All actuators -> SAFE/CLOSED")

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
        if spec.type == TC_DIFFERENTIAL:
            return self._sine_sample(
                t, _TC_BASE_V, _TC_AMP_V, _TC_PERIOD_S, _TC_NOISE_V
            )
        if spec.type == LC_DIRECT:
            return self._lc_sample(fire_elapsed)
        return 0.0

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
        """
        Simulate load cell voltage.

        At rest: near-zero with small noise.
        During fire: ramp up over _LC_RAMP_S to ~500 lbf (5.0 V at
        slope=100), then hold with thrust oscillation.
        """
        if fire_elapsed is None:
            return self._noise() * _LC_NOISE_V

        ramp        = min(1.0, fire_elapsed / _LC_RAMP_S)
        thrust_v    = _LC_FIRE_V * ramp
        oscillation = _LC_OSC_V * math.sin(2 * math.pi * fire_elapsed / _LC_OSC_PERIOD_S)
        noise       = self._noise() * _LC_FIRE_NOISE_V
        return thrust_v + oscillation + noise

    def _move_duration(self, spec: ActuatorSpec, state: int) -> float:
        """
        How long a commanded stepper move should stay in flight.

        Uses the manifest's real step count and pulse rate when they're
        filled in, and a fixed stand-in while they're still null.
        """
        steps = spec.steps_open if state == 1 else spec.steps_close
        if steps and spec.pulse_freq_hz:
            return steps / spec.pulse_freq_hz
        return _DEFAULT_MOVE_DURATION_S

    def _refresh_fire_state(self) -> None:
        """
        Track when firing starts, so the load cell ramp has a real origin.

        Firing means every stepper main is commanded open - the mock's
        stand-in for "the engine is running".
        """
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
        """
        Gaussian-ish noise in [-1, 1].
        Uses a xorshift32 PRNG so output is reproducible given the same seed.
        Sum of 4 uniform samples approximates gaussian by CLT.
        """
        total = 0.0
        for _ in range(4):
            x = self._rng_state
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= (x >> 17) & 0xFFFFFFFF
            x ^= (x << 5)  & 0xFFFFFFFF
            self._rng_state = x & 0xFFFFFFFF
            total += (x / 0xFFFFFFFF) * 2.0 - 1.0
        return total / 4.0
