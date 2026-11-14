"""
daq/hardware/mock.py

Simulated LabJack T7 for hardware-free runs and tests. Same public
interface as hardware/interface.py; keep the two in sync.

Signals: sine drift plus noise; load cells ramp once both mains are open;
a photogate is blocked while its main is closed and idle.
"""

from __future__ import annotations

import math
import time
import threading
from typing import Optional, Sequence

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
    PHOTOGATE,
    VOLTAGE,
)


# -- Sensor Simulation Parameters -----------------------------

# Plant: each pressure moves toward a valve-set target, first order with tau.
_PT_PSI_PER_V      = 400.0
_DPT_PSI_PER_V     = 5.80151
_TANK_PRESS_PSI    = 250.0
_TANK_FLOW_SAG     = 0.9      # press open with the main flowing
_TANK_PRESS_TAU_S  = 2.0
_TANK_VENT_TAU_S   = 1.5
_TANK_BLOWDOWN_TAU_S = 6.0    # main open, press closed
_CHAMBER_RATIO     = 0.6      # Pc / lower tank pressure, both mains open
_CHAMBER_PURGE_PSI = 15.0
_CHAMBER_TAU_S     = 0.3
_ORIFICE_DP_PSI    = 0.5      # at _TANK_PRESS_PSI, main open
# pt index -> role; dpt index -> side. Matches the dashboard's PT roles.
_TANK_SIDES = (
    ("lox",  "lox_press",  "tank_depress", "lox_main"),
    ("fuel", "fuel_press", "fuel_vent",    "fuel_main"),
)

# Spare PTs (pt3 on): base + sine drift + noise, offset so traces don't overlap.
_PT_BASE_V     = 0.50
_PT_OFFSET_V   = 0.04
_PT_AMP_V      = 0.020
_PT_PERIOD_S   = 12.0
_PT_NOISE_V    = 0.002
_DPT_NOISE_V   = 0.001

# ~-177 C (Type T) against the mock's 25 C cold junction.
_TC_BASE_V     = -0.006200
_TC_AMP_V      = 0.000050
_TC_PERIOD_S   = 25.0
_TC_NOISE_V    = 0.000005

# ~500 lbf per cell at full Pc, 440.92 lbf/V.
_LC_NOISE_V       = 0.003
_LC_FIRE_V        = 1.134
_LC_OSC_V         = 0.1
_LC_OSC_PERIOD_S  = 0.8

# Igniter sense node, calculated from the igniter schematic: no e-match
# fitted (until testing), armed (~battery), firing.
_IGN_SAFE_V    = 4.4
_IGN_ARMED_V   = 9.0
_IGN_FIRING_V  = 2.0
_IGN_NOISE_V   = 0.01

_CJC_BASE_C  = 25.0
_CJC_DRIFT_C = 1.0               # slow +/-1 C swing, like a warming T7

# -- Actuator Simulation Parameters ---------------------------

# -- Timing Parameters ----------------------------------------
_STREAM_HZ         = 500


class MockLabJack:
    """Simulated LabJack T7; same public interface as LabJackT7."""

    def __init__(
        self,
        channels:  Sequence[ChannelSpec]  = (),
        actuators: Sequence[ActuatorSpec] = (),
        stream_hz: int = _STREAM_HZ,
        ip:        Optional[str] = None,   # unused; matches LabJackT7
    ) -> None:
        self._stream_hz      = stream_hz
        self._scans_per_read = max(1, stream_hz // 20)   # 20 batches/s, like the T7
        self._batch_duration = self._scans_per_read / stream_hz

        self._lock = threading.Lock()

        self._stream_channels = [
            spec for spec in channels if spec.active and spec.is_streamed
        ]
        self._photogates = [
            spec for spec in channels
            if spec.active and spec.type == PHOTOGATE
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

        self._connected      = False
        self._streaming      = False
        self._stream_start: float = 0.0
        self._next_batch_time: float = 0.0
        # Plant pressures, psig: tanks start loaded but unpressurised.
        self._psi = {"lox": 0.0, "fuel": 0.0, "chamber": 0.0}

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

        for i in range(self._scans_per_read):
            t = t_elapsed - (self._scans_per_read - 1 - i) / self._stream_hz
            batch["scan_times"].append(t)
            self._step_plant(1.0 / self._stream_hz)

            for spec in self._stream_channels:
                batch[spec.id].append(self._sample(spec, t))

        return batch

    def read_cjc_celsius(self) -> float:
        """Cold junction C, near 25 C."""
        t = time.perf_counter() - self._stream_start
        drift = _CJC_DRIFT_C * math.sin(2 * math.pi * t / 60.0)
        noise = self._noise() * 0.02
        return _CJC_BASE_C + drift + noise

    def read_photogates(self) -> dict[str, int]:
        """Photogate levels by channel id: 0 (blocked) while its main is closed and idle."""
        now = time.perf_counter()
        active = {spec.id for spec in self._photogates}
        with self._lock:
            return {
                spec.position_feedback: int(
                    self._actuators[spec.id] == 1
                    or self._move_deadline.get(spec.id, 0.0) > now
                )
                for spec in self._steppers if spec.position_feedback in active
            }

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

    def _step_plant(self, dt: float) -> None:
        """Move the tank and chamber pressures toward their valve-set targets."""
        def approach(key: str, target: float, tau: float) -> None:
            self._psi[key] += (target - self._psi[key]) * min(1.0, dt / tau)

        on = lambda name: self._actuators.get(name) == 1
        for side, press, vent, main in _TANK_SIDES:
            if on(vent):
                approach(side, 0.0, _TANK_VENT_TAU_S)
            elif on(press):
                sag = _TANK_FLOW_SAG if on(main) else 1.0
                approach(side, _TANK_PRESS_PSI * sag, _TANK_PRESS_TAU_S)
            elif on(main):
                approach(side, 0.0, _TANK_BLOWDOWN_TAU_S)

        if on("lox_main") and on("fuel_main"):
            target = _CHAMBER_RATIO * min(self._psi["lox"], self._psi["fuel"])
        elif on("lox_purge") or on("fuel_purge"):
            target = _CHAMBER_PURGE_PSI
        else:
            target = 0.0
        approach("chamber", target, _CHAMBER_TAU_S)

    def _sample(self, spec: ChannelSpec, t: float) -> float:
        """Generate one raw voltage sample for a channel, by its manifest type."""
        if spec.type == PT_DIRECT:
            i = self._pt_index.get(spec.id, 0)
            role = ("lox", "fuel", "chamber")[i] if i < 3 else None
            if role:
                return self._psi[role] / _PT_PSI_PER_V + self._noise() * _PT_NOISE_V
            return self._sine_sample(
                t,
                _PT_BASE_V + _PT_OFFSET_V * i,
                _PT_AMP_V,
                _PT_PERIOD_S + i,
                _PT_NOISE_V,
            )
        if spec.type == PT_DIFFERENTIAL:
            side, _, _, main = _TANK_SIDES[min(self._dpt_index.get(spec.id, 0), 1)]
            dp_psi = (_ORIFICE_DP_PSI * self._psi[side] / _TANK_PRESS_PSI
                      if self._actuators.get(main) == 1 else 0.0)
            return dp_psi / _DPT_PSI_PER_V + self._noise() * _DPT_NOISE_V
        if spec.type == TC_DIFFERENTIAL:
            return self._sine_sample(
                t, _TC_BASE_V, _TC_AMP_V, _TC_PERIOD_S, _TC_NOISE_V
            )
        if spec.type == LC_DIRECT:
            return self._lc_sample(t)
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

    def _lc_sample(self, t: float) -> float:
        """Thrust follows chamber pressure, with combustion roughness while burning."""
        # Purge gas alone gives no thrust.
        full = _CHAMBER_RATIO * _TANK_PRESS_PSI - _CHAMBER_PURGE_PSI
        level = max(0.0, self._psi["chamber"] - _CHAMBER_PURGE_PSI) / full
        oscillation = _LC_OSC_V * level * math.sin(2 * math.pi * t / _LC_OSC_PERIOD_S)
        return _LC_FIRE_V * level + oscillation + self._noise() * _LC_NOISE_V

    def _move_duration(self, spec: ActuatorSpec, state: int) -> float:
        """Move time: steps / pulse rate."""
        steps = spec.steps_open if state == 1 else spec.steps_close
        return steps / spec.pulse_freq_hz

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
