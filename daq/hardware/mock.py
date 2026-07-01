"""
daq/hardware/mock.py

Simulated LabJack T7 for development and hardware-free testing.

Provides mathematical models of pressures, temperatures, and load cells
to produce realistic, responsive signals matching real test physics.

Sensor simulation:
  - Pressure transducers: slow sine-wave drift around realistic setpoints
  - Thermocouples: stable with small noise, LOX TC near -160 C
  - Load cells: zero until a "fire" state is active, then ramp up
  - CJC (LM34): fixed room temperature with minor drift
"""

from __future__ import annotations

import math
import time
import threading
from typing import Any


# -- Sensor Simulation Parameters -----------------------------
# tag -> (base_voltage, amplitude, period_s, noise_scale)
# Voltage formula: V = base + amplitude * sin(2pi * t / period) + noise
_PT_SENSORS: dict[str, tuple[float, float, float, float]] = {
    # tag:   base_V  amp_V  period_s  noise_V
    "POT":   (1.4663, 0.030, 18.0, 0.002),  # LOX tank       ~250 psi
    "PFT":   (1.4663, 0.025, 20.0, 0.002),  # Fuel tank      ~250 psi
    "POI":   (1.2679, 0.020, 15.0, 0.002),  # LOX inlet      ~200 psi
    "PFI":   (1.2679, 0.020, 15.0, 0.002),  # Fuel inlet     ~200 psi
    "PFO":   (1.1885, 0.018, 14.0, 0.002),  # Fuel outlet    ~180 psi
    "PC":    (1.0694, 0.040, 10.0, 0.003),  # Chamber        ~150 psi
    "PNS":   (1.4663, 0.010, 30.0, 0.001),  # System GN2     ~250 psi
    "PNP":   (1.1885, 0.008, 25.0, 0.001),  # Pneumatics GN2 ~180 psi
}

_TC_SENSORS: dict[str, tuple[float, float, float, float]] = {
    # tag:      base_V   amp_V   period_s  noise_V
    "TOI":   (-0.006200, 0.000050, 25.0, 0.000005),  # LOX inlet TC  ~-160 C
    "TFI":   ( 0.000800, 0.000100, 20.0, 0.000008),  # Fuel inlet TC ~+45 C
    "TFO":   ( 0.001200, 0.000120, 18.0, 0.000010),  # Fuel outlet TC ~+55 C
}

_LC_SENSORS: dict[str, tuple[float, float, float, float]] = {
    "LC_1":  (0.0, 0.0, 1.0, 0.002),
    "LC_2":  (0.0, 0.0, 1.0, 0.002),
}

_CJC_BASE_V  = 0.770
_CJC_DRIFT_V = 0.001

# -- Actuator Definitions -------------------------------------

_ACTUATOR_NAMES: tuple[str, ...] = (
    "LOx Press",
    "Fuel Press",
    "LOx Purge",
    "Fuel Purge",
    "LOx Main",
    "Fuel Main",
    "LOx Vent",
    "Fuel Vent",
    "Ignition",
)

# -- Timing Parameters ----------------------------------------

_STREAM_HZ         = 500
_SCANS_PER_READ    = 50
_BATCH_DURATION_S  = _SCANS_PER_READ / _STREAM_HZ   # 0.1 s


class MockLabJack:
    """
    Simulated LabJack T7.

    Usage:
        device = MockLabJack()
        device.open()
        device.start_stream()

        while running:
            batch = device.stream_read()   # blocks ~0.1 s
            cjc_v = device.read_cjc()
            device.write_actuator("LOx Main", state=1)

        device.stop_stream()
        device.close()
    """

    def __init__(self, stream_hz: int = _STREAM_HZ) -> None:
        self._stream_hz      = stream_hz
        self._scans_per_read = _SCANS_PER_READ
        self._batch_duration = _SCANS_PER_READ / stream_hz

        self._lock           = threading.Lock()
        self._actuators: dict[str, int] = {name: 0 for name in _ACTUATOR_NAMES}
        self._connected      = False
        self._streaming      = False

        self._stream_start: float = 0.0
        self._next_batch_time: float = 0.0

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
            Dict mapping sensor tag -> list of raw voltages, one per scan.
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

        fire_active = self._is_firing()

        for i in range(self._scans_per_read):
            t = t_elapsed - (self._scans_per_read - 1 - i) / self._stream_hz
            batch["scan_times"].append(t)

            for tag, (base, amp, period, noise) in _PT_SENSORS.items():
                batch[tag].append(self._sine_sample(t, base, amp, period, noise))

            for tag, (base, amp, period, noise) in _TC_SENSORS.items():
                batch[tag].append(self._sine_sample(t, base, amp, period, noise))

            for tag in _LC_SENSORS:
                batch[tag].append(self._lc_sample(t, fire_active))

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

    # --------------------------------------------------------
    # Actuator control
    # --------------------------------------------------------

    def write_actuator(self, name: str, state: int) -> None:
        """
        Set an actuator to open (1) or closed (0).

        Args:
            name:  Actuator name, must be one of the keys in _ACTUATOR_NAMES.
            state: 1 = energised/open, 0 = de-energised/closed.

        Raises:
            KeyError: If name is not a recognised actuator.
        """
        if name not in self._actuators:
            raise KeyError(f"Unknown actuator: '{name}'")
        with self._lock:
            self._actuators[name] = state
        print(f"[MOCK] {name} -> {'OPEN' if state else 'CLOSED'}")

    def read_actuator(self, name: str) -> int:
        """
        Return the current logical state of an actuator.

        Args:
            name: Actuator name.

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
        print("[MOCK] All actuators -> SAFE/CLOSED")

    def actuator_states(self) -> dict[str, int]:
        """Returns a copy of all current mock actuator states."""
        with self._lock:
            return dict(self._actuators)

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
        return list(_PT_SENSORS) + list(_TC_SENSORS) + list(_LC_SENSORS)

    # --------------------------------------------------------
    # Private Simulation Helpers
    # --------------------------------------------------------

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

    def _lc_sample(self, t: float, fire_active: bool) -> float:
        """
        Simulate load cell voltage.

        At rest: near-zero with small noise.
        During fire: ramp up over 0.5 s to ~500 lbf (5.0 V at slope=100),
        then hold with thrust oscillation.
        """
        if not fire_active:
            return self._noise() * 0.003

        fire_elapsed = t - self._fire_start_time()
        ramp = min(1.0, fire_elapsed / 0.5)
        thrust_v = 5.0 * ramp
        oscillation = 0.1 * math.sin(2 * math.pi * fire_elapsed / 0.8)
        noise = self._noise() * 0.005
        return thrust_v + oscillation + noise

    def _is_firing(self) -> bool:
        """True if both main valves are open (LOx Main AND Fuel Main)."""
        with self._lock:
            return (
                self._actuators.get("LOx Main", 0) == 1
                and self._actuators.get("Fuel Main", 0) == 1
            )

    def _fire_start_time(self) -> float:
        """Simulates sequential valve movement during autosequence startup."""
        return time.perf_counter() - self._stream_start - 0.1

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