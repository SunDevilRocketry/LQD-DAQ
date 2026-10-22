"""
daq/hardware/interface.py

LabJack T7 driver (LJM). Same public interface as hardware/mock.py; keep
the two in sync. Channels and actuators with a null pin are skipped.

binary_dio lines are active-low; this layer inverts so 1 = energised.
The watchdog covers binary_dio lines only: stepper lines stay out until
watchdog vs DIO_EF Pulse Out is bench-tested.
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Sequence

from labjack import ljm

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
    check_ain,
)


# CJC: internal sensor minus 3 C for the AIN0-3 screw terminals (LabJack 18.0).
_CJC_REGISTER          = "TEMPERATURE_DEVICE_K"
_CJC_SCREW_TERMINAL_C  = -3.0
_KELVIN_OFFSET         = 273.15

# -- Stream & Safety Parameters -------------------------------
_STREAM_HZ       = 500
_SCANS_PER_READ  = 50     # replaced in start_stream() by rate // 20
_SETTLING_US     = 100
_RESOLUTION      = 3      # STREAM_RESOLUTION_INDEX, applies to every channel

# Only command-response traffic resets it; stream data doesn't.
_WATCHDOG_TIMEOUT_S = 5

# -- DIO_EF Parameters ----------------------------------------
# 80 MHz / 8 = 10 MHz tick; the roll value fits from 10 Hz to 20 kHz.
_DIO_EF_CORE_HZ      = 80_000_000
_DIO_EF_CLOCK_DIV    = 8
_DIO_EF_TICK_HZ      = _DIO_EF_CORE_HZ // _DIO_EF_CLOCK_DIV

_DIO_EF_INDEX_PULSE_OUT = 2
_DIO_EF_INDEX_COUNTER   = 8

# Bank -> flat DIO number (MIO0 = DIO20).
_DIO_BANK_OFFSETS = {"FIO": 0, "EIO": 8, "CIO": 16, "MIO": 20}


def _dio_number(name: str) -> int:
    """
    DIO number for 'EIO0' / 'MIO1' / 'DIO8', for the watchdog bitmasks.

    Raises:
        ValueError: If the name isn't a recognised DIO register.
    """
    upper = name.strip().upper()
    if upper.startswith("DIO"):
        return int(upper[3:])
    bank = upper[:3]
    if bank in _DIO_BANK_OFFSETS:
        return _DIO_BANK_OFFSETS[bank] + int(upper[3:])
    raise ValueError(f"Unrecognised DIO register name: '{name}'")


class LabJackT7:
    """LabJack T7 driver; same public interface as MockLabJack."""

    def __init__(
        self,
        channels:  Sequence[ChannelSpec],
        actuators: Sequence[ActuatorSpec],
        stream_hz: int = _STREAM_HZ,
    ) -> None:
        self._stream_hz      = stream_hz
        self._scans_per_read = _SCANS_PER_READ

        self._handle: Optional[int] = None
        self._lock = threading.Lock()

        # Active channels with a pin; the rest are never touched.
        self._channels = [
            spec for spec in channels if spec.active and spec.is_wired
        ]
        self._stream_channels = [
            spec for spec in self._channels if spec.is_streamed
        ]
        self._counter_channels = [
            spec for spec in self._channels if spec.type == PHOTOGATE_COUNTER
        ]

        for spec in self._channels:
            for name in (spec.ain, spec.ain_pos, spec.ain_neg):
                check_ain(name)

        self._actuator_specs: dict[str, ActuatorSpec] = {
            spec.id: spec for spec in actuators
        }
        self._actuators: dict[str, int] = {
            spec.id: 0 for spec in actuators
        }
        # Per stepper: the burst is in flight until perf_counter() passes this.
        self._move_deadline: dict[str, float] = {}

        self._warn_unwired(channels, actuators)

        self._connected  = False
        self._streaming  = False
        self._stream_start: float = 0.0
        self._actual_hz: int = 0

        # Filled by _build_scan_list().
        self._scan_tags:    list[str] = []
        self._scan_addrs:   list[int] = []
        self._n_channels:   int = 0

        # Shared CLOCK0 roll value, set in _configure_steppers().
        self._pulse_roll: int = 0

        # Watchdog mask: binary_dio lines only (valves, igniter).
        self._actuator_mask = 0
        for spec in actuators:
            if spec.type == BINARY_DIO and spec.dio is not None:
                self._actuator_mask |= 1 << _dio_number(spec.dio)

    @staticmethod
    def _warn_unwired(
        channels: Sequence[ChannelSpec], actuators: Sequence[ActuatorSpec]
    ) -> None:
        """Print a startup warning for anything active but missing a pin."""
        for spec in channels:
            if spec.active and not spec.is_wired:
                print(f"[T7] WARNING: channel '{spec.id}' has no pin assigned "
                      f"in channels.yaml - it will report no data")
        for spec in actuators:
            if not spec.is_wired:
                print(f"[T7] WARNING: actuator '{spec.id}' is not fully "
                      f"specified in actuators.yaml - it cannot be driven")

    # --------------------------------------------------------
    # Lifecycle management
    # --------------------------------------------------------

    def open(self) -> None:
        """
        Connects to the physical T7 and configures startup safety states.

        Raises:
            ljm.LJMError: If no T7 is found or connection fails.
        """
        self._handle = ljm.openS("T7", "ANY", "ANY")
        info = ljm.getHandleInfo(self._handle)
        self._serial = str(info[2])

        self._configure_channels()
        self._configure_counters()
        self._configure_steppers()
        self._safe_all_hardware()
        self._arm_watchdog()

        self._connected = True
        print(f"[T7] Connected: serial {self._serial}")

    def close(self) -> None:
        """Close the connection, driving all actuators safe first."""
        if self._handle is None:
            return
        try:
            if self._streaming:
                self.stop_stream()
            self._safe_all_hardware()
            ljm.close(self._handle)
        except ljm.LJMError as exc:
            print(f"[T7] Warning during close: {exc}")
        finally:
            self._handle    = None
            self._connected = False
            self._streaming = False
            print("[T7] Disconnected")

    def start_stream(self) -> int:
        """
        Starts the background AIN stream, halving rate on Scan Overlap.

        Returns:
            Actual stream rate negotiated with the hardware (Hz).

        Raises:
            RuntimeError: If streaming cannot be started at any rate >= 10 Hz,
                          or if no channel in the manifest is wired.
            ljm.LJMError: On unexpected LJM errors.
        """
        self._build_scan_list()

        if self._n_channels == 0:
            raise RuntimeError(
                "No wired analog channels in channels.yaml - nothing to "
                "stream. Fill in the ain/ain_pos/ain_neg fields for the "
                "channels present on this cart."
            )

        ljm.eWriteName(self._handle, "STREAM_SETTLING_US",     _SETTLING_US)
        ljm.eWriteName(self._handle, "STREAM_RESOLUTION_INDEX", _RESOLUTION)

        target = self._stream_hz
        actual = None

        while target >= 10:
            try:
                scans = max(1, target // 20)
                actual = ljm.eStreamStart(
                    self._handle,
                    scans,
                    self._n_channels,
                    self._scan_addrs,
                    target,
                )
                self._scans_per_read = scans
                break
            except ljm.LJMError as exc:
                if exc.errorCode == 2942:   # STREAM_SCAN_OVERLAP
                    print(f"[T7] {target} Hz failed (SCAN_OVERLAP), "
                          f"retrying at {target // 2} Hz")
                    target //= 2
                else:
                    raise

        if actual is None:
            raise RuntimeError(
                "Could not start stream at any rate >= 10 Hz. "
                "Check channel count and cable connections."
            )

        self._actual_hz   = int(actual)
        self._streaming   = True
        if not self._stream_start:   # once, so time_s runs on across reconnects
            self._stream_start = time.perf_counter()
        print(f"[T7] Stream started: requested {self._stream_hz} Hz, "
              f"actual {self._actual_hz} Hz, "
              f"{self._scans_per_read} scans/read")
        return self._actual_hz

    def stop_stream(self) -> None:
        """Stops the active hardware AIN stream."""
        self._streaming = False
        if self._handle is not None:
            try:
                ljm.eStreamStop(self._handle)
            except ljm.LJMError:
                pass
        print("[T7] Stream stopped")

    # --------------------------------------------------------
    # Data acquisition
    # --------------------------------------------------------

    def stream_read(self) -> dict[str, list[float]]:
        """
        Reads one batch of scan times and raw voltages from LJM stream.

        Returns:
            Dict: channel id -> list[float] raw voltages (one per scan).
                  Also "scan_times" -> list[float] elapsed seconds.
                  Also "backlog_device" and "backlog_ljm" -> int counts.

        Raises:
            RuntimeError: If called before start_stream().
            ljm.LJMError: On stream read errors (engine.py should handle).
        """
        if not self._streaming:
            raise RuntimeError("stream_read() called before start_stream()")

        ret    = ljm.eStreamRead(self._handle)
        data   = ret[0]
        dev_bl = ret[1]
        ljm_bl = ret[2]

        n_scans = len(data) // self._n_channels
        t_now   = time.perf_counter() - self._stream_start

        batch: dict[str, list[float]] = {tag: [] for tag in self._scan_tags}
        batch["scan_times"]    = []
        batch["backlog_device"] = [dev_bl]   # type: ignore[assignment]
        batch["backlog_ljm"]    = [ljm_bl]   # type: ignore[assignment]

        for sc in range(n_scans):
            base = sc * self._n_channels
            t_scan = t_now - (n_scans - 1 - sc) / self._actual_hz
            batch["scan_times"].append(t_scan)

            for i, tag in enumerate(self._scan_tags):
                batch[tag].append(data[base + i])

        return batch

    def read_cjc_celsius(self) -> float:
        """Cold junction C. Command-response, so it also resets the watchdog."""
        kelvin = ljm.eReadName(self._handle, _CJC_REGISTER)
        return kelvin - _KELVIN_OFFSET + _CJC_SCREW_TERMINAL_C

    def read_counters(self) -> dict[str, float]:
        """Photogate edge counts by channel id; a failed read is omitted, not zeroed."""
        counts: dict[str, float] = {}
        for spec in self._counter_channels:
            try:
                counts[spec.id] = ljm.eReadName(
                    self._handle, f"{spec.dio}_EF_READ_A"
                )
            except ljm.LJMError:
                continue
        return counts

    # --------------------------------------------------------
    # Actuator control
    # --------------------------------------------------------

    def write_actuator(self, name: str, state: int) -> None:
        """
        Energise (1) or de-energise (0) an actuator.

        A stepper starts a fixed-count burst and returns at once; it reports
        moving=True until the burst should have finished.

        Raises:
            KeyError:      If name is not a recognised actuator.
            RuntimeError:  If the actuator has no pin assignment yet.
            ljm.LJMError:  If the LJM write fails.
        """
        spec = self._actuator_specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown actuator: '{name}'")
        if not spec.is_wired:
            raise RuntimeError(
                f"Actuator '{name}' is not fully specified in actuators.yaml "
                f"- fill in its pin/step fields before commanding it"
            )

        if spec.type == BINARY_DIO:
            hw_state = 0 if state == 1 else 1   # active-low
            ljm.eWriteName(self._handle, spec.dio, hw_state)
        else:
            self._start_pulse_burst(spec, state)

        with self._lock:
            self._actuators[name] = state

    def read_actuator(self, name: str) -> int:
        """Returns the cached software state of a given actuator."""
        with self._lock:
            return self._actuators[name]

    def all_safe(self) -> None:
        """
        De-energise every binary_dio line and cancel any stepper burst.

        Steppers keep their last commanded state: a cancelled burst leaves
        the valve where it is, so reporting 0 would claim a closed main.
        """
        self._safe_all_hardware()
        print("[T7] All actuators -> SAFE/CLOSED")

    def actuator_states(self) -> dict[str, ActuatorReading]:
        """Returns a copy of all current actuator states."""
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
        return getattr(self, "_serial", "UNKNOWN")

    @property
    def stream_time_s(self) -> float:
        """Now on the time_s clock (seconds since the first stream start)."""
        return time.perf_counter() - self._stream_start

    @property
    def stream_rate_hz(self) -> int:
        return self._actual_hz if self._actual_hz else self._stream_hz

    @property
    def sensor_tags(self) -> list[str]:
        """All sensor tags produced by stream_read()."""
        return [spec.id for spec in self._stream_channels]

    # --------------------------------------------------------
    # Private Hardware Configuration Helpers
    # --------------------------------------------------------

    def _configure_channels(self) -> None:
        """
        Set negative channel and range per sensor.

        Resolution is stream-wide (STREAM_RESOLUTION_INDEX in start_stream()).
        """
        for spec in self._stream_channels:
            if spec.type in (PT_DIRECT, PT_DIFFERENTIAL, LC_DIRECT, VOLTAGE):
                ljm.eWriteName(self._handle, f"{spec.ain}_NEGATIVE_CH", 199)
                ljm.eWriteName(self._handle, f"{spec.ain}_RANGE",       10.0)
            elif spec.type == TC_DIFFERENTIAL:
                neg_num = int(spec.ain_neg.replace("AIN", ""))
                ljm.eWriteName(self._handle, f"{spec.ain_pos}_NEGATIVE_CH", neg_num)
                ljm.eWriteName(self._handle, f"{spec.ain_pos}_RANGE",       0.1)

    def _configure_counters(self) -> None:
        """Puts each photogate DIO into DIO_EF Counter mode."""
        for spec in self._counter_channels:
            ljm.eWriteName(self._handle, f"{spec.dio}_EF_ENABLE", 0)
            ljm.eWriteName(self._handle, f"{spec.dio}_EF_INDEX",
                           _DIO_EF_INDEX_COUNTER)
            ljm.eWriteName(self._handle, f"{spec.dio}_EF_ENABLE", 1)

    def _configure_steppers(self) -> None:
        """
        Configure DIO_EF CLOCK0 once for every STEP line.

        It's never touched again (disabling it truncates a running burst), so
        all steppers must share one pulse_freq_hz; a mismatch raises.
        """
        steppers = [
            spec for spec in self._actuator_specs.values()
            if spec.type == PULSE_STEPPER and spec.is_wired
        ]
        if not steppers:
            return

        rates = {spec.pulse_freq_hz for spec in steppers}
        if len(rates) > 1:
            raise ValueError(
                f"All pulse_stepper actuators must share one pulse_freq_hz "
                f"(they share DIO_EF CLOCK0); actuators.yaml declares "
                f"{sorted(rates)}"
            )
        rate = rates.pop()
        if not rate or rate <= 0:
            raise ValueError(
                f"pulse_freq_hz must be a positive number; got {rate!r}"
            )
        self._pulse_roll = int(_DIO_EF_TICK_HZ / rate)

        ljm.eWriteName(self._handle, "DIO_EF_CLOCK0_ENABLE",  0)
        ljm.eWriteName(self._handle, "DIO_EF_CLOCK0_DIVISOR", _DIO_EF_CLOCK_DIV)
        ljm.eWriteName(self._handle, "DIO_EF_CLOCK0_ROLL_VALUE", self._pulse_roll)
        ljm.eWriteName(self._handle, "DIO_EF_CLOCK0_ENABLE",  1)

        for spec in steppers:
            ljm.eWriteName(self._handle, f"{spec.step_dio}_EF_ENABLE", 0)
            if spec.ena_dio is not None:
                ljm.eWriteName(self._handle, spec.ena_dio, 1)   # start disabled

    def _start_pulse_burst(self, spec: ActuatorSpec, state: int) -> None:
        """
        Arm a fixed-count Pulse Out burst (open-loop) on one STEP line.

        Leaves CLOCK0 alone so the other stepper's burst isn't cut short.
        """
        steps = spec.steps_open if state == 1 else spec.steps_close
        roll  = self._pulse_roll

        # DIR first, so it's settled before the first STEP edge.
        ljm.eWriteName(self._handle, spec.dir_dio, 1 if state == 1 else 0)
        if spec.ena_dio is not None:
            ljm.eWriteName(self._handle, spec.ena_dio, 0)   # active-low enable

        step_dio = spec.step_dio
        ljm.eWriteName(self._handle, f"{step_dio}_EF_ENABLE",   0)
        ljm.eWriteName(self._handle, f"{step_dio}_EF_INDEX",    _DIO_EF_INDEX_PULSE_OUT)
        ljm.eWriteName(self._handle, f"{step_dio}_EF_CONFIG_A", roll // 2)  # 50% duty
        ljm.eWriteName(self._handle, f"{step_dio}_EF_CONFIG_B", 0)          # no phase offset
        ljm.eWriteName(self._handle, f"{step_dio}_EF_CONFIG_C", steps)
        ljm.eWriteName(self._handle, f"{step_dio}_EF_ENABLE",   1)

        with self._lock:
            self._move_deadline[spec.id] = (
                time.perf_counter() + steps / spec.pulse_freq_hz
            )

    def _build_scan_list(self) -> None:
        """
        Build the eStreamStart address list.

        A TC streams its positive AIN only (NEGATIVE_CH sets the pair). The
        CJC and counters are polled instead; TEMPERATURE_DEVICE_K can't stream.
        """
        names: list[str] = []
        tags:  list[str] = []

        for spec in self._stream_channels:
            tags.append(spec.id)
            names.append(spec.ain_pos if spec.type == TC_DIFFERENTIAL else spec.ain)

        self._scan_tags   = tags
        self._scan_addrs  = [ljm.nameToAddress(name)[0] for name in names]
        self._n_channels  = len(names)

    def _safe_all_hardware(self) -> None:
        """
        Drive every binary_dio line high (de-energised) and stop stepper
        bursts. The cached states drop to 0 only if every write succeeded.
        """
        if self._handle is None:
            return
        all_ok = True
        for spec in self._actuator_specs.values():
            try:
                if spec.type == BINARY_DIO and spec.dio is not None:
                    ljm.eWriteName(self._handle, spec.dio, 1)   # active-low: 1 = safe
                elif spec.type == PULSE_STEPPER and spec.is_wired:
                    ljm.eWriteName(self._handle, f"{spec.step_dio}_EF_ENABLE", 0)
                    if spec.ena_dio is not None:
                        ljm.eWriteName(self._handle, spec.ena_dio, 1)   # de-assert enable
            except ljm.LJMError:
                all_ok = False
        if all_ok:
            with self._lock:
                for name, spec in self._actuator_specs.items():
                    if spec.type == BINARY_DIO:
                        self._actuators[name] = 0
                self._move_deadline.clear()

    def _arm_watchdog(self) -> None:
        """
        Arm the watchdog to drive the binary_dio lines high on comms loss.
        The *_DEFAULT registers live in flash, so only differing ones are
        written. It stays armed after exit, so the cart fails safe.
        """
        h = self._handle
        if self._actuator_mask == 0:
            print("[T7] WARNING: no binary_dio actuator pins assigned - "
                  "watchdog NOT armed")
            return

        desired = {
            "WATCHDOG_TIMEOUT_S_DEFAULT":     _WATCHDOG_TIMEOUT_S,
            "WATCHDOG_RESET_ENABLE_DEFAULT":  0,   # no reboot on timeout
            "WATCHDOG_DIO_ENABLE_DEFAULT":    1,
            # Inhibit every line outside the mask.
            "WATCHDOG_DIO_INHIBIT_DEFAULT":   (~self._actuator_mask) & 0xFFFFFFFF,
            "WATCHDOG_DIO_DIRECTION_DEFAULT": self._actuator_mask,
            "WATCHDOG_DIO_STATE_DEFAULT":     self._actuator_mask,
        }
        changed = {
            name: value for name, value in desired.items()
            if int(ljm.eReadName(h, name)) != value
        }
        enabled = int(ljm.eReadName(h, "WATCHDOG_ENABLE_DEFAULT")) == 1

        if changed:
            # The watchdog must be disabled before its other registers change.
            ljm.eWriteName(h, "WATCHDOG_ENABLE_DEFAULT", 0)
            for name, value in changed.items():
                ljm.eWriteName(h, name, value)
        if changed or not enabled:
            ljm.eWriteName(h, "WATCHDOG_ENABLE_DEFAULT", 1)
            print(f"[T7] Watchdog armed: {_WATCHDOG_TIMEOUT_S} s timeout, "
                  f"valves and igniter -> safe on comms loss "
                  f"(stepper lines excluded)")
        else:
            print("[T7] Watchdog already configured; nothing written")
