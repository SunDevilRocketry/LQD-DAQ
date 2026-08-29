"""
daq/hardware/interface.py

Production LabJack T7 driver wrapping the LJM library.

Implements the exact same public interface as hardware/mock.py so that
engine.py is hardware-agnostic. Mock.py must stay in sync vise versa.

Channel and actuator layout is built at construction time from the 
ChannelSpec/ActuatorSpec manifests loaded from
channels.yaml / actuators.yaml:
    pt_direct         - single-ended, AIN_NEGATIVE_CH=199, ±5 V range
    tc_differential   - differential pair, ±0.1 V range
    lc_direct         - single-ended, AIN_NEGATIVE_CH=199, ±5 V range
    photogate_counter - DIO_EF Counter, polled outside the stream
    CJC               - LM34 on AIN58, single-ended, ±1 V range (read outside stream)

A channel or actuator whose physical pin is still null in the manifest is
skipped: it is not added to the scan list and cannot be driven.

Actuators:
    binary_dio    - active-low on the EIO/CIO bank
                    hardware logical 0 = energised, logical 1 = de-energised
                    We invert at this layer so the rest of the system uses:
                        1 = open/energised, 0 = closed/safe
    pulse_stepper - DM860T via DIO_EF Pulse Out (mode 2) on the STEP line,
                    with DIR/ENA as static DIO writes. Hardware generates a
                    fixed-count pulse burst from one register write; no CPU
                    polling once started. Open-loop: the burst is commanded,
                    not verified.

Watchdog:
    Armed during open(). If the host loses comms for > 5 s the T7 drives the
    covered DIO lines to their safe (de-energised) defaults.

    Coverage is scoped to binary_dio (solenoid) lines only. 
    !!!pulse_stepper lines are excluded until the watchdog-vs-DIO_EF-Pulse-Out
    interaction has been bench-tested. !!! watchdog reset firing mid-burst on
    a line the pulse engine owns is untested behaviour.

Reference:
    LabJack T7 User Guide - https://labjack.com/pages/support?doc=/datasheets/t7-datasheet/
    LJM Library - https://labjack.com/pages/support?doc=/software-driver/ljm-users-guide/
    DIO_EF Pulse Out - https://support.labjack.com/docs/13-2-4-pulse-out-t-series-datasheet
    Stepper control app note - https://support.labjack.com/docs/stepper-motor-controller
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
    TC_DIFFERENTIAL,
    LC_DIRECT,
    PHOTOGATE_COUNTER,
)


_CJC_CHANNEL = "AIN58"   # LM34 cold junction sensor

# -- Stream & Safety Parameters -------------------------------
_STREAM_HZ       = 500
_SCANS_PER_READ  = 50     # yields ~10 batches/s, 0.1 s per read call
_SETTLING_US     = 100    # AIN settling time in microseconds
_RESOLUTION      = 3      # resolution index (1=fastest, 12=slowest/quietest)

# Hardware watchdog safety settings (T7 executes autonomously on host crash)
_WATCHDOG_TIMEOUT_S = 5

# -- DIO_EF Parameters ----------------------------------------
# T7 core clock is 80 MHz. Divisor 8 gives a 10 MHz tick, which keeps the
# roll value inside 32 bits across the whole stepper rate range we care
# about (10 Hz -> 1 M ticks, 20 kHz -> 500 ticks).
_DIO_EF_CORE_HZ      = 80_000_000
_DIO_EF_CLOCK_DIV    = 8
_DIO_EF_TICK_HZ      = _DIO_EF_CORE_HZ // _DIO_EF_CLOCK_DIV

_DIO_EF_INDEX_PULSE_OUT = 2
_DIO_EF_INDEX_COUNTER   = 8

# DIO bit offsets for the T7's named banks (T7 datasheet, DIO numbering).
_DIO_BANK_OFFSETS = {"FIO": 0, "EIO": 8, "CIO": 16, "MIO": 20}


def _dio_number(name: str) -> int:
    """
    Resolve a T7 DIO register name to its DIO number.

    Accepts either the bank form ('EIO0', 'CIO1') or the flat form
    ('DIO8'). Needed b/c the watchdog mask registers are bitmasks over
    flat DIO numbers, while the manifest names lines the way the wiring
    diagram does.

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
    """
    Production LabJack T7 driver.

    Public interface is identical to MockLabJack. Only hardware-specific
    implementation details are documented here.

    Usage:
        device = LabJackT7(channels, actuators)
        device.open()
        device.start_stream()

        while running:
            batch = device.stream_read()
            cjc_v = device.read_cjc()
            device.write_actuator("lox_main", state=1)

        device.stop_stream()
        device.close()
    """

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

        # Only channels the manifest marks active AND names a pin for can be
        # touched. Everything else stays in the manifest but is not wired.
        self._channels = [
            spec for spec in channels if spec.active and spec.is_wired
        ]
        self._stream_channels = [
            spec for spec in self._channels if spec.is_streamed
        ]
        self._counter_channels = [
            spec for spec in self._channels if spec.type == PHOTOGATE_COUNTER
        ]

        self._actuator_specs: dict[str, ActuatorSpec] = {
            spec.id: spec for spec in actuators
        }
        self._actuators: dict[str, int] = {
            spec.id: 0 for spec in actuators
        }
        # Monotonic deadline per stepper; a commanded burst is still in
        # flight until perf_counter() passes it.
        self._move_deadline: dict[str, float] = {}

        self._warn_unwired(channels, actuators)

        self._connected  = False
        self._streaming  = False
        self._stream_start: float = 0.0
        self._actual_hz: int = 0

        # Pre-compile the scan list addresses for stream efficiency
        self._scan_tags:    list[str] = []
        self._scan_addrs:   list[int] = []
        self._scan_types:   list[int] = []
        self._n_channels:   int = 0

        # Shared DIO_EF CLOCK0 roll value, resolved once in
        # _configure_steppers() from the single stepper pulse rate.
        self._pulse_roll: int = 0

        # Watchdog covers solenoid lines only (see module docstring).
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

    def read_cjc(self) -> float:
        """
        Reads the raw cold junction temperature reference (AIN58).

        Returns:
            Voltage in volts (~0.770 V at 25 °C room temperature).
        """
        return ljm.eReadName(self._handle, _CJC_CHANNEL)

    def read_counters(self) -> dict[str, float]:
        """
        Reads the photogate position-feedback counters.

        Polled out-of-band on the same tick as read_cjc(), not
        part of the AIN scan list. Informational telemetry only.

        Returns:
            Dict: channel id -> accumulated edge count. Channels whose read
            fails are omitted rather than reported as a bogus zero.
        """
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
        Commands an actuator to its open (1) or closed/safe (0) position.

        binary_dio actuators are a single DIO write with the active-low
        hardware inversion applied here. pulse_stepper actuators start a
        fixed-count DIO_EF pulse burst; the call returns as soon as the
        burst is armed, and the actuator reports moving=True until the
        commanded burst duration has elapsed.

        Args:
            name:  Actuator id from actuators.yaml.
            state: 1 = energise/open, 0 = de-energise/safe.

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
            hw_state = 0 if state == 1 else 1   # Active-low hardware inversion :>
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
        De-energise all hardware outputs immediately.

        Solenoids drop to their safe state. Steppers have any in-flight
        pulse burst cancelled and their driver disabled, which leaves the
        valve wherever it currently sits.
        An open-loop stepper can't be shut immediately, ordered closure of the
        mains is abort.yaml's job, which Engine.abort() runs before reaching here.

        B/c of that, only binary_dio actuators are reported closed here.
        A stepper keeps its last commanded state: forcing it to 0 would tell
        the operator the main is shut at the exact moment we have deliberately
        left it wherever it sat. `moving` does drop to false for every
        actuator, since any in-flight burst really has been cancelled.
        """
        self._safe_all_hardware()
        with self._lock:
            for name, spec in self._actuator_specs.items():
                if spec.type == BINARY_DIO:
                    self._actuators[name] = 0
            self._move_deadline.clear()
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
        Set AIN range, resolution, and negative channel for every sensor.

        pt_direct / lc_direct:
            Single-ended (NEGATIVE_CH = 199 = GND reference)
            ±5 V range, resolution index 1 (fastest)

        tc_differential:
            Differential (NEGATIVE_CH = partner channel number)
            ±0.1 V range, resolution index 3 (quieter for small signal)

        CJC (LM34 on AIN58):
            Single-ended, ±1 V range, resolution 4
        """
        for spec in self._stream_channels:
            if spec.type in (PT_DIRECT, LC_DIRECT):
                ljm.eWriteName(self._handle, f"{spec.ain}_NEGATIVE_CH",    199)
                ljm.eWriteName(self._handle, f"{spec.ain}_RANGE",          5.0)
                ljm.eWriteName(self._handle, f"{spec.ain}_RESOLUTION_INDEX", 1)
            elif spec.type == TC_DIFFERENTIAL:
                neg_num = int(spec.ain_neg.replace("AIN", ""))
                ljm.eWriteName(self._handle, f"{spec.ain_pos}_NEGATIVE_CH",    neg_num)
                ljm.eWriteName(self._handle, f"{spec.ain_pos}_RANGE",          0.1)
                ljm.eWriteName(self._handle, f"{spec.ain_pos}_RESOLUTION_INDEX", 3)

        ljm.eWriteName(self._handle, f"{_CJC_CHANNEL}_NEGATIVE_CH",    199)
        ljm.eWriteName(self._handle, f"{_CJC_CHANNEL}_RANGE",          1.0)
        ljm.eWriteName(self._handle, f"{_CJC_CHANNEL}_RESOLUTION_INDEX", 4)

    def _configure_counters(self) -> None:
        """Puts each photogate DIO into DIO_EF Counter mode."""
        for spec in self._counter_channels:
            ljm.eWriteName(self._handle, f"{spec.dio}_EF_ENABLE", 0)
            ljm.eWriteName(self._handle, f"{spec.dio}_EF_INDEX",
                           _DIO_EF_INDEX_COUNTER)
            ljm.eWriteName(self._handle, f"{spec.dio}_EF_ENABLE", 1)

    def _configure_steppers(self) -> None:
        """
        Configures the shared DIO_EF clock used by every STEP line.

        The clock is set up once; each burst then only has to write that
        actuator's own pulse-count/width registers. DIR and ENA are plain
        DIO lines and need no EF configuration.

        CLOCK0 is a single shared resource. It is configured and enabled
        exactly once here and never touched again, because disabling it
        while any STEP line is mid-burst truncates that burst.

        The consequence is that every stepper must share one
        pulse_freq_hz. A mismatch is rejected here rather than silently
        running whichever stepper was armed second at the wrong rate.
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
            ljm.eWriteName(self._handle, spec.ena_dio, 1)   # start disabled

    def _start_pulse_burst(self, spec: ActuatorSpec, state: int) -> None:
        """
        Arms a fixed-count DIO_EF Pulse Out burst on a stepper's STEP line.

        Open-loop and fire-and-forget (T7's pulse engine emits the whole
        burst in hardware).

        Touches only this actuator's own EF registers. The shared CLOCK0 is
        owned by _configure_steppers() and must not be disturbed here, or a
        burst already running on the other STEP line would be cut short.
        """
        steps = spec.steps_open if state == 1 else spec.steps_close
        roll  = self._pulse_roll

        # DIR first so the line is settled b/f the first STEP edge.
        ljm.eWriteName(self._handle, spec.dir_dio, 1 if state == 1 else 0)
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
        Build the ordered address list passed to eStreamStart.

        Only the positive channel of each TC pair goes into the stream.
        The negative channel is configured via NEGATIVE_CH register,
        not as a separate scan slot.

        CJC (AIN58) and the photogate counters are intentionally excluded;
        both are polled out-of-band.
        """
        names: list[str] = []
        tags:  list[str] = []

        for spec in self._stream_channels:
            tags.append(spec.id)
            names.append(spec.ain_pos if spec.type == TC_DIFFERENTIAL else spec.ain)

        addrs = [0] * len(names)
        types = [0] * len(names)
        for i, name in enumerate(names):
            info      = ljm.nameToAddress(name)
            addrs[i]  = info[0]
            types[i]  = info[1]

        self._scan_tags   = tags
        self._scan_addrs  = addrs
        self._scan_types  = types
        self._n_channels  = len(names)

    def _safe_all_hardware(self) -> None:
        """
        Forces every solenoid DIO line high (safe/de-energized state) and
        cancels any in-flight stepper burst.
        """
        if self._handle is None:
            return
        for spec in self._actuator_specs.values():
            try:
                if spec.type == BINARY_DIO and spec.dio is not None:
                    ljm.eWriteName(self._handle, spec.dio, 1)   # active-low: 1 = safe
                elif spec.type == PULSE_STEPPER and spec.is_wired:
                    ljm.eWriteName(self._handle, f"{spec.step_dio}_EF_ENABLE", 0)
                    ljm.eWriteName(self._handle, spec.ena_dio, 1)   # de-assert enable
            except ljm.LJMError:
                pass

    def _arm_watchdog(self) -> None:
        """
        Arms the T7 hardware watchdog to safe outputs on host crash.

        The covered mask holds binary_dio lines only.
        """
        h = self._handle
        if self._actuator_mask == 0:
            print("[T7] WARNING: no binary_dio actuator pins assigned - "
                  "watchdog NOT armed")
            return

        ljm.eWriteName(h, "WATCHDOG_ENABLE_DEFAULT",        0)
        ljm.eWriteName(h, "WATCHDOG_TIMEOUT_S_DEFAULT",     _WATCHDOG_TIMEOUT_S)
        ljm.eWriteName(h, "WATCHDOG_DIO_ENABLE_DEFAULT",    1)

        # Bits not in actuator mask are inhibited (ignored by watchdog)
        inhibit = (~self._actuator_mask) & 0xFFFFFFFF
        ljm.eWriteName(h, "WATCHDOG_DIO_INHIBIT_DEFAULT",   inhibit)
        ljm.eWriteName(h, "WATCHDOG_DIO_DIRECTION_DEFAULT", self._actuator_mask)
        ljm.eWriteName(h, "WATCHDOG_DIO_STATE_DEFAULT",     self._actuator_mask)
        ljm.eWriteName(h, "WATCHDOG_ENABLE_DEFAULT",        1)
        print(f"[T7] Watchdog armed: {_WATCHDOG_TIMEOUT_S} s timeout, "
              f"solenoids -> safe on comms loss "
              f"(stepper lines excluded by design)")
