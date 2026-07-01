"""
daq/hardware/interface.py
 
Production LabJack T7 driver wrapping the LJM library.
 
Implements the exact same public interface as hardware/mock.py so that
engine.py is hardware-agnostic. Mock.py must stay in sync vise versa.

Channel layout (must match mock.py and engine.py):
    PTs  - single-ended, AIN_NEGATIVE_CH=199, ±5 V range
    TCs  - differential pairs, ±0.1 V range
    LCs  - single-ended, AIN_NEGATIVE_CH=199, ±5 V range
    CJC  - LM34 on AIN58, single-ended, ±1 V range (read outside stream)
 
Actuators (active-low on EIO/CIO bank):
    hardware logical 0 = energised, logical 1 = de-energised
    We invert at this layer so the rest of the system uses:
        1 = open/energised, 0 = closed/safe
 
Watchdog:
    Armed during start_stream(). If the host loses comms for > 5 s the
    T7 drives all DIO lines to their safe (de-energised) defaults.
 
Reference:
    LabJack T7 User Guide - https://labjack.com/pages/support?doc=/datasheets/t7-datasheet/
    LJM Library - https://labjack.com/pages/support?doc=/software-driver/ljm-users-guide/
"""

from __future__ import annotations

import time
import threading
from typing import Optional

from labjack import ljm


# -- Hardware Channel Allocations -----------------------------
# PTs: Single-ended, GND reference (NEG=199), ±5V range.
# TCs: Differential pairs, ±0.1V range.
# LCs: Single-ended on AIN50, AIN51.
# CJC: LM34 on AIN58, ±1V range (polled outside stream).

_PT_CHANNELS: dict[str, str] = {
    "POT": "AIN122",   # LOX tank pressure
    "PFT": "AIN123",   # Fuel tank pressure
    "POI": "AIN124",   # LOX inlet pressure
    "PFI": "AIN125",   # Fuel channel inlet pressure
    "PFO": "AIN126",   # Fuel channel outlet pressure
    "PC":  "AIN127",   # Chamber pressure
    "PNS": "AIN52",    # System GN2 pressure
    "PNP": "AIN53",    # Pneumatics GN2 pressure
}

_TC_CHANNELS: dict[str, tuple[str, str]] = {
    "TOI": ("AIN48", "AIN56"),   # LOX inlet temperature (differential)
    "TFI": ("AIN0",  "AIN1"),    # Fuel inlet temperature
    "TFO": ("AIN2",  "AIN3"),    # Fuel outlet temperature
}

_LC_CHANNELS: dict[str, str] = {
    "LC_1": "AIN50",
    "LC_2": "AIN51",
}

_CJC_CHANNEL = "AIN58"   # LM34 cold junction sensor

# -- Actuator Allocations -------------------------------------
# Actuator DIO channels map to the active-low EIO/CIO bank (0=On, 1=Off).
# The driver inverts this state internally so callers use 1=Open, 0=Closed.

_ACTUATOR_CHANNELS: dict[str, str] = {
    "LOx Press":  "EIO0",
    "Fuel Press": "EIO1",
    "LOx Purge":  "EIO2",
    "Fuel Purge": "EIO3",
    "LOx Main":   "EIO4",
    "Fuel Main":  "EIO5",
    "LOx Vent":   "EIO6",
    "Fuel Vent":  "EIO7",
    "Ignition":   "CIO1",
}

# -- Stream & Safety Parameters -------------------------------
_STREAM_HZ       = 500
_SCANS_PER_READ  = 50     # yields ~10 batches/s, 0.1 s per read call
_SETTLING_US     = 100    # AIN settling time in microseconds
_RESOLUTION      = 3      # resolution index (1=fastest, 12=slowest/quietest)

# Hardware watchdog safety settings (T7 executes autonomously on host crash)
_WATCHDOG_TIMEOUT_S = 5

_EIO_MASK = 0x0000FF00
_CIO1_MASK = 0x00020000
_ACTUATOR_MASK = _EIO_MASK | _CIO1_MASK


class LabJackT7:
    """
    Production LabJack T7 driver.

    Public interface is identical to MockLabJack. Only hardware-specific
    implementation details are documented here.

    Usage:
        device = LabJackT7()
        device.open()
        device.start_stream()

        while running:
            batch = device.stream_read()
            cjc_v = device.read_cjc()
            device.write_actuator("LOx Main", state=1)

        device.stop_stream()
        device.close()
    """

    def __init__(self, stream_hz: int = _STREAM_HZ) -> None:
        self._stream_hz      = stream_hz
        self._scans_per_read = _SCANS_PER_READ

        self._handle: Optional[int] = None
        self._lock = threading.Lock()

        self._actuators: dict[str, int] = {
            name: 0 for name in _ACTUATOR_CHANNELS
        }

        self._connected  = False
        self._streaming  = False
        self._stream_start: float = 0.0
        self._actual_hz: int = 0

        # Pre-compile the scan list addresses for stream efficiency
        self._scan_tags:    list[str] = []
        self._scan_addrs:   list[int] = []
        self._scan_types:   list[int] = []
        self._n_channels:   int = 0

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
            RuntimeError: If streaming cannot be started at any rate >= 10 Hz.
            ljm.LJMError: On unexpected LJM errors.
        """
        self._build_scan_list()

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
            Dict: sensor tag -> list[float] raw voltages (one per scan).
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

    # --------------------------------------------------------
    # Actuator control
    # --------------------------------------------------------

    def write_actuator(self, name: str, state: int) -> None:
        """
        Writes state to actuator, applying active-low hardware inversion.

        Args:
            name:  Actuator name. Must be a key in _ACTUATOR_CHANNELS.
            state: 1 = energise/open, 0 = de-energise/safe.

        Raises:
            KeyError:      If name is not a recognised actuator.
            ljm.LJMError:  If the LJM write fails.
        """
        if name not in _ACTUATOR_CHANNELS:
            raise KeyError(f"Unknown actuator: '{name}'")

        hw_state = 0 if state == 1 else 1   # Active-low hardware inversion :>
        ljm.eWriteName(self._handle, _ACTUATOR_CHANNELS[name], hw_state)

        with self._lock:
            self._actuators[name] = state

    def read_actuator(self, name: str) -> int:
        """Returns the cached software state of a given actuator."""
        with self._lock:
            return self._actuators[name]

    def all_safe(self) -> None:
        """De-energise all actuators immediately."""
        self._safe_all_hardware()
        with self._lock:
            for name in self._actuators:
                self._actuators[name] = 0
        print("[T7] All actuators -> SAFE/CLOSED")

    def actuator_states(self) -> dict[str, int]:
        """Returns a copy of all current actuator states."""
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
        return getattr(self, "_serial", "UNKNOWN")

    @property
    def stream_rate_hz(self) -> int:
        return self._actual_hz if self._actual_hz else self._stream_hz

    @property
    def sensor_tags(self) -> list[str]:
        """All sensor tags produced by stream_read()."""
        return list(_PT_CHANNELS) + list(_TC_CHANNELS) + list(_LC_CHANNELS)

    # --------------------------------------------------------
    # Private Hardware Configuration Helpers
    # --------------------------------------------------------

    def _configure_channels(self) -> None:
        """
        Set AIN range, resolution, and negative channel for every sensor.

        PTs and LCs:
            Single-ended (NEGATIVE_CH = 199 = GND reference)
            ±5 V range, resolution index 1 (fastest)

        TCs:
            Differential (NEGATIVE_CH = partner channel number)
            ±0.1 V range, resolution index 3 (quieter for small signal)

        CJC (LM34 on AIN58):
            Single-ended, ±1 V range, resolution 4
        """
        for ch in list(_PT_CHANNELS.values()) + list(_LC_CHANNELS.values()):
            ljm.eWriteName(self._handle, f"{ch}_NEGATIVE_CH",    199)
            ljm.eWriteName(self._handle, f"{ch}_RANGE",          5.0)
            ljm.eWriteName(self._handle, f"{ch}_RESOLUTION_INDEX", 1)

        for pos_ch, neg_ch in _TC_CHANNELS.values():
            neg_num = int(neg_ch.replace("AIN", ""))
            ljm.eWriteName(self._handle, f"{pos_ch}_NEGATIVE_CH",    neg_num)
            ljm.eWriteName(self._handle, f"{pos_ch}_RANGE",          0.1)
            ljm.eWriteName(self._handle, f"{pos_ch}_RESOLUTION_INDEX", 3)

        ljm.eWriteName(self._handle, f"{_CJC_CHANNEL}_NEGATIVE_CH",    199)
        ljm.eWriteName(self._handle, f"{_CJC_CHANNEL}_RANGE",          1.0)
        ljm.eWriteName(self._handle, f"{_CJC_CHANNEL}_RESOLUTION_INDEX", 4)

    def _build_scan_list(self) -> None:
        """
        Build the ordered address list passed to eStreamStart.

        Only the positive channel of each TC pair goes into the stream.
        The negative channel is configured via NEGATIVE_CH register,
        not as a separate scan slot.

        CJC (AIN58) is intentionally excluded for latency improv.
        """
        names: list[str] = []
        tags:  list[str] = []

        for tag, ch in _PT_CHANNELS.items():
            tags.append(tag)
            names.append(ch)

        for tag, (pos_ch, _) in _TC_CHANNELS.items():
            tags.append(tag)
            names.append(pos_ch)

        for tag, ch in _LC_CHANNELS.items():
            tags.append(tag)
            names.append(ch)

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
        """Forces all actuator DIO lines high (safe/de-energized state)."""
        if self._handle is None:
            return
        for ch in _ACTUATOR_CHANNELS.values():
            try:
                ljm.eWriteName(self._handle, ch, 1)   # active-low: 1 = safe
            except ljm.LJMError:
                pass

    def _arm_watchdog(self) -> None:
        """Arms the T7 hardware watchdog to safe outputs autonomously on host crash."""
        h = self._handle
        ljm.eWriteName(h, "WATCHDOG_ENABLE_DEFAULT",        0)
        ljm.eWriteName(h, "WATCHDOG_TIMEOUT_S_DEFAULT",     _WATCHDOG_TIMEOUT_S)
        ljm.eWriteName(h, "WATCHDOG_DIO_ENABLE_DEFAULT",    1)
        
        # Bits not in actuator mask are inhibited (ignored by watchdog)
        inhibit = (~_ACTUATOR_MASK) & 0xFFFFFFFF
        ljm.eWriteName(h, "WATCHDOG_DIO_INHIBIT_DEFAULT",   inhibit)
        ljm.eWriteName(h, "WATCHDOG_DIO_DIRECTION_DEFAULT", _ACTUATOR_MASK)
        ljm.eWriteName(h, "WATCHDOG_DIO_STATE_DEFAULT",     _ACTUATOR_MASK)
        ljm.eWriteName(h, "WATCHDOG_ENABLE_DEFAULT",        1)
        print(f"[T7] Watchdog armed: {_WATCHDOG_TIMEOUT_S} s timeout, "
              f"all actuators -> safe on comms loss")