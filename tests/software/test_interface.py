"""
tests/software/test_interface.py

The real T7 driver with LJM writes recorded instead of sent; no device.
"""

import time

import pytest
import yaml

from daq.hardware import interface
from daq.hardware.interface import LabJackT7, _dio_number
from daq.manifest import (
    ActuatorSpec, ChannelSpec, BINARY_DIO, PT_DIRECT, PULSE_STEPPER, TC_DIFFERENTIAL,
    load_actuators, load_channels,
)

from tests.software._helpers import REPO_ACTUATORS, REPO_CHANNELS


@pytest.fixture
def writes(monkeypatch):
    """Record every ljm.eWriteName call as (register, value)."""
    calls = []
    monkeypatch.setattr(
        interface.ljm, "eWriteName",
        lambda handle, name, value: calls.append((name, value)),
    )
    return calls


def _fake_t7(monkeypatch, registers):
    """Back eReadName/eWriteName with a dict; returns the list of writes."""
    written = []
    def write(handle, name, value):
        written.append(name)
        registers[name] = value
    monkeypatch.setattr(interface.ljm, "eWriteName", write)
    monkeypatch.setattr(
        interface.ljm, "eReadName", lambda handle, name: registers.get(name, 0)
    )
    return written


def _mask(*names):
    bits = 0
    for name in names:
        bits |= 1 << _dio_number(name)
    return bits


def _stepper(**extra):
    return ActuatorSpec(
        id="lox_main", type=PULSE_STEPPER,
        step_dio="FIO3", dir_dio="FIO2",
        steps_open=400, steps_close=400, pulse_freq_hz=2000, **extra,
    )


class TestWatchdogMask:

    def test_shipped_manifest_covers_the_relay_and_igniter_lines(self, monkeypatch):
        device = LabJackT7(
            load_channels(REPO_CHANNELS), load_actuators(REPO_ACTUATORS)
        )
        expected = _mask("EIO1", "EIO2", "EIO3", "EIO4", "EIO7", "CIO1",
                         "MIO0", "MIO1")
        assert device._actuator_mask == expected

        device._handle = 1
        registers = {"WATCHDOG_RESET_ENABLE_DEFAULT": 1}   # left on in flash
        _fake_t7(monkeypatch, registers)
        device._arm_watchdog()
        assert registers["WATCHDOG_DIO_DIRECTION_DEFAULT"] == expected
        assert registers["WATCHDOG_DIO_STATE_DEFAULT"] == expected
        assert registers["WATCHDOG_DIO_INHIBIT_DEFAULT"] == (~expected) & 0xFFFFFFFF
        assert registers["WATCHDOG_RESET_ENABLE_DEFAULT"] == 0
        assert registers["WATCHDOG_ENABLE_DEFAULT"] == 1

    def test_rearming_an_unchanged_config_writes_nothing(self, monkeypatch):
        device = LabJackT7([], load_actuators(REPO_ACTUATORS))
        device._handle = 1
        registers = {}
        written = _fake_t7(monkeypatch, registers)
        device._arm_watchdog()
        written.clear()

        device._arm_watchdog()

        assert written == [], f"flash written again: {written}"


class TestOptionalStepperEnable:

    def test_stepper_without_ena_is_wired(self):
        assert _stepper().is_wired

    def test_no_ena_line_is_ever_written_when_ena_is_unset(self, writes):
        device = LabJackT7([], [_stepper()])
        device._handle = 1
        device._configure_steppers()
        device._start_pulse_burst(device._actuator_specs["lox_main"], 1)
        device._safe_all_hardware()

        bare_lines = {name for name, _ in writes if "_" not in name}
        assert bare_lines == {"FIO2"}, "only the DIR line is a plain DIO write"

    def test_ena_line_is_still_driven_when_set(self, writes):
        device = LabJackT7([], [_stepper(ena_dio="FIO4")])
        device._handle = 1
        device._configure_steppers()
        device._start_pulse_burst(device._actuator_specs["lox_main"], 1)
        device._safe_all_hardware()

        assert [v for name, v in writes if name == "FIO4"] == [1, 0, 1]


class TestColdJunction:

    def test_reads_internal_sensor_with_screw_terminal_correction(self, monkeypatch):
        reads = []
        def fake_read(handle, name):
            reads.append(name)
            return 298.15
        monkeypatch.setattr(interface.ljm, "eReadName", fake_read)

        device = LabJackT7([], [])
        device._handle = 1
        # 298.15 K = 25 C, minus LabJack's 3 C screw-terminal correction.
        assert device.read_cjc_celsius() == pytest.approx(22.0)
        assert reads == ["TEMPERATURE_DEVICE_K"]

    def test_configuring_shipped_channels_touches_no_extended_channel(self, writes):
        device = LabJackT7(load_channels(REPO_CHANNELS), [])
        device._handle = 1
        device._configure_channels()

        for name, _ in writes:
            if name.startswith("AIN"):
                number = int(name[3:].split("_")[0])
                assert number <= 15, f"{name} is an extended channel"


class TestExtendedChannelGuard:
    """AIN16+ reads drive MIO0-2, which carry the igniter lines."""

    @pytest.mark.parametrize("field", ["ain", "ain_pos", "ain_neg"])
    def test_manifest_refuses_extended_channel(self, tmp_path, field):
        entry = {"id": "x", "type": "tc_differential", "unit": "celsius",
                 "ain_pos": "AIN2", "ain_neg": "AIN3"}
        if field == "ain":
            entry = {"id": "x", "type": "pt_direct", "unit": "Pa"}
        entry[field] = "AIN58"
        path = tmp_path / "channels.yaml"
        path.write_text(yaml.safe_dump({"channels": [entry]}))
        with pytest.raises(ValueError, match="extended channel"):
            load_channels(str(path))

    def test_manifest_refuses_integer_extended_channel(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text(yaml.safe_dump({"channels": [
            {"id": "x", "type": "pt_direct", "unit": "Pa", "ain": 16},
        ]}))
        with pytest.raises(ValueError, match="extended channel"):
            load_channels(str(path))

    def test_manifest_accepts_ain15(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text(yaml.safe_dump({"channels": [
            {"id": "x", "type": "pt_direct", "unit": "Pa", "ain": "AIN15"},
        ]}))
        assert load_channels(str(path))[0].ain == "AIN15"

    def test_driver_refuses_extended_channel(self):
        spec = ChannelSpec(id="x", type=PT_DIRECT, unit="Pa", ain="AIN58")
        with pytest.raises(ValueError, match="extended channel"):
            LabJackT7([spec], [])


class TestChannelConfig:

    def test_shipped_channels_get_range_and_negative_channel_only(self, writes):
        device = LabJackT7(load_channels(REPO_CHANNELS), [])
        device._handle = 1
        device._configure_channels()
        regs = dict(writes)

        assert regs["AIN0_RANGE"] == 10.0 and regs["AIN0_NEGATIVE_CH"] == 199
        assert regs["AIN1_RANGE"] == 10.0          # load cell, 0-10 V
        assert regs["AIN2_RANGE"] == 0.1 and regs["AIN2_NEGATIVE_CH"] == 3
        assert regs["AIN13_RANGE"] == 10.0 and regs["AIN13_NEGATIVE_CH"] == 199
        assert not any(name.endswith("_RESOLUTION_INDEX") for name, _ in writes)


class TestStreamClock:

    def test_stream_clock_survives_a_restart(self, monkeypatch, writes):
        """A reconnect restarts the stream; time_s must keep counting."""
        monkeypatch.setattr(interface.ljm, "nameToAddress", lambda name: (0, 3))
        monkeypatch.setattr(interface.ljm, "eStreamStart", lambda *args: 500.0)
        monkeypatch.setattr(interface.ljm, "eStreamStop", lambda handle: None)
        device = LabJackT7(load_channels(REPO_CHANNELS), [])
        device._handle = 1

        device.start_stream()
        started = device._stream_start
        time.sleep(0.01)
        device.stop_stream()
        device.start_stream()

        assert device._stream_start == started


class TestSafingCache:
    """Reported states drop to 0 only when the safing writes reached the T7."""

    def _device(self):
        device = LabJackT7([], load_actuators(REPO_ACTUATORS))
        device._handle = 1
        device._actuators["fuel_vent"] = 1
        return device

    def test_successful_safing_clears_the_cache(self, writes):
        device = self._device()
        device._safe_all_hardware()
        assert device.read_actuator("fuel_vent") == 0

    def test_failed_safing_keeps_the_cache(self, monkeypatch):
        def fail(handle, name, value):
            raise interface.ljm.LJMError(errorCode=1, errorString="comms lost")
        monkeypatch.setattr(interface.ljm, "eWriteName", fail)
        device = self._device()
        device._safe_all_hardware()
        assert device.read_actuator("fuel_vent") == 1


class _FakeLJM:
    """The ljm calls on the open/stream/close path, backed by a register dict."""

    HANDLE = 7
    SERIAL = 470012345

    def __init__(self):
        self.registers = {}
        self.events = []          # ("write", name, value) / ("stop",) / ("close",)
        self.stream_starts = []   # (scans_per_read, n_channels, addrs, hz)
        self.max_hz = None        # above this, eStreamStart raises SCAN_OVERLAP
        self.start_error = None
        self.close_error = None
        self.reads = []           # queued eStreamRead returns

    def install(self, monkeypatch):
        for name in ("openS", "getHandleInfo", "eWriteName", "eReadName",
                     "nameToAddress", "eStreamStart", "eStreamRead",
                     "eStreamStop", "close"):
            monkeypatch.setattr(interface.ljm, name, getattr(self, name))

    def writes(self):
        return [(e[1], e[2]) for e in self.events if e[0] == "write"]

    def openS(self, device_type, connection_type, identifier):
        return self.HANDLE

    def getHandleInfo(self, handle):
        return (7, 3, self.SERIAL, 0, 502, 1040)

    def eWriteName(self, handle, name, value):
        self.events.append(("write", name, value))
        self.registers[name] = value

    def eReadName(self, handle, name):
        return self.registers.get(name, 0)

    def nameToAddress(self, name):
        return (2 * int(name[3:]), 3)   # AINn is Modbus address 2n

    def eStreamStart(self, handle, scans_per_read, n_channels, addrs, hz):
        self.stream_starts.append((scans_per_read, n_channels, list(addrs), hz))
        if self.start_error is not None:
            raise self.start_error
        if self.max_hz is not None and hz > self.max_hz:
            raise interface.ljm.LJMError(errorCode=2942,
                                         errorString="STREAM_SCAN_OVERLAP")
        return float(hz)

    def eStreamRead(self, handle):
        return self.reads.pop(0)

    def eStreamStop(self, handle):
        self.events.append(("stop",))

    def close(self, handle):
        self.events.append(("close",))
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def fake_ljm(monkeypatch):
    fake = _FakeLJM()
    fake.install(monkeypatch)
    return fake


def _pt_and_tc():
    return [
        ChannelSpec(id="pt0", type=PT_DIRECT, unit="psi", ain="AIN0"),
        ChannelSpec(id="tc0", type=TC_DIFFERENTIAL, unit="celsius",
                    ain_pos="AIN2", ain_neg="AIN3"),
    ]


class TestStreamStart:

    def test_scan_list_follows_the_shipped_manifest(self, fake_ljm):
        channels = load_channels(REPO_CHANNELS)
        device = LabJackT7(channels, [])
        device._handle = _FakeLJM.HANDLE

        assert device.start_stream() == 500

        streamed = [c for c in channels if c.active and c.is_wired and c.is_streamed]
        pins = [c.ain_pos if c.type == TC_DIFFERENTIAL else c.ain for c in streamed]
        scans, n, addrs, hz = fake_ljm.stream_starts[-1]
        assert addrs == [2 * int(pin[3:]) for pin in pins]
        assert n == len(pins) and hz == 500 and scans == 25
        assert device.sensor_tags == [c.id for c in streamed]
        assert "tc0" in device.sensor_tags and "pos_lox_main" not in device.sensor_tags
        assert device.is_streaming

    def test_stream_settings_are_written_before_the_start(self, fake_ljm):
        device = LabJackT7(_pt_and_tc(), [])
        device._handle = _FakeLJM.HANDLE
        device.start_stream()
        regs = dict(fake_ljm.writes())
        assert regs["STREAM_SETTLING_US"] == interface._SETTLING_US
        assert regs["STREAM_RESOLUTION_INDEX"] == interface._RESOLUTION

    def test_scan_overlap_halves_the_rate_until_it_fits(self, fake_ljm):
        fake_ljm.max_hz = 150
        device = LabJackT7(_pt_and_tc(), [], stream_hz=500)
        device._handle = _FakeLJM.HANDLE

        assert device.start_stream() == 125
        assert [s[3] for s in fake_ljm.stream_starts] == [500, 250, 125]
        assert device._scans_per_read == 6
        assert device.stream_rate_hz == 125

    def test_gives_up_below_10_hz(self, fake_ljm):
        fake_ljm.max_hz = 5
        device = LabJackT7(_pt_and_tc(), [], stream_hz=500)
        device._handle = _FakeLJM.HANDLE

        with pytest.raises(RuntimeError, match="any rate >= 10 Hz"):
            device.start_stream()
        assert [s[3] for s in fake_ljm.stream_starts] == [500, 250, 125, 62, 31, 15]
        assert not device.is_streaming

    def test_other_ljm_errors_are_not_retried(self, fake_ljm):
        fake_ljm.start_error = interface.ljm.LJMError(
            errorCode=1227, errorString="LJME_DEVICE_NOT_FOUND")
        device = LabJackT7(_pt_and_tc(), [])
        device._handle = _FakeLJM.HANDLE

        with pytest.raises(interface.ljm.LJMError):
            device.start_stream()
        assert len(fake_ljm.stream_starts) == 1
        assert not device.is_streaming

    def test_refuses_with_no_wired_channel(self, fake_ljm):
        device = LabJackT7([], [])
        device._handle = _FakeLJM.HANDLE
        with pytest.raises(RuntimeError, match="No wired analog channels"):
            device.start_stream()
        assert fake_ljm.stream_starts == []


class TestStreamRead:

    def test_read_before_start_raises(self, fake_ljm):
        device = LabJackT7(_pt_and_tc(), [])
        device._handle = _FakeLJM.HANDLE
        with pytest.raises(RuntimeError, match="before start_stream"):
            device.stream_read()

    def test_batch_is_deinterleaved_by_scan_list_order(self, fake_ljm):
        device = LabJackT7(_pt_and_tc(), [])
        device._handle = _FakeLJM.HANDLE
        device.start_stream()
        fake_ljm.reads.append(([1.0, 0.01, 2.0, 0.02, 3.0, 0.03], 4, 10))

        batch = device.stream_read()

        assert batch["pt0"] == [1.0, 2.0, 3.0]
        assert batch["tc0"] == [0.01, 0.02, 0.03]
        assert batch["backlog_device"] == [4]
        assert batch["backlog_ljm"] == [10]

    def test_scan_times_are_spaced_at_the_actual_rate_ending_now(self, fake_ljm):
        fake_ljm.max_hz = 300
        device = LabJackT7(_pt_and_tc(), [], stream_hz=500)
        device._handle = _FakeLJM.HANDLE
        device.start_stream()                       # negotiates 250 Hz
        fake_ljm.reads.append(([0.0] * 8, 0, 0))   # 4 scans x 2 channels

        before = device.stream_time_s
        times = device.stream_read()["scan_times"]
        after = device.stream_time_s

        assert len(times) == 4
        for a, b in zip(times, times[1:]):
            assert b - a == pytest.approx(1 / 250)
        assert before <= times[-1] <= after


class TestLifecycle:

    def _device(self):
        return LabJackT7(load_channels(REPO_CHANNELS), load_actuators(REPO_ACTUATORS))

    def test_open_safes_every_valve_before_arming_the_watchdog(self, fake_ljm):
        device = self._device()
        device.open()

        assert device.is_connected
        assert device.serial_number == str(_FakeLJM.SERIAL)
        valves = [s.dio for s in load_actuators(REPO_ACTUATORS)
                  if s.type == BINARY_DIO and s.dio is not None]
        assert valves and all(fake_ljm.registers[dio] == 1 for dio in valves)

        writes = [name for name, _ in fake_ljm.writes()]
        last_safe = max(i for i, name in enumerate(writes) if name in valves)
        assert last_safe < writes.index("WATCHDOG_ENABLE_DEFAULT")
        assert fake_ljm.registers["WATCHDOG_ENABLE_DEFAULT"] == 1

    def test_close_stops_the_stream_and_safes_before_closing(self, fake_ljm):
        device = self._device()
        device.open()
        device.start_stream()
        device.write_actuator("fuel_vent", 1)
        assert fake_ljm.registers["EIO7"] == 0          # active-low: energised
        fake_ljm.events.clear()

        device.close()

        kinds = [e[0] for e in fake_ljm.events]
        assert kinds[0] == "stop" and kinds[-1] == "close"
        assert ("write", "EIO7", 1) in fake_ljm.events
        assert not device.is_connected and not device.is_streaming
        assert device._handle is None

    def test_close_survives_an_ljm_error(self, fake_ljm):
        fake_ljm.close_error = interface.ljm.LJMError(
            errorCode=1239, errorString="LJME_RECONNECT_FAILED")
        device = self._device()
        device.open()

        device.close()

        assert not device.is_connected and device._handle is None
