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
    ActuatorSpec, ChannelSpec, PT_DIRECT, PULSE_STEPPER, load_actuators, load_channels,
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
