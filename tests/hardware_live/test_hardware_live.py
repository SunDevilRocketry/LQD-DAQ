"""
tests/hardware_live/test_hardware_live.py

Hardware-in-the-loop (HIL) sanity checks against a real, physically
connected LabJack T7. Requires:
  - The labjack-ljm native driver installed on this machine.
  - A LabJack T7 plugged in over USB (or reachable over Ethernet/WiFi).

These are are not run by `run.sh --test` / `run.bat --test` by default.

Run this file directly, on the bench, with the hardware attached:

    python -m pytest tests/hardware_live/ -v

If no physical device is present, every test here will fail fast with a
device/connection error.

The driver is manifest-driven, so these tests run against the repo's real
channels.yaml / actuators.yaml. Any test that needs a wired pin skips with
a clear reason while that pin is still null - the point of running this on
the bench is to confirm the wiring you just did, so it should tell you
what isn't wired rather than fail opaquely.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import pytest

from daq.engine import DEFAULT_ACTUATORS_PATH, DEFAULT_CHANNELS_PATH
from daq.hardware.interface import LabJackT7
from daq.manifest import BINARY_DIO, load_actuators, load_channels


@pytest.fixture(scope="module")
def channels():
    return load_channels(DEFAULT_CHANNELS_PATH)


@pytest.fixture(scope="module")
def actuators():
    return load_actuators(DEFAULT_ACTUATORS_PATH)


@pytest.fixture
def device(channels, actuators):
    dev = LabJackT7(channels, actuators)
    dev.open()
    try:
        yield dev
    finally:
        dev.close()


class TestPhysicalConnection:

    def test_physical_connection(self, device):
        """Verify a real T7 is detected and can open a handle."""
        assert device.is_connected is True
        assert device.serial_number, "No serial number returned - is a T7 attached?"
        print(f"Verified T7 Serial: {device.serial_number}")

    def test_physical_stream(self, device):
        """Verify real 500 Hz streaming pulls at least one non-empty batch."""
        tags = device.sensor_tags
        if not tags:
            pytest.skip(
                "No channel in channels.yaml has an AIN assigned yet - "
                "nothing to stream"
            )

        device.start_stream()
        try:
            batch = device.stream_read()
            for tag in tags:
                assert tag in batch, f"{tag} missing from batch"
                assert len(batch[tag]) > 0, f"{tag} batch is empty"
        finally:
            device.stop_stream()

    def test_physical_actuator_round_trip(self, device, actuators):
        """
        Write then read back a single solenoid on real hardware, then
        immediately return it to the safe (closed) state.

        Deliberately picks a binary_dio actuator: a stepper round-trip
        would physically move a main valve, which is not something a
        connectivity smoke test should do on its own.
        """
        wired = [
            spec for spec in actuators
            if spec.type == BINARY_DIO and spec.is_wired
        ]
        if not wired:
            pytest.skip(
                "No binary_dio actuator in actuators.yaml has a DIO "
                "assigned yet"
            )

        name = wired[0].id
        try:
            device.write_actuator(name, 1)
            assert device.read_actuator(name) == 1
        finally:
            device.write_actuator(name, 0)
