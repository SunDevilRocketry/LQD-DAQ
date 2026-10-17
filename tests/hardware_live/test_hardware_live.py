"""
tests/hardware_live/test_hardware_live.py

Bench checks against a real T7 with LJM, using the repo's manifests. Not in
CI; run with `Start_Linux.sh --test-live` or
`python -m pytest tests/hardware_live/ -v`. Tests that need an unwired pin
skip and say which.
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
        Energise then de-energise the binary_dio actuator named in
        LQD_LIVE_ACTUATOR; skipped if unset. Never name one with gas connected.
        """
        name = os.environ.get("LQD_LIVE_ACTUATOR")
        if not name:
            pytest.skip("Set LQD_LIVE_ACTUATOR=<id> to energise that actuator")
        spec = next((s for s in actuators if s.id == name), None)
        if spec is None or spec.type != BINARY_DIO or not spec.is_wired:
            pytest.fail(f"'{name}' is not a wired binary_dio actuator")

        try:
            device.write_actuator(name, 1)
            assert device.read_actuator(name) == 1
        finally:
            device.write_actuator(name, 0)
