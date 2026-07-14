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
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import pytest

from daq.hardware.interface import LabJackT7


@pytest.fixture
def device():
    dev = LabJackT7()
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
        device.start_stream()
        try:
            batch = device.stream_read()
            assert "POT" in batch
            assert len(batch["POT"]) > 0
        finally:
            device.stop_stream()

    def test_physical_actuator_round_trip(self, device):
        """Write then read back a single actuator on real hardware, then
        immediately return it to the safe (closed) state."""
        try:
            device.write_actuator("LOx Vent", 1)
            assert device.read_actuator("LOx Vent") == 1
        finally:
            device.write_actuator("LOx Vent", 0)
