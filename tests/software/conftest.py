"""
tests/software/conftest.py

Forces every test under tests/software/ to run against MockLabJack, even if
a physical LabJack + LJM driver are present on the machine running pytest.

Better as conftest.py to avoid binding module-level name to every tests. 
Test file pytest collection order does not matter.

Tests that need a real, physically-connected LabJack T7 live in
tests/hardware_live/
"""

import os
import sys

# Make the `daq` package importable when running `pytest tests/` from the
# repo root w/o an editable install.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import daq.hardware as _hw
from daq.hardware.mock import MockLabJack

_hw.Device = MockLabJack
_hw.USING_MOCK = True
