"""
tests/software/conftest.py

Forces MockLabJack for every test here, even with a T7 and LJM present.
Runs before any test module imports daq.engine.
"""

import os
import sys

# Importable without an editable install.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import daq.hardware as _hw
from daq.hardware.mock import MockLabJack

_hw.Device = MockLabJack
_hw.USING_MOCK = True
