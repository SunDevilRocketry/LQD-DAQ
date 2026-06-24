"""
daq/hardware/__init__.py
 
Selects the hardware driver at import time based on LJM availability.
 
If the LabJack LJM library is installed and importable, the real T7
driver (LabJackT7) is exported. Otherwise the mock driver
is used transparently so the rest of the system never needs to branch.
 
Usage (everywhere in the codebase):
    from daq.hardware import Device, USING_MOCK
 
    device = Device()
    device.open()
 
The exported name is always "Device" regardless of which backend
is active. USING_MOCK is True when running against simulated hardware.
"""
 
try:
    from labjack import ljm as _ljm  # noqa: F401 - import just to probe availability
    from daq.hardware.interface import LabJackT7 as Device
    USING_MOCK = False
except ImportError:
    from daq.hardware.mock import MockLabJack as Device  # type: ignore[assignment]
    USING_MOCK = True
 
__all__ = ["Device", "USING_MOCK"]
 