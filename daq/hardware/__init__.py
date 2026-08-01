"""
daq/hardware/__init__.py
 
Dynamic hardware driver selector.

Attempts to import the LabJack LJM library to expose the physical LabJackT7 driver.
Falls back to MockLabJack transparently if LJM is unavailable.

Usage:
    from daq.hardware import Device, USING_MOCK
"""
 
try:
    from labjack import ljm as _ljm                         # Probe LJM driver availability

    if getattr(_ljm, "_staticLib", None) is None:
        raise ImportError("LJM native library not found")

    from daq.hardware.interface import LabJackT7 as Device
    USING_MOCK = False
except ImportError:
    from daq.hardware.mock import MockLabJack as Device     # type: ignore[assignment]
    USING_MOCK = True
 
__all__ = ["Device", "USING_MOCK"]
 
