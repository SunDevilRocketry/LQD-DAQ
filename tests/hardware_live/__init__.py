"""
tests/hardware_live/ - Tests that require a real, physically-connected
LabJack T7 with the LJM driver installed.

These tests will fail with a device-not-found style error on any machine 
w/o the physical hardware attached

Run only on the bench, with the LabJack plugged in:
    python -m pytest tests/hardware_live/ -v
"""
