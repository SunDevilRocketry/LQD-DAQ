"""
tests/software/ - Software-only test suite (mock hardware, no LabJack required).

Every test under this package runs against MockLabJack regardless of what
hardware is physically attached to the machine running pytest.

For tests that require a real, physically-connected LabJack T7, see
tests/hardware_live/ instead.
"""
