"""tests/software/test_main.py: startup guards in daq/__main__.py."""

import pytest

import daq.hardware as hw
from daq.__main__ import main


def test_refuses_to_run_on_mock_unless_asked(monkeypatch):
    """LJM missing + no --mock must stop, not silently stream fake data."""
    monkeypatch.setattr(hw, "USING_MOCK", True)
    monkeypatch.setattr("sys.argv", ["daq"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
