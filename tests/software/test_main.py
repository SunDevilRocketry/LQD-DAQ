"""tests/software/test_main.py: config loading and startup in daq/__main__.py."""

import os
import types

import pytest
import yaml

import daq.__main__ as main_module
import daq.hardware as hw
from daq import api, calculations
from daq.__main__ import _DEFAULT_CONFIG, _load_config, main

from tests.software._helpers import REPO_CONFIG, wait_for_first_batch


_REPO = os.path.dirname(REPO_CONFIG)


def _write_config(path, cfg):
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_refuses_to_run_on_mock_unless_asked(monkeypatch):
    """LJM missing + no --mock must stop, not silently stream fake data."""
    monkeypatch.setattr(hw, "USING_MOCK", True)
    monkeypatch.setattr("sys.argv", ["daq"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


class TestLoadConfig:

    def test_missing_file_gives_the_defaults(self, tmp_path):
        assert _load_config(str(tmp_path / "nope.yaml")) == _DEFAULT_CONFIG

    def test_sections_merge_key_by_key(self, tmp_path):
        path = _write_config(tmp_path / "config.yaml", {
            "server": {"port": 9000},
            "paths": {"output_dir": "elsewhere"},
        })
        cfg = _load_config(path)
        assert cfg["server"] == {**_DEFAULT_CONFIG["server"], "port": 9000}
        assert cfg["paths"]["output_dir"] == "elsewhere"
        assert cfg["paths"]["lox_table"] == _DEFAULT_CONFIG["paths"]["lox_table"]

    def test_merging_leaves_the_defaults_untouched(self, tmp_path):
        path = _write_config(tmp_path / "config.yaml", {"server": {"port": 9000}})
        _load_config(path)
        assert _DEFAULT_CONFIG["server"]["port"] == 8000

    def test_unknown_and_non_dict_sections_pass_through(self, tmp_path):
        path = _write_config(tmp_path / "config.yaml", {
            "extra": {"a": 1}, "thresholds": None,
        })
        cfg = _load_config(path)
        assert cfg["extra"] == {"a": 1}
        assert cfg["thresholds"] is None

    def test_empty_file_gives_the_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("")
        assert _load_config(str(path)) == _DEFAULT_CONFIG

    def test_shipped_config_keeps_every_default_key(self):
        cfg = _load_config(REPO_CONFIG)
        for section, content in _DEFAULT_CONFIG.items():
            assert set(content) <= set(cfg[section]), section
        assert cfg["derived"]["impulse"]["channels"] == ["lc0", "lc1"]
        assert "ign_sense" in cfg["labels"]


@pytest.fixture
def startup(tmp_path, monkeypatch):
    """main() on the mock with a scratch config; global state restored after."""
    monkeypatch.setattr(hw, "Device", hw.Device)
    monkeypatch.setattr(hw, "USING_MOCK", hw.USING_MOCK)
    monkeypatch.setattr(api, "_engine", None)
    monkeypatch.setattr(api, "_logger", None)
    monkeypatch.setattr(calculations, "_lox_T_rankine", [])
    monkeypatch.setattr(calculations, "_lox_density_lbm_ft3", [])
    monkeypatch.setattr(main_module.signal, "signal", lambda sig, handler: None)

    def run(*flags, **overrides):
        cfg = {
            "server": {"host": "127.0.0.1", "port": 9123},
            "paths": {
                "output_dir":   str(tmp_path / "data"),
                "sequence_dir": os.path.join(_REPO, "sequences"),
                "cal_file":     os.path.join(_REPO, "daq", "calibration.json"),
                "channels":     os.path.join(_REPO, "channels.yaml"),
                "actuators":    os.path.join(_REPO, "actuators.yaml"),
                "lox_table":    os.path.join(_REPO, "daq",
                                             "LOX_table_100_0-260_99R.csv"),
            },
        }
        cfg["paths"].update(overrides)
        path = _write_config(tmp_path / "config.yaml", cfg)
        monkeypatch.setattr("sys.argv", ["daq", "--config", path, *flags])
        main()
        return api._engine, api._logger

    return run


def _headless_for_one_reading(monkeypatch):
    """The headless loop's sleep: first waits for data, then acts as Ctrl+C."""
    calls = []
    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 1:
            wait_for_first_batch(api._engine)
        else:
            raise KeyboardInterrupt
    monkeypatch.setattr(main_module, "time", types.SimpleNamespace(sleep=sleep))


class TestHeadlessStartup:

    def test_runs_streams_and_shuts_down_cleanly(self, startup, monkeypatch, capsys):
        _headless_for_one_reading(monkeypatch)

        engine, logger = startup("--mock", "--no-server")

        out = capsys.readouterr().out
        assert "pt0=" in out and "seq=False" in out
        assert "Clean shutdown complete" in out
        assert not engine._device.is_connected
        assert not logger._running

    def test_loads_the_lox_table(self, startup, monkeypatch):
        _headless_for_one_reading(monkeypatch)
        startup("--mock", "--no-server")
        assert calculations._lox_T_rankine
        assert calculations.lox_density_from_celsius(-183.0) is not None

    def test_missing_lox_table_warns_and_still_runs(self, startup, monkeypatch, capsys):
        _headless_for_one_reading(monkeypatch)
        startup("--mock", "--no-server", lox_table="/nonexistent/lox.csv")
        out = capsys.readouterr().out
        assert "LOX table not found" in out
        assert "Clean shutdown complete" in out
        assert calculations._lox_T_rankine == []

    def test_mock_fills_placeholder_step_counts(self, startup, monkeypatch):
        _headless_for_one_reading(monkeypatch)
        engine, _ = startup("--mock", "--no-server")
        mains = [s for s in engine.actuator_specs if s.id in ("lox_main", "fuel_main")]
        assert len(mains) == 2
        assert all(s.steps_open and s.steps_close and s.pulse_freq_hz for s in mains)


class TestServerStartup:

    def test_uvicorn_gets_the_configured_server(self, startup, monkeypatch):
        calls = []
        monkeypatch.setattr(main_module.uvicorn, "run",
                            lambda app, **kw: calls.append((app, kw)))

        engine, logger = startup("--mock")

        assert calls == [("daq.api:app", {
            "host": "127.0.0.1", "port": 9123, "log_level": "warning",
        })]
        assert not engine._device.is_connected
        assert not logger._running
