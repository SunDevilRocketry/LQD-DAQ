"""
tests/software/test_api.py

HTTP endpoint tests for daq/api.py via FastAPI's TestClient, running
against an Engine wired to mock hardware.

No real hardware or network required.
Run with: python -m pytest tests/software/test_api.py -v
"""

import tempfile
import time

import pytest
from fastapi.testclient import TestClient

import daq.api as api_module
from daq.api import app
from daq.logger import Logger

from tests.software._helpers import make_engine


@pytest.fixture(scope="module")
def api_client():
    """Stand up engine + API for the full module test run."""
    engine = make_engine()
    logger = Logger(output_dir=tempfile.mkdtemp())
    logger.open()
    engine.start()
    api_module.set_engine(engine, logger)
    time.sleep(0.3)

    client = TestClient(app)
    yield client

    engine.stop()
    logger.close()


class TestAPI:

    def test_status_ok(self, api_client):
        r = api_client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "stream_hz" in data

    def test_snapshot_has_pc(self, api_client):
        r = api_client.get("/snapshot")
        assert r.status_code == 200
        data = r.json()
        assert "PC" in data
        assert data["PC"] is not None

    def test_snapshot_has_derived_fields(self, api_client):
        r = api_client.get("/snapshot")
        data = r.json()
        assert "lox_mdot"      in data
        assert "mixture_ratio" in data
        assert "impulse_ns"    in data

    def test_actuators_endpoint(self, api_client):
        r = api_client.get("/actuators")
        assert r.status_code == 200
        data = r.json()
        assert "LOx Main" in data
        assert data["LOx Main"] in (0, 1)

    def test_events_endpoint(self, api_client):
        r = api_client.get("/events")
        assert r.status_code == 200
        data = r.json()
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_logger_status_endpoint(self, api_client):
        r = api_client.get("/logger")
        assert r.status_code == 200
        data = r.json()
        assert "recording" in data

    def test_actuator_command(self, api_client):
        r = api_client.post("/actuator", json={"name": "Fuel Vent", "state": 1})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        # Clean up
        api_client.post("/actuator", json={"name": "Fuel Vent", "state": 0})

    def test_actuator_invalid_state(self, api_client):
        r = api_client.post("/actuator", json={"name": "Fuel Vent", "state": 99})
        assert r.status_code == 422

    def test_actuator_unknown_name(self, api_client):
        r = api_client.post("/actuator", json={"name": "Mystery Valve", "state": 1})
        assert r.status_code == 422

    def test_safe_endpoint(self, api_client):
        r = api_client.post("/safe")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_tare_endpoint(self, api_client):
        r = api_client.post("/tare")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_reset_impulse_endpoint(self, api_client):
        r = api_client.post("/reset_impulse")
        assert r.status_code == 200

    def test_abort_endpoint(self, api_client):
        r = api_client.post("/abort")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_calibration_update_endpoint(self, api_client):
        r = api_client.post("/calibration", json={
            "tag": "PC", "slope": 128.0, "intercept": -62.8
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_log_start_and_stop(self, api_client):
        r_start = api_client.post("/log/start", json={"prefix": "test_log"})
        assert r_start.status_code == 200
        assert r_start.json()["ok"] is True
        time.sleep(0.1)
        r_stop = api_client.post("/log/stop")
        assert r_stop.status_code == 200

    def test_log_start_conflict(self, api_client):
        api_client.post("/log/start", json={"prefix": "conflict_a"})
        r = api_client.post("/log/start", json={"prefix": "conflict_b"})
        # Should 409 because already recording
        assert r.status_code == 409
        api_client.post("/log/stop")

    def test_fire_then_abort(self, api_client):
        # Ensure any prior sequence has finished before firing
        for _ in range(20):
            snap = api_client.get("/snapshot").json()
            if not snap.get("sequence_active"):
                break
            time.sleep(0.3)
        r_fire = api_client.post("/fire")
        assert r_fire.status_code == 200, (
            f"Fire returned {r_fire.status_code}: {r_fire.json()}"
        )
        time.sleep(0.1)
        r_abort = api_client.post("/abort")
        assert r_abort.status_code == 200

    def test_fire_conflict(self, api_client):
        api_client.post("/abort")   # ensure clean state
        time.sleep(0.5)
        # Start fire
        api_client.post("/fire")
        time.sleep(0.05)
        # Fire again while running should return 409
        snap_r = api_client.get("/snapshot")
        if snap_r.json().get("sequence_active"):
            r = api_client.post("/fire")
            assert r.status_code == 409
        api_client.post("/abort")

    def test_snapshot_does_not_error_under_concurrent_polls(self, api_client):
        """10 rapid-fire polls should all return 200."""
        results = []
        for _ in range(10):
            r = api_client.get("/snapshot")
            results.append(r.status_code)
        assert all(s == 200 for s in results), f"Some polls failed: {results}"


# -- Data Freshness --------------------------

class TestDataFreshnessAPI:
    """API-level stale-data behavior: soft signal on /status, hard 500 on /snapshot."""

    def test_status_has_freshness_fields_when_fresh(self, api_client):
        r = api_client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert "stale" in data
        assert "data_age_s" in data
        assert data["stale"] is False

    def test_snapshot_has_data_age_when_fresh(self, api_client):
        r = api_client.get("/snapshot")
        assert r.status_code == 200
        assert "data_age_s" in r.json()

    def test_snapshot_returns_500_when_stale(self):
        """Uses its own engine/client (not the shared api_client fixture)
        so breaking the device doesn't affect other API tests. api.py's
        _engine/_logger are module-level singletons shared with whatever
        else is using `app` (e.g. the api_client fixture), so we must
        save and restore them - otherwise this test leaves a dead engine
        wired into the global app for every test that runs after it."""
        prev_engine, prev_logger = api_module._engine, api_module._logger

        engine = make_engine()
        logger = Logger(output_dir=tempfile.mkdtemp())
        logger.open()
        engine.start()
        api_module.set_engine(engine, logger)
        time.sleep(0.3)

        try:
            device = engine._device

            def always_fail():
                raise RuntimeError("simulated comms loss")

            device.stream_read = always_fail
            time.sleep(1.3)   # > _DATA_STALE_THRESHOLD_S (1.0s)

            client = TestClient(app)

            r_snap = client.get("/snapshot")
            assert r_snap.status_code == 500

            # /status must still succeed and report the staleness, never error.
            r_status = client.get("/status")
            assert r_status.status_code == 200
            assert r_status.json()["stale"] is True
        finally:
            engine.stop()
            logger.close()
            api_module.set_engine(prev_engine, prev_logger)
