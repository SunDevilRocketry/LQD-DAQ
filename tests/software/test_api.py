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

from tests.software._helpers import make_engine, wait_for_first_batch


def _wait_until_idle(api_client, timeout=20.0):
    """
    Blocks until no sequence is active. abort.yaml's last action lands at
    T+5s and its own post_record_seconds adds another 5s on top, so a full
    abort cycle takes ~10s to clear sequence_active - mirrors
    test_fire_then_abort's retry loop.

    abort() can restart abort.yaml right as a previous run finishes. To 
    avoid catching the gap between cycles, require two consecutive 
    inactive checks b/f declaring idle.
    """
    deadline = time.time() + timeout
    consecutive_idle = 0
    while time.time() < deadline:
        if not api_client.get("/snapshot").json().get("sequence_active"):
            consecutive_idle += 1
            if consecutive_idle >= 2:
                return
        else:
            consecutive_idle = 0
        time.sleep(0.3)


def _ensure_idle(api_client, timeout=60.0):
    """
    Waits for a clean, settled idle state before a test that needs one.

    Don't call POST /abort here. Earlier tests already triggered aborts, 
    and calling it again just adds another ~10s cycle. Just wait for the
    existing run to drain using _wait_until_idle.
    """
    _wait_until_idle(api_client, timeout=timeout)


def _start_reliably(api_client, path, attempts=40, poll_interval=0.5):
    """
    POST to a sequence-starting route (/fire or /sequence/start), retrying
    if it silently no-oped, and returning the call's JSON body once the
    engine confirms it actually took hold.

    Both fire() and start_sequence() check if the previous thread is still alive.
    Right after a thread finishes, that check can briefly read stale, causing the 
    call to log and return 200 without actually running. Retry briefly to handle
    the race.
    """
    for _ in range(attempts):
        body = api_client.post(path).json()
        time.sleep(poll_interval)
        if api_client.get("/status").json().get("sequence_active"):
            return body
    raise AssertionError(f"POST {path} never actually started after retries")


@pytest.fixture(scope="module")
def api_client():
    """Stand up engine + API for the full module test run."""
    engine = make_engine()
    logger = Logger(output_dir=tempfile.mkdtemp(), channels=engine.channel_specs)
    logger.open()
    engine.start()
    api_module.set_engine(engine, logger)
    wait_for_first_batch(engine)

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

    def test_status_reports_unwired_actuators(self, api_client):
        """An operator's only pre-test view of actuators that cannot be
        driven."""
        data = api_client.get("/status").json()
        assert data["unwired_actuators"] == []

    def test_snapshot_channels_are_manifest_keyed(self, api_client):
        r = api_client.get("/snapshot")
        assert r.status_code == 200
        channels = r.json()["channels"]
        assert "pt0" in channels
        assert channels["pt0"]["value"] is not None
        assert channels["pt0"]["unit"] == "Pa"

    def test_snapshot_reports_inactive_channels_as_null(self, api_client):
        """An inactive channel stays in the reported set so a consumer's
        key set matches the manifest - it just carries no value."""
        channels = api_client.get("/snapshot").json()["channels"]
        assert "pt7" in channels
        assert channels["pt7"]["value"] is None

    def test_snapshot_has_no_derived_channels(self, api_client):
        """Derived channels were deleted, not generalised (handoff sec 3/5)."""
        data = api_client.get("/snapshot").json()
        for gone in ("lox_mdot", "fuel_mdot", "mixture_ratio", "impulse_ns"):
            assert gone not in data

    def test_actuators_endpoint(self, api_client):
        r = api_client.get("/actuators")
        assert r.status_code == 200
        data = r.json()
        assert "lox_main" in data
        assert data["lox_main"]["state"] in (0, 1)
        assert isinstance(data["lox_main"]["moving"], bool)

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
        r = api_client.post("/actuator", json={"name": "fuel_vent", "state": 1})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        # Clean up
        api_client.post("/actuator", json={"name": "fuel_vent", "state": 0})

    def test_actuator_invalid_state(self, api_client):
        r = api_client.post("/actuator", json={"name": "fuel_vent", "state": 99})
        assert r.status_code == 422

    def test_actuator_unknown_name(self, api_client):
        r = api_client.post("/actuator", json={"name": "mystery_valve", "state": 1})
        assert r.status_code == 422

    def test_safe_endpoint(self, api_client):
        r = api_client.post("/safe")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_tare_endpoint(self, api_client):
        r = api_client.post("/tare")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_abort_endpoint(self, api_client):
        r = api_client.post("/abort")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_calibration_update_endpoint(self, api_client):
        r = api_client.post("/calibration", json={
            "tag": "pt0", "slope": 500.0, "intercept": 0.0
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

    def test_sequence_get_returns_named_steps(self, api_client):
        r = api_client.get("/sequence")
        assert r.status_code == 200
        names = [s["name"] for s in r.json()["steps"]]
        assert len(names) == 9
        assert "purge_open" in names
        assert "purge_final_close" in names

    def test_sequence_time_backward_rejected_while_running(self, api_client):
        _ensure_idle(api_client)
        _start_reliably(api_client, "/fire")
        try:
            time.sleep(0.05)  # clearly past T=0, so seconds=0 below is unambiguously backward
            r = api_client.post("/sequence/time", json={"seconds": 0})
            assert r.status_code == 409
        finally:
            api_client.post("/sequence/stop")   # fast: no abort.yaml tail to drain

    def test_sequence_time_forward_while_running_replays_and_keeps_ticking(self, api_client):
        _ensure_idle(api_client)
        _start_reliably(api_client, "/fire")
        try:
            r = api_client.post("/sequence/time", json={"seconds": 5.0})
            assert r.status_code == 200
            assert r.json()["sequence_time_ms"] >= 5000

            # The jump fires main_valves_open (T+5.0s) immediately.
            # Wait for the acquisition loop to publish the updated writes to /actuators.
            time.sleep(0.3)
            actuators = api_client.get("/actuators").json()
            assert actuators["lox_main"]["state"] == 1

            # Ticking resumed live from the new position, not left stopped.
            status = api_client.get("/status").json()
            assert status["sequence_active"] is True
        finally:
            api_client.post("/sequence/stop")

    def test_sequence_step_unknown_name_422(self, api_client):
        # An unknown name is rejected before any running-state check, so
        # this doesn't depend on idle state (see jump_to_step()).
        r = api_client.post("/sequence/step", json={"name": "not_a_real_step"})
        assert r.status_code == 422

    def test_sequence_start_stop_roundtrip(self, api_client):
        _ensure_idle(api_client)

        body = _start_reliably(api_client, "/sequence/start")
        assert set(body) == {"step", "server_time_ms", "sequence_time_ms"}

        r_stop = api_client.post("/sequence/stop")
        assert r_stop.status_code == 200

        # /status reflects the published snapshot, refreshed by the
        # acquisition loop on its own cadence - give it a moment to catch
        # up with stop_sequence()'s already-applied internal state.
        time.sleep(0.3)
        status = api_client.get("/status").json()
        assert status["sequence_active"] is False

    def test_sequence_start_conflicts_with_active_fire(self, api_client):
        _ensure_idle(api_client)
        _start_reliably(api_client, "/fire")
        r = api_client.post("/sequence/start")
        assert r.status_code == 409
        api_client.post("/sequence/stop")   # fast: no abort.yaml tail to drain

    def test_sequence_start_refused_after_abort_until_fresh_fire(self, api_client):
        _ensure_idle(api_client)
        _start_reliably(api_client, "/fire")
        time.sleep(0.1)
        api_client.post("/abort")
        _wait_until_idle(api_client)  # abort.yaml's own ~10s tail

        r = api_client.post("/sequence/start")
        assert r.status_code == 409

        _start_reliably(api_client, "/fire")   # fresh fire clears the invalidation
        api_client.post("/sequence/stop")

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
        logger = Logger(output_dir=tempfile.mkdtemp(), channels=engine.channel_specs)
        logger.open()
        engine.start()
        api_module.set_engine(engine, logger)
        wait_for_first_batch(engine)

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
