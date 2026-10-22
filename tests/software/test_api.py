"""tests/software/test_api.py: HTTP routes via TestClient on a mock Engine."""

import tempfile
import threading
import time
import types

import pytest
import yaml
from fastapi.testclient import TestClient

import daq.api as api_module
from daq.api import app
from daq.logger import Logger

from tests.software._helpers import REPO_ACTUATORS, make_engine, wait_for_first_batch


def _wait_until_idle(api_client, timeout=30.0):
    """
    Block until the engine has no active sequence and no live sequence thread
    (an abort cycle takes ~10 s). Asks the engine, not the lagging snapshot.
    """
    engine = api_module._engine
    deadline = time.time() + timeout
    while time.time() < deadline:
        thread = engine._sequence_thread
        if not engine._sequence_active and not (thread and thread.is_alive()):
            return
        time.sleep(0.1)
    raise AssertionError(f"engine still busy after {timeout} s")


def _ensure_idle(api_client, timeout=60.0):
    """Halt whatever an earlier test left running (/deenergize), then wait."""
    api_client.post("/deenergize")
    _wait_until_idle(api_client, timeout=timeout)


def _start(api_client, path):
    """POST /fire or /sequence/start; it must answer 200 and really start."""
    r = api_client.post(path)
    assert r.status_code == 200, r.text
    assert api_module._engine._sequence_active, f"POST {path}: 200 but no start"
    return r.json()


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
        assert "backlog" in data

    def test_status_reports_unwired_actuators(self):
        """The shipped manifest: both mains have no step counts yet."""
        prev_engine, prev_logger = api_module._engine, api_module._logger
        api_module.set_engine(make_engine(actuators_path=REPO_ACTUATORS))
        try:
            data = TestClient(app).get("/status").json()
        finally:
            api_module.set_engine(prev_engine, prev_logger)
        assert sorted(data["unwired_actuators"]) == ["fuel_main", "lox_main"]

    def test_snapshot_channels_are_manifest_keyed(self, api_client):
        r = api_client.get("/snapshot")
        assert r.status_code == 200
        channels = r.json()["channels"]
        assert "pt0" in channels
        assert channels["pt0"]["value"] is not None
        assert channels["pt0"]["unit"] == "Pa"

    def test_snapshot_reports_inactive_channels_as_null(self, api_client):
        """Inactive channels stay in the set, with no value."""
        channels = api_client.get("/snapshot").json()["channels"]
        assert "pt7" in channels
        assert channels["pt7"]["value"] is None

    def test_snapshot_carries_derived_values(self, api_client):
        derived = api_client.get("/snapshot").json()["derived"]
        assert set(derived) == {
            "lox_mass_flow_rate_kg_s", "fuel_mass_flow_rate_kg_s", "mixture_ratio",
            "total_impulse_lbf_s", "lox_boiling_advisory",
        }

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
        for key in ("recording", "rows_written", "rows_dropped", "error"):
            assert key in data

    def test_actuator_command(self, api_client):
        r = api_client.post("/actuator", json={"name": "fuel_vent", "state": 1})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        api_client.post("/actuator", json={"name": "fuel_vent", "state": 0})

    def test_actuator_invalid_state(self, api_client):
        r = api_client.post("/actuator", json={"name": "fuel_vent", "state": 99})
        assert r.status_code == 422

    def test_actuator_unknown_name(self, api_client):
        r = api_client.post("/actuator", json={"name": "mystery_valve", "state": 1})
        assert r.status_code == 422

    def test_status_reports_null_name_when_idle(self, api_client):
        _ensure_idle(api_client)
        time.sleep(0.2)   # a batch after going idle
        assert api_client.get("/status").json()["sequence_name"] is None

    def test_deenergize_endpoint(self, api_client):
        r = api_client.post("/deenergize")
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

    def test_calibration_unknown_tag_422(self, api_client):
        r = api_client.post("/calibration", json={
            "tag": "pt0_typo", "slope": 500.0, "intercept": 0.0
        })
        assert r.status_code == 422

    def test_log_start_and_stop(self, api_client):
        r_start = api_client.post("/log/start", json={"prefix": "test_log"})
        assert r_start.status_code == 200
        assert r_start.json()["ok"] is True
        time.sleep(0.1)
        r_stop = api_client.post("/log/stop")
        assert r_stop.status_code == 200

    def test_log_start_bad_prefix_422(self, api_client):
        r = api_client.post("/log/start", json={"prefix": "../escape"})
        assert r.status_code == 422

    def test_log_start_conflict(self, api_client):
        api_client.post("/log/start", json={"prefix": "conflict_a"})
        r = api_client.post("/log/start", json={"prefix": "conflict_b"})
        # Should 409 because already recording
        assert r.status_code == 409
        api_client.post("/log/stop")

    def test_fire_then_abort(self, api_client):
        _ensure_idle(api_client)
        r_fire = api_client.post("/fire")
        assert r_fire.status_code == 200, (
            f"Fire returned {r_fire.status_code}: {r_fire.json()}"
        )
        time.sleep(0.1)
        r_abort = api_client.post("/abort")
        assert r_abort.status_code == 200

    def test_fire_conflict(self, api_client):
        _ensure_idle(api_client)
        _start(api_client, "/fire")
        r = api_client.post("/fire")
        assert r.status_code == 409
        api_client.post("/sequence/stop")

    def test_sequence_get_returns_named_steps(self, api_client):
        r = api_client.get("/sequence")
        assert r.status_code == 200
        names = [s["name"] for s in r.json()["steps"]]
        assert len(names) == 10
        assert "purge_open" in names
        assert "purge_final_close" in names

    def test_sequence_time_backward_rejected_while_running(self, api_client):
        _ensure_idle(api_client)
        _start(api_client, "/fire")
        try:
            time.sleep(0.05)  # past T=0, so seconds=0 is a backward seek
            r = api_client.post("/sequence/time", json={"seconds": 0})
            assert r.status_code == 409
        finally:
            api_client.post("/sequence/stop")   # fast: no abort.yaml tail to drain

    def test_sequence_time_forward_while_running_replays_and_keeps_ticking(self, api_client):
        _ensure_idle(api_client)
        _start(api_client, "/fire")
        try:
            r = api_client.post("/sequence/time", json={"seconds": 5.0})
            assert r.status_code == 200
            assert r.json()["sequence_time_ms"] >= 5000

            # The jump fires main_valves_open (T+5.0s) immediately.
            # /actuators updates on the next batch.
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
        _start(api_client, "/fire")
        assert api_client.post("/sequence/stop").status_code == 200   # pause

        body = _start(api_client, "/sequence/start")                  # resume
        assert set(body) == {"step", "state", "server_time_ms", "sequence_time_ms"}

        r_stop = api_client.post("/sequence/stop")
        assert r_stop.status_code == 200

        # /status lags by up to one batch.
        time.sleep(0.3)
        status = api_client.get("/status").json()
        assert status["sequence_active"] is False

    def test_sequence_start_conflicts_with_active_fire(self, api_client):
        _ensure_idle(api_client)
        _start(api_client, "/fire")
        r = api_client.post("/sequence/start")
        assert r.status_code == 409
        api_client.post("/sequence/stop")   # fast: no abort.yaml tail to drain

    def test_sequence_start_refused_after_abort_until_fresh_fire(self, api_client):
        _ensure_idle(api_client)
        _start(api_client, "/fire")
        time.sleep(0.1)
        api_client.post("/abort")
        _ensure_idle(api_client)      # cut abort.yaml's tail short

        r = api_client.post("/sequence/start")
        assert r.status_code == 409

        _start(api_client, "/fire")   # fresh fire clears the invalidation
        api_client.post("/sequence/stop")

    def test_sequence_start_mismatch_409_lists_differences(self, tmp_path):
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir()
        (seq_dir / "fire.yaml").write_text(yaml.safe_dump({
            "post_record_seconds": 0,
            "steps": [
                {"name": "purge", "actions": [[0.0, "lox_purge", 1]]},
                {"name": "later", "actions": [[100.0, "fuel_purge", 1]]},
            ],
        }))
        engine = make_engine(sequence_dir=str(seq_dir))
        prev_engine, prev_logger = api_module._engine, api_module._logger
        api_module.set_engine(engine)
        try:
            client = TestClient(app)
            engine.fire()
            time.sleep(0.1)
            engine.stop_sequence()
            engine.write_actuator("lox_purge", 0)

            r = client.post("/sequence/start")
            assert r.status_code == 409
            assert r.json()["differences"] == [
                {"actuator": "lox_purge", "expected": 1, "actual": 0}
            ]
            assert client.post(
                "/sequence/start", json={"on_mismatch": "maybe"}
            ).status_code == 422

            r = client.post("/sequence/start", json={"on_mismatch": "converge"})
            assert r.status_code == 200
            engine.stop_sequence()
        finally:
            api_module.set_engine(prev_engine, prev_logger)

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
        """Own engine, so breaking it can't affect other tests; api.py's
        module-level engine is saved and restored."""
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


# -- Foreground Write Mutex ------------------------

class _BlockingStubEngine:
    """Engine double that holds _control_lock until released (the mock is too fast)."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def set_state_listener(self, listener):
        pass

    def _block(self, name, result=None):
        self.calls.append(name)
        self.entered.set()
        self.release.wait(timeout=5.0)
        return result

    def write_actuator(self, name, state):
        self._block("write_actuator")

    def fire(self):
        self._block("fire")

    def abort(self):
        self.calls.append("abort")

    def deenergize(self):
        self.calls.append("deenergize")

    def start_sequence(self):
        return self._block(
            "start_sequence",
            {"step": None, "server_time_ms": 0, "sequence_time_ms": 0},
        )

    @property
    def snapshot(self):
        return types.SimpleNamespace(sequence_active=False, sequence_name="")


@pytest.fixture
def stub_client():
    """Swap in _BlockingStubEngine, restoring the real engine afterwards."""
    prev_engine, prev_logger = api_module._engine, api_module._logger
    stub = _BlockingStubEngine()
    api_module.set_engine(stub)
    try:
        yield TestClient(app), stub
    finally:
        api_module.set_engine(prev_engine, prev_logger)


class TestControlLockSynchronization:
    """Busy lock: /actuator, /fire, /sequence/* get 423; /abort, /deenergize skip it."""

    def test_actuator_busy_returns_423(self, stub_client):
        client, stub = stub_client
        result = {}

        def call_first():
            result["r1"] = client.post(
                "/actuator", json={"name": "lox_main", "state": 1}
            )

        t1 = threading.Thread(target=call_first)
        t1.start()
        assert stub.entered.wait(timeout=2.0), "first call never entered the guard"

        r2 = client.post("/actuator", json={"name": "lox_main", "state": 1})
        assert r2.status_code == 423
        assert r2.json() == {"detail": "busy"}

        stub.release.set()
        t1.join(timeout=2.0)
        assert result["r1"].status_code == 200

    def test_actuator_busy_blocks_sequence_start_with_423(self, stub_client):
        """One lock shared across the whole discard group, not per-route:
        an /actuator command in flight also discards /sequence/start."""
        client, stub = stub_client

        t1 = threading.Thread(
            target=lambda: client.post(
                "/actuator", json={"name": "lox_main", "state": 1}
            )
        )
        t1.start()
        assert stub.entered.wait(timeout=2.0)

        r2 = client.post("/sequence/start")
        assert r2.status_code == 423
        assert "start_sequence" not in stub.calls

        stub.release.set()
        t1.join(timeout=2.0)

    def test_abort_and_deenergize_skip_a_busy_lock(self, stub_client):
        client, stub = stub_client
        t1 = threading.Thread(
            target=lambda: client.post(
                "/actuator", json={"name": "lox_main", "state": 1}
            )
        )
        t1.start()
        assert stub.entered.wait(timeout=2.0)
        try:
            for path, call in (("/abort", "abort"), ("/deenergize", "deenergize")):
                t0 = time.perf_counter()
                r = client.post(path)
                assert r.status_code == 200
                assert time.perf_counter() - t0 < 0.5, f"{path} waited for the lock"
                assert stub.calls[-1] == call
        finally:
            stub.release.set()
            t1.join(timeout=2.0)
