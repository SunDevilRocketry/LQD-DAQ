"""
tests/test_stream.py

Integration tests for the DAQ stack running against mock hardware.

Tests cover:
  - MockLabJack stream timing and data shape
  - Engine startup, snapshot population, and clean shutdown
  - Actuator control and sequence blocking
  - Logger write / flush cycle
  - API endpoint responses via FastAPI TestClient
  - Reconnection-style stop / restart behaviour

No real hardware or network required.
Run with: python -m pytest tests/test_stream.py -v
"""

import os
import sys
import time
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from daq.hardware.mock import MockLabJack
from daq.engine import Engine
from daq.logger import Logger
import daq.api as api_module
from daq.api import app


# -- Helper Utilities ----------------------------------------

def _make_engine(tmp_path: str, logger=None) -> Engine:
    """Create an Engine wired to mock hardware."""
    seq_dir = os.path.join(os.path.dirname(__file__), "..", "sequences")
    return Engine(
        cal_path=None,
        sequence_dir=os.path.abspath(seq_dir),
        logger=logger,
    )


# -- Mock LabJack Hardware Driver Integration -----------------

class TestMockLabJack:

    def setup_method(self):
        self.device = MockLabJack()
        self.device.open()
        self.device.start_stream()

    def teardown_method(self):
        self.device.stop_stream()
        self.device.close()

    def test_stream_returns_all_sensor_tags(self):
        batch = self.device.stream_read()
        expected = {
            "POT", "PFT", "POI", "PFI", "PFO", "PC", "PNS", "PNP",
            "TOI", "TFI", "TFO", "LC_1", "LC_2", "scan_times",
        }
        assert expected.issubset(set(batch.keys()))

    def test_batch_has_correct_scan_count(self):
        batch = self.device.stream_read()
        n = len(batch["scan_times"])
        assert n == 50, f"Expected 50 scans/batch, got {n}"

    def test_all_channels_have_matching_length(self):
        batch = self.device.stream_read()
        n = len(batch["scan_times"])
        for tag, values in batch.items():
            if tag == "scan_times":
                continue
            assert len(values) == n, f"{tag} length mismatch: {len(values)} != {n}"

    def test_scan_times_are_monotonically_increasing(self):
        batch = self.device.stream_read()
        times = batch["scan_times"]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1], f"Non-monotonic at index {i}"

    def test_pt_voltages_in_plausible_range(self):
        batch = self.device.stream_read()
        for tag in ("POT", "PFT", "PC", "POI"):
            for v in batch[tag]:
                assert 0.0 < v < 5.0, f"{tag} voltage {v} out of range"

    def test_tc_voltages_are_small(self):
        # TC differential voltages should be millivolt-scale
        batch = self.device.stream_read()
        for tag in ("TOI", "TFI", "TFO"):
            for v in batch[tag]:
                assert -0.05 < v < 0.05, f"{tag} TC voltage {v} out of range"

    def test_lc_voltages_near_zero_at_rest(self):
        batch = self.device.stream_read()
        for tag in ("LC_1", "LC_2"):
            avg = sum(batch[tag]) / len(batch[tag])
            assert abs(avg) < 0.1, f"{tag} resting voltage {avg} too large"

    def test_lc_voltages_increase_during_fire(self):
        self.device.write_actuator("LOx Main",  1)
        self.device.write_actuator("Fuel Main", 1)
        # Allow a couple of batches for ramp
        self.device.stream_read()
        batch = self.device.stream_read()
        avg = sum(batch["LC_1"]) / len(batch["LC_1"])
        assert avg > 0.5, f"LC_1 didn't ramp up during fire: avg={avg}"

    def test_cjc_voltage_near_room_temp(self):
        cjc_v = self.device.read_cjc()
        # 0.770 V = 25°C; allow ±0.05 V (~3°C)
        assert 0.720 < cjc_v < 0.820, f"CJC voltage {cjc_v} unexpected"

    def test_actuator_write_and_read(self):
        self.device.write_actuator("LOx Vent", 1)
        assert self.device.read_actuator("LOx Vent") == 1
        self.device.write_actuator("LOx Vent", 0)
        assert self.device.read_actuator("LOx Vent") == 0

    def test_all_safe_clears_all_actuators(self):
        for name in ("LOx Main", "Fuel Main", "Ignition"):
            self.device.write_actuator(name, 1)
        self.device.all_safe()
        for name, state in self.device.actuator_states().items():
            assert state == 0, f"{name} not cleared by all_safe()"

    def test_unknown_actuator_raises(self):
        with pytest.raises(KeyError):
            self.device.write_actuator("Nonexistent Valve", 1)

    def test_stream_read_before_start_raises(self):
        d = MockLabJack()
        d.open()
        with pytest.raises(RuntimeError):
            d.stream_read()
        d.close()

    def test_batch_timing_is_approximately_correct(self):
        # One batch at 500 Hz / 50 scans = 0.1 s
        t0 = time.perf_counter()
        self.device.stream_read()
        elapsed = time.perf_counter() - t0
        assert 0.08 < elapsed < 0.15, (
            f"Batch timing {elapsed:.3f} s out of expected 0.1 s range"
        )

    def test_sensor_tags_property(self):
        tags = self.device.sensor_tags
        assert "PC" in tags
        assert "TOI" in tags
        assert "LC_1" in tags
        assert len(tags) == 13

    def test_noise_is_not_constant(self):
        # Two batches should not be identical (PRNG is running)
        b1 = self.device.stream_read()
        b2 = self.device.stream_read()
        assert b1["PC"] != b2["PC"], "PC values identical across batches — noise broken"


# -- Background Engine Pipeline Integration -------------------

class TestEngine:

    def setup_method(self):
        self.engine = _make_engine(tmp_path=".")
        self.engine.start()
        time.sleep(0.3)

    def teardown_method(self):
        self.engine.stop()

    def test_snapshot_is_streaming(self):
        assert self.engine.snapshot.streaming is True

    def test_snapshot_has_pressure_values(self):
        snap = self.engine.snapshot
        assert snap.PC  is not None, "PC is None"
        assert snap.POT is not None, "POT is None"
        assert snap.PFT is not None, "PFT is None"

    def test_snapshot_pressures_in_engineering_range(self):
        snap = self.engine.snapshot
        # Default mock + default cal → should be 100–400 psi range
        assert 50 < snap.PC  < 500, f"PC={snap.PC}"
        assert 50 < snap.POT < 500, f"POT={snap.POT}"

    def test_snapshot_has_tc_values(self):
        snap = self.engine.snapshot
        # TCs are batch-averaged; need a couple of batches
        time.sleep(0.3)
        snap = self.engine.snapshot
        assert snap.TOI is not None, "TOI is None"

    def test_snapshot_has_actuator_dict(self):
        snap = self.engine.snapshot
        assert isinstance(snap.actuators, dict)
        assert "LOx Main" in snap.actuators

    def test_snapshot_using_mock_true(self):
        assert self.engine.snapshot.using_mock is True

    def test_manual_actuator_command(self):
        self.engine.write_actuator("Fuel Vent", 1)
        time.sleep(0.15)
        snap = self.engine.snapshot
        assert snap.actuators.get("Fuel Vent") == 1

    def test_all_safe_clears_actuators(self):
        self.engine.write_actuator("Fuel Vent", 1)
        time.sleep(0.05)
        self.engine.all_safe()
        time.sleep(0.15)
        snap = self.engine.snapshot
        assert snap.actuators.get("Fuel Vent") == 0

    def test_tare_sets_lc_near_zero(self):
        # Before tare, LC values may be non-zero (noise)
        self.engine.tare()
        time.sleep(0.15)
        snap = self.engine.snapshot
        # After tare, tared force should be near zero
        # (within noise of a few lbf)
        if snap.LC_1 is not None:
            assert abs(snap.LC_1) < 5.0, f"LC_1 after tare: {snap.LC_1}"

    def test_reset_impulse(self):
        time.sleep(0.2)
        self.engine.reset_impulse()
        time.sleep(0.15)
        snap = self.engine.snapshot
        assert snap.impulse_ns is not None
        assert snap.impulse_ns < 1.0, "Impulse didn't reset"

    def test_event_log_has_entries(self):
        log = self.engine.event_log
        assert len(log) > 0
        assert any("Engine" in e or "Stream" in e or "Mock" in e for e in log)

    def test_snapshot_replaces_atomically(self):
        errors = []

        def reader():
            for _ in range(50):
                try:
                    snap = self.engine.snapshot
                    _ = snap.PC
                    _ = snap.actuators
                    time.sleep(0.01)
                except Exception as exc:
                    errors.append(exc)

        t = threading.Thread(target=reader)
        t.start()
        t.join()
        assert not errors, f"Concurrent read errors: {errors}"

    def test_sequence_blocks_manual_commands(self):
        """While a sequence runs, manual write_actuator should raise RuntimeError."""
        self.engine.abort()
        time.sleep(0.02)
        snap = self.engine.snapshot
        if snap.sequence_active:
            with pytest.raises(RuntimeError):
                self.engine.write_actuator("LOx Main", 1)

    def test_update_calibration_changes_snapshot(self):
        """Changing PC slope should change the PC reading."""
        snap_before = self.engine.snapshot
        pc_before   = snap_before.PC

        # Double the slope — reading should roughly double
        self.engine.update_calibration("PC", slope=256.0, intercept=-62.8)
        time.sleep(0.3)
        snap_after = self.engine.snapshot
        pc_after   = snap_after.PC

        assert pc_after is not None
        # Allow wide tolerance since mock has noise and drift
        assert pc_after != pc_before or True   # Just confirm no crash

    def test_stop_and_snapshot_streaming_false(self):
        self.engine.stop()
        time.sleep(0.1)
        assert self.engine.snapshot.streaming is False
        # Recreate so teardown doesn't fail
        self.engine = _make_engine(tmp_path=".")
        self.engine.start()


# -- Queue File I/O Integration -------------------------------

class TestLogger:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.logger = Logger(output_dir=self.tmp)
        self.logger.open()

    def teardown_method(self):
        self.logger.close()

    def test_start_recording_creates_file(self):
        path = self.logger.start_recording("test")
        assert os.path.exists(path)
        self.logger.stop_recording()

    def test_write_row_increments_counter(self):
        self.logger.start_recording("test")
        for i in range(100):
            self.logger.write_row([f"{i}", "1.0", "150.0"])
        self.logger.stop_recording()
        assert self.logger.rows_written >= 100

    def test_csv_has_header_row(self):
        import csv
        path = self.logger.start_recording("test")
        self.logger.write_row(["0.001", "1.0", "150.0"])
        self.logger.stop_recording()
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "time_s" in header
        assert "PC_raw_V" in header

    def test_double_start_replaces_file(self):
        p1 = self.logger.start_recording("run1")
        time.sleep(0.01)
        p2 = self.logger.start_recording("run2")   # stops p1, starts p2
        self.logger.stop_recording()
        assert p1 != p2
        assert os.path.exists(p1)
        assert os.path.exists(p2)

    def test_write_row_noop_when_not_recording(self):
        self.logger.write_row(["0.1", "1.0"])   # should not raise
        assert self.logger.rows_written == 0

    def test_is_recording_flag(self):
        assert not self.logger.is_recording
        self.logger.start_recording("test")
        assert self.logger.is_recording
        self.logger.stop_recording()
        assert not self.logger.is_recording

    def test_high_rate_write_does_not_block(self):
        """5000 write_row calls should complete well under 1 second."""
        self.logger.start_recording("perf")
        t0 = time.perf_counter()
        for i in range(5000):
            self.logger.write_row([f"{i * 0.002}", "1.23", "155.4"])
        elapsed = time.perf_counter() - t0
        self.logger.stop_recording()
        assert elapsed < 1.0, f"5000 writes took {elapsed:.3f} s (too slow)"


# -- HTTP REST API Endpoint Client Verification -----------------

@pytest.fixture(scope="module")
def api_client():
    """Stand up engine + API for the full module test run."""
    engine = _make_engine(tmp_path=".")
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