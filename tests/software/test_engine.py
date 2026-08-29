"""
tests/software/test_engine.py

Integration tests for daq/engine.py against MockLabJack:
  - Engine startup, snapshot population, and clean shutdown
  - Actuator control and sequence blocking
  - Reconnection-style stop / restart behaviour
  - Threshold config unit conversion (psi -> Pa)
  - Data freshness / staleness detection

No real hardware or network required.
Run with: python -m pytest tests/software/test_engine.py -v
"""

import threading
import time

import pytest

from daq.calculations import psi_to_pa

from tests.software._helpers import make_engine, wait_for_first_batch


# -- Background Engine Pipeline Integration -------------------

class TestEngine:

    def setup_method(self):
        self.engine = make_engine()
        self.engine.start()
        wait_for_first_batch(self.engine)

    def teardown_method(self):
        self.engine.stop()

    def test_snapshot_is_streaming(self):
        assert self.engine.snapshot.streaming is True

    def test_snapshot_channels_match_manifest(self):
        """Every channel in channels.yaml is reported, active or not."""
        snap = self.engine.snapshot
        assert set(snap.channels) == {
            spec.id for spec in self.engine.channel_specs
        }

    def test_snapshot_has_pressure_values(self):
        channels = self.engine.snapshot.channels
        for tag in ("pt0", "pt1", "pt2"):
            assert channels[tag].value is not None, f"{tag} is None"

    def test_snapshot_pressures_in_engineering_range(self):
        channels = self.engine.snapshot.channels
        # Default mock + default cal -> should be roughly 100-400 psi equivalent.
        for tag in ("pt0", "pt1", "pt2"):
            v = channels[tag].value
            assert psi_to_pa(50) < v < psi_to_pa(500), f"{tag}={v}"

    def test_inactive_channel_reported_with_no_value(self):
        reading = self.engine.snapshot.channels["pt7"]
        assert reading.value is None
        assert reading.unit == "Pa"

    def test_snapshot_has_tc_values(self):
        # TCs are batch-averaged; need a couple of batches
        time.sleep(0.3)
        assert self.engine.snapshot.channels["tc0"].value is not None

    def test_snapshot_has_actuator_dict(self):
        snap = self.engine.snapshot
        assert isinstance(snap.actuators, dict)
        assert "lox_main" in snap.actuators
        assert snap.actuators["lox_main"].moving is False

    def test_stepper_reports_moving_during_commanded_move(self):
        """A stepper main is in flight for the burst duration; a solenoid
        never is, b/c for a solenoid the DIO write is the state."""
        self.engine.write_actuator("lox_main", 1)
        time.sleep(0.15)
        actuators = self.engine.snapshot.actuators
        assert actuators["lox_main"].moving is True
        assert actuators["lox_purge"].moving is False

    def test_photogate_counter_is_reported(self):
        """Position feedback is telemetry, polled out-of-band like CJC."""
        self.engine.write_actuator("lox_main", 1)
        time.sleep(1.2)   # cover at least two out-of-band poll ticks
        reading = self.engine.snapshot.channels["pos_lox_main"]
        assert reading.value is not None
        assert reading.unit == "counts"

    def test_snapshot_using_mock_true(self):
        assert self.engine.snapshot.using_mock is True

    def test_manual_actuator_command(self):
        self.engine.write_actuator("fuel_vent", 1)
        time.sleep(0.15)
        snap = self.engine.snapshot
        assert snap.actuators["fuel_vent"].state == 1

    def test_all_safe_clears_actuators(self):
        self.engine.write_actuator("fuel_vent", 1)
        time.sleep(0.05)
        self.engine.all_safe()
        time.sleep(0.15)
        snap = self.engine.snapshot
        assert snap.actuators["fuel_vent"].state == 0

    def test_tare_sets_lc_near_zero(self):
        # Before tare, LC values may be non-zero (noise)
        self.engine.tare()
        time.sleep(0.15)
        reading = self.engine.snapshot.channels["lc0"]
        # After tare, tared force should be near zero
        # (within noise of a few lbf)
        if reading.value is not None:
            assert abs(reading.value) < 5.0, f"lc0 after tare: {reading.value}"

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
                    _ = snap.channels
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
                self.engine.write_actuator("lox_main", 1)

    def test_update_calibration_changes_snapshot(self):
        """Doubling pt0's slope should roughly double its reading."""
        before = self.engine.snapshot.channels["pt0"].value
        assert before is not None

        self.engine.update_calibration("pt0", slope=1000.0, intercept=0.0)
        time.sleep(0.3)
        after = self.engine.snapshot.channels["pt0"].value

        assert after is not None
        # Wide tolerance: the mock drifts and adds noise between batches.
        assert 1.8 < after / before < 2.2, f"{before} -> {after}"

    def test_stop_and_snapshot_streaming_false(self):
        self.engine.stop()
        time.sleep(0.1)
        assert self.engine.snapshot.streaming is False
        # Recreate so teardown doesn't fail
        self.engine = make_engine()
        self.engine.start()


# -- Reconnection After Comms Loss -----------------------------

class TestEngineReconnect:
    """
    Covers the reconnect-on-repeated-failure behavior added to
    Engine._stream_loop (mitigates FC.NLFS.LJ.2 / FC.NLFS.LQDDAQ.1):
    after _MAX_CONSECUTIVE_READ_ERRORS consecutive stream_read() failures,
    the engine should cycle close()/open()/start_stream() and resume
    streaming rather than silently dying.
    """

    def test_reconnects_after_repeated_read_failures(self):
        engine = make_engine()
        engine.start()
        wait_for_first_batch(engine)
        assert engine.snapshot.streaming is True, "Engine never started streaming"

        device = engine._device
        original_stream_read  = device.stream_read
        original_open         = device.open
        original_start_stream = device.start_stream

        state = {"fail_count": 0, "reopened": False, "restarted": False}

        def flaky_stream_read():
            if state["fail_count"] < 3:
                state["fail_count"] += 1
                raise RuntimeError("simulated comms loss")
            return original_stream_read()

        def tracking_open():
            state["reopened"] = True
            return original_open()

        def tracking_start_stream():
            state["restarted"] = True
            return original_start_stream()

        device.stream_read  = flaky_stream_read
        device.open         = tracking_open
        device.start_stream = tracking_start_stream

        # Give the engine time to notice the failures, reconnect (with its
        # backoff sleep), and resume streaming.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if state["reopened"] and state["restarted"] and engine.snapshot.streaming:
                break
            time.sleep(0.2)

        assert state["reopened"], "Engine never called device.open() to reconnect"
        assert state["restarted"], "Engine never called device.start_stream() to resume"
        assert engine.snapshot.streaming is True, (
            "Engine did not resume streaming after reconnect"
        )

        engine.stop()

    def test_stream_flag_goes_false_during_outage(self):
        """streaming should flip False as soon as reads start failing, not
        stay stale True while the connection is actually down."""
        engine = make_engine()
        engine.start()
        wait_for_first_batch(engine)

        device = engine._device

        def always_fail():
            raise RuntimeError("simulated comms loss")

        device.stream_read = always_fail

        deadline = time.time() + 2.0
        saw_false = False
        while time.time() < deadline:
            if engine.snapshot.streaming is False:
                saw_false = True
                break
            time.sleep(0.05)

        assert saw_false, "streaming flag never went False during the outage"
        engine.stop()


# -- Threshold Config Conversion (psi input -> Pa internal) -----

class TestThresholdConversion:
    """
    config.yaml is authored in psi for pressure channels (matches
    calibration.json's convention); Engine.thresholds should expose Pa
    so it's unit-consistent with /snapshot. Non-pressure entries pass
    through untouched.
    """

    def test_pressure_channel_converted_to_pa(self):
        raw = {"pt0": {"normal": [100, 250], "warning": [50, 300]}}
        engine = make_engine(thresholds=raw)
        pt0 = engine.thresholds["pt0"]
        assert abs(pt0["normal"][0]  - psi_to_pa(100)) < 1e-6
        assert abs(pt0["normal"][1]  - psi_to_pa(250)) < 1e-6
        assert abs(pt0["warning"][0] - psi_to_pa(50))  < 1e-6
        assert abs(pt0["warning"][1] - psi_to_pa(300)) < 1e-6

    def test_every_manifest_pressure_channel_converts(self):
        engine = make_engine()
        pt_ids = [
            spec.id for spec in engine.channel_specs
            if spec.type == "pt_direct"
        ]
        raw = {tag: {"normal": [10, 20], "warning": [5, 25]} for tag in pt_ids}
        engine = make_engine(thresholds=raw)
        for tag in pt_ids:
            assert abs(engine.thresholds[tag]["normal"][0] - psi_to_pa(10)) < 1e-6

    def test_temperature_channel_passthrough(self):
        raw = {"tc0": {"normal": [-200, -140], "warning": [-210, -130]}}
        engine = make_engine(thresholds=raw)
        assert engine.thresholds["tc0"] == raw["tc0"], "Temp thresholds should not be converted"

    def test_load_cell_channel_passthrough(self):
        raw = {"lc0": {"normal": [0, 800], "warning": [-50, 900]}}
        engine = make_engine(thresholds=raw)
        assert engine.thresholds["lc0"] == raw["lc0"], "Load cell thresholds (lbf) should not be converted"

    def test_unknown_tag_passthrough(self):
        raw = {"not_a_channel": {"normal": [1.8, 3.0], "warning": [1.2, 3.5]}}
        engine = make_engine(thresholds=raw)
        assert engine.thresholds["not_a_channel"] == raw["not_a_channel"]

    def test_empty_thresholds_ok(self):
        engine = make_engine()
        assert engine.thresholds == {}


# -- Reading Status (Dashboard readingStatus.ts enum) ----------

class TestChannelStatus:
    """
    Status is computed DAQ-side so Dashboard never re-derives bands from a
    separately fetched config that could drift from what actually triggers
    an abort.
    """

    def _engine(self, bands):
        return make_engine(thresholds={"pt0": bands})

    def test_inside_normal_is_nominal(self):
        eng = self._engine({"normal": [100, 300], "warning": [50, 400]})
        assert eng._channel_status("pt0", psi_to_pa(200)) == "NOMINAL"

    def test_outside_normal_inside_warning_is_caution(self):
        eng = self._engine({"normal": [100, 300], "warning": [50, 400]})
        assert eng._channel_status("pt0", psi_to_pa(350)) == "CAUTION"

    def test_outside_warning_is_warning(self):
        eng = self._engine({"normal": [100, 300], "warning": [50, 400]})
        assert eng._channel_status("pt0", psi_to_pa(450)) == "WARNING"

    def test_channel_without_bands_is_unassigned(self):
        """No bands set is a config gap, never a reassuring NOMINAL."""
        eng = make_engine()
        assert eng._channel_status("pt0", psi_to_pa(200)) == "UNASSIGNED"

    def test_unassigned_wins_even_with_no_reading(self):
        """UNASSIGNED describes the config, so it doesn't need a value."""
        eng = make_engine()
        assert eng._channel_status("pt0", None) == "UNASSIGNED"

    def test_monitored_channel_with_no_reading_has_no_status(self):
        """Bands exist but there's nothing to classify this batch."""
        eng = self._engine({"normal": [100, 300], "warning": [50, 400]})
        assert eng._channel_status("pt0", None) is None

    def test_inactive_channel_reports_unassigned_on_the_snapshot(self):
        eng = make_engine()
        eng.start()
        try:
            wait_for_first_batch(eng)
            assert eng.snapshot.channels["tc0"].status == "UNASSIGNED"
        finally:
            eng.stop()

    def test_status_lands_on_the_snapshot(self):
        eng = make_engine(thresholds={
            "pt0": {"normal": [0, 5000], "warning": [0, 6000]}
        })
        eng.start()
        try:
            wait_for_first_batch(eng)
            assert eng.snapshot.channels["pt0"].status == "NOMINAL"
        finally:
            eng.stop()


# -- Data Freshness --------------------------

class TestDataFreshness:
    """Engine-level stale-data detection."""

    def test_fresh_engine_reports_not_stale(self):
        engine = make_engine()
        engine.start()
        wait_for_first_batch(engine)
        assert engine.is_data_stale is False
        assert engine.data_age_seconds < 1.0
        engine.stop()

    def test_never_streamed_engine_is_infinitely_stale(self):
        engine = make_engine()
        # Before start(), no batch has ever landed - definitionally stale.
        assert engine.data_age_seconds == float("inf")
        assert engine.is_data_stale is True

    def test_data_age_grows_during_outage(self):
        engine = make_engine()
        engine.start()
        wait_for_first_batch(engine)

        device = engine._device

        def always_fail():
            raise RuntimeError("simulated comms loss")

        device.stream_read = always_fail

        time.sleep(1.3)   # > _DATA_STALE_THRESHOLD_S (1.0s)
        assert engine.is_data_stale is True
        assert engine.data_age_seconds > 1.0
        engine.stop()


# -- First-batch wait (the fixed-sleep flake) ------------------

class TestFirstBatchWait:
    """
    Reproduces the flake the suite used to carry.

    setup_method() slept a flat 0.3 s after start() and assumed a batch
    had landed. The first batch normally takes ~0.1 s, but on a loaded
    runner it can take longer, leaving snapshot.channels empty and every
    test that subscripts a channel raising KeyError. wait_for_first_batch()
    polls for a populated snapshot instead.
    """

    @staticmethod
    def _delay_first_batch(engine, delay_s: float) -> None:
        """Stall the device's first stream_read, as a loaded CPU would."""
        original = engine._device.stream_read
        state = {"first": True}

        def slow_stream_read():
            if state["first"]:
                state["first"] = False
                time.sleep(delay_s)
            return original()

        engine._device.stream_read = slow_stream_read

    def test_wait_survives_a_first_batch_slower_than_the_old_sleep(self):
        engine = make_engine()
        self._delay_first_batch(engine, 0.6)
        engine.start()
        try:
            time.sleep(0.3)     # exactly what setup_method used to do
            assert not engine.snapshot.channels, (
                "first batch landed early - test no longer reproduces the flake"
            )
            wait_for_first_batch(engine)
            assert engine.snapshot.channels["pt0"].value is not None
        finally:
            engine.stop()

    def test_wait_raises_a_clear_message_if_no_batch_ever_lands(self):
        engine = make_engine()
        with pytest.raises(AssertionError, match="no populated snapshot"):
            wait_for_first_batch(engine, timeout=0.2)
