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
import yaml

from daq.calculations import psi_to_pa
from daq.engine import SequenceRefused
from daq.manifest import load_actuators

from tests.software._helpers import (
    REPO_ACTUATORS,
    make_engine,
    wait_for_first_batch,
)


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
        assert eng._channel_status("pt0", psi_to_pa(200)) == "UNCONFIGURED"

    def test_unassigned_wins_even_with_no_reading(self):
        """UNCONFIGURED describes the config, so it doesn't need a value."""
        eng = make_engine()
        assert eng._channel_status("pt0", None) == "UNCONFIGURED"

    def test_monitored_channel_with_no_reading_has_no_status(self):
        """Bands exist but there's nothing to classify this batch."""
        eng = self._engine({"normal": [100, 300], "warning": [50, 400]})
        assert eng._channel_status("pt0", None) is None

    def test_inactive_channel_reports_unassigned_on_the_snapshot(self):
        eng = make_engine()
        eng.start()
        try:
            wait_for_first_batch(eng)
            assert eng.snapshot.channels["tc0"].status == "UNCONFIGURED"
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


# -- Unwired actuators  -----------

class TestUnwiredActuatorGate:
    """
    Channels and actuators used to treat an unfilled manifest differently.
    With no channel wired, start_stream() raises and the engine cannot
    pretend to run. With no actuator wired, nothing was gated: every
    device write raised, _run_sequence logged it and advanced, and the
    sequence reported completion having moved nothing.
    """

    @staticmethod
    def _seq_engine(tmp_path, steps, actuators_path=REPO_ACTUATORS):
        """An engine whose sequences/ holds an abort.yaml with `steps`."""
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir()
        (seq_dir / "abort.yaml").write_text(
            yaml.safe_dump({"post_record_seconds": 0, "steps": steps})
        )
        kwargs = {"sequence_dir": str(seq_dir)}
        if actuators_path is not None:
            kwargs["actuators_path"] = actuators_path
        return make_engine(**kwargs)

    @staticmethod
    def _wait_for_sequence_end(engine, timeout=5.0):
        """
        Join the sequence thread if one was spawned. The engine is never
        start()ed here, so the snapshot is not a usable signal.

        A refused sequence is now rejected before any thread starts, so
        having no thread at all is a valid outcome.
        """
        thread = engine._sequence_thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        assert not thread.is_alive(), "sequence never finished"

    def test_shipped_manifest_reports_every_actuator_unwired(self):
        engine = make_engine(actuators_path=REPO_ACTUATORS)
        assert set(engine.unwired_actuators) == {
            spec.id for spec in engine.actuator_specs
        }, "the shipped actuators.yaml assigns no pins at all"

    def test_fixture_manifest_reports_nothing_unwired(self):
        assert make_engine().unwired_actuators == []

    def test_sequence_commanding_an_unwired_actuator_is_refused(self, tmp_path):
        engine = self._seq_engine(tmp_path, [[0.0, "lox_vent", 0]])
        engine.abort()
        self._wait_for_sequence_end(engine)

        log = engine.event_log
        assert any("REFUSED" in e and "lox_vent" in e for e in log), log
        assert not any("T+0.00s" in e for e in log), (
            "a step ran despite the actuator having no pin assignment"
        )

    def test_refused_abort_still_drives_the_hardware_safe(self, tmp_path):
        """An ordered closure is impossible, so fall back to de-energising."""
        engine = self._seq_engine(tmp_path, [[0.0, "lox_vent", 0]])
        calls = []
        engine._device.all_safe = lambda: calls.append("all_safe")

        engine.abort()
        self._wait_for_sequence_end(engine)
        assert calls == ["all_safe"]

    def test_sequence_naming_an_actuator_that_does_not_exist_is_refused(
        self, tmp_path
    ):
        engine = self._seq_engine(
            tmp_path, [[0.0, "no_such_valve", 1]], actuators_path=None
        )
        engine.abort()
        self._wait_for_sequence_end(engine)
        assert any("REFUSED" in e and "no_such_valve" in e
                   for e in engine.event_log)

    def test_fully_wired_sequence_still_runs_every_step(self, tmp_path):
        engine = self._seq_engine(
            tmp_path,
            [[0.0, "lox_vent", 1], [0.0, "fuel_vent", 1]],
            actuators_path=None,
        )
        engine.abort()
        self._wait_for_sequence_end(engine)

        log = engine.event_log
        assert not any("REFUSED" in e for e in log), log
        assert engine._device.read_actuator("lox_vent") == 1
        assert engine._device.read_actuator("fuel_vent") == 1


class TestActuatorNormalState:
    """
    `normal` is the actuator's de-energised resting state, reported to
    Dashboard alongside commanded state. It is a P&ID fact, so an
    unconfirmed one stays null rather than being defaulted to a guess.
    """

    @staticmethod
    def _manifest(tmp_path, **extra):
        entry = {"id": "lox_vent", "type": "binary_dio", "dio": "EIO4", **extra}
        path = tmp_path / "actuators.yaml"
        path.write_text(yaml.safe_dump({"actuators": [entry]}))
        return str(path)

    @pytest.mark.parametrize("normal", ["open", "closed"])
    def test_valid_resting_states_parse(self, tmp_path, normal):
        specs = load_actuators(self._manifest(tmp_path, normal=normal))
        assert specs[0].normal == normal

    def test_omitted_resting_state_stays_none(self, tmp_path):
        specs = load_actuators(self._manifest(tmp_path))
        assert specs[0].normal is None

    def test_explicitly_null_resting_state_stays_none(self, tmp_path):
        specs = load_actuators(self._manifest(tmp_path, normal=None))
        assert specs[0].normal is None

    def test_unrecognised_resting_state_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="invalid normal"):
            load_actuators(self._manifest(tmp_path, normal="ajar"))

    def test_shipped_manifest_leaves_every_resting_state_unconfirmed(self):
        engine = make_engine(actuators_path=REPO_ACTUATORS)
        assert all(spec.normal is None for spec in engine.actuator_specs)


class TestDerivedChannels:
    """
    Mass flow and mixture ratio are computed from the differential PTs and
    the geometry in config.yaml's `derived` block.
    """

    CFG = {
        "lox": {
            "dp_channel": "dpt0",
            "inlet_temp_channel": "tc0",
            "orifice_diameter_in": 0.199,
            "orifice_count": 1,
            "discharge_coefficient": 0.6,
        },
        "fuel": {
            "dp_channel": "dpt1",
            "orifice_diameter_in": 0.280,
            "orifice_count": 1,
            "discharge_coefficient": 0.67,
            "density_kg_m3": 800.0,
        },
    }

    @staticmethod
    def _latest(dp0=psi_to_pa(100.0), dp1=psi_to_pa(50.0), tc0=-160.0):
        return {"dpt0": dp0, "dpt1": dp1, "tc0": tc0}

    def _engine(self, cfg=None):
        return make_engine(derived=cfg if cfg is not None else self.CFG)

    def test_all_three_outputs_are_always_present(self):
        """Shape is stable even with no config - values go null, keys don't."""
        result = make_engine(derived={})._compute_derived(self._latest())
        assert set(result) == {
            "lox_mass_flow_rate_kg_s",
            "fuel_mass_flow_rate_kg_s",
            "mixture_ratio",
        }
        assert all(v is None for v in result.values())

    def test_fuel_flow_computes_without_a_thermocouple(self):
        """Fuel density is a constant, so fuel flow needs no temperature."""
        result = self._engine()._compute_derived(self._latest(tc0=None))
        assert result["fuel_mass_flow_rate_kg_s"] > 0.0

    def test_lox_flow_requires_inlet_temperature(self):
        """LOX density comes from a table keyed on temperature."""
        result = self._engine()._compute_derived(self._latest(tc0=None))
        assert result["lox_mass_flow_rate_kg_s"] is None
        assert result["mixture_ratio"] is None

    def test_non_positive_dp_yields_none(self):
        result = self._engine()._compute_derived(self._latest(dp0=0.0, dp1=-1.0))
        assert result["lox_mass_flow_rate_kg_s"] is None
        assert result["fuel_mass_flow_rate_kg_s"] is None

    def test_missing_geometry_yields_none_not_a_guess(self):
        cfg = {"lox": {"dp_channel": "dpt0", "inlet_temp_channel": "tc0"},
               "fuel": {"dp_channel": "dpt1"}}
        result = self._engine(cfg)._compute_derived(self._latest())
        assert all(v is None for v in result.values())

    def test_orifice_count_scales_flow(self):
        """Flow area, and so mass flow, scales linearly with element count."""
        one  = self._engine()._compute_derived(self._latest())
        four_cfg = {
            "lox": self.CFG["lox"],
            "fuel": {**self.CFG["fuel"], "orifice_count": 4},
        }
        four = self._engine(four_cfg)._compute_derived(self._latest())
        ratio = four["fuel_mass_flow_rate_kg_s"] / one["fuel_mass_flow_rate_kg_s"]
        assert abs(ratio - 4.0) < 1e-9

    def test_shipped_config_names_the_differential_channels(self):
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)["derived"]
        assert cfg["lox"]["dp_channel"] == "dpt0"
        assert cfg["fuel"]["dp_channel"] == "dpt1"


class TestFireRefusalReachesTheCaller:
    """
    A refused sequence used to be discovered inside the sequence thread,
    after fire() had already returned. POST /fire answered 200 and nothing
    moved. Validation now runs on the caller's thread so the refusal is
    something a caller can act on.
    """

    @staticmethod
    def _engine(tmp_path, steps, name="fire.yaml", actuators_path=REPO_ACTUATORS):
        """
        fire.yaml uses the named-phase schema (_load_fire_sequence); every
        other name (abort.yaml) keeps the flat schema (_load_sequence).
        Callers still pass flat [t, actuator, state] triples either way -
        for fire.yaml each triple becomes its own auto-named single-action
        step.
        """
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir(exist_ok=True)
        if name == "fire.yaml":
            payload = {
                "post_record_seconds": 0,
                "steps": [
                    {"name": f"step_{i}", "actions": [list(s)]}
                    for i, s in enumerate(steps)
                ],
            }
        else:
            payload = {"post_record_seconds": 0, "steps": steps}
        (seq_dir / name).write_text(yaml.safe_dump(payload))
        kwargs = {"sequence_dir": str(seq_dir)}
        if actuators_path is not None:
            kwargs["actuators_path"] = actuators_path
        return make_engine(**kwargs)

    def test_fire_raises_when_an_actuator_is_unwired(self, tmp_path):
        engine = self._engine(tmp_path, [[0.0, "lox_vent", 1]])
        with pytest.raises(SequenceRefused, match="not wired"):
            engine.fire()

    def test_refused_fire_starts_nothing(self, tmp_path):
        engine = self._engine(tmp_path, [[0.0, "lox_vent", 1]])
        with pytest.raises(SequenceRefused):
            engine.fire()
        assert engine._sequence_thread is None
        assert engine.snapshot.sequence_active is False

    def test_unreadable_sequence_file_also_raises(self, tmp_path):
        """The other silent-success path: a malformed fire.yaml."""
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir()
        (seq_dir / "fire.yaml").write_text("steps: [[0.0, 'lox_vent']]")
        engine = make_engine(sequence_dir=str(seq_dir))
        with pytest.raises(SequenceRefused, match="could not be loaded"):
            engine.fire()

    def test_missing_sequence_file_raises(self, tmp_path):
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir()
        engine = make_engine(sequence_dir=str(seq_dir))
        with pytest.raises(SequenceRefused, match="could not be loaded"):
            engine.fire()

    def test_a_drivable_fire_still_starts(self, tmp_path):
        engine = self._engine(tmp_path, [[0.0, "lox_vent", 1]],
                              actuators_path=None)
        engine.fire()
        assert engine._sequence_thread is not None
        engine._sequence_thread.join(timeout=5.0)

    def test_abort_never_raises_and_still_safes_the_hardware(self, tmp_path):
        """
        abort() absorbs the refusal fire() propagates: leaving the cart
        de-energised matters more than reporting the failure upward.
        """
        engine = self._engine(tmp_path, [[0.0, "lox_vent", 0]],
                              name="abort.yaml")
        calls = []
        engine._device.all_safe = lambda: calls.append("all_safe")

        engine.abort()   # must not raise

        assert calls == ["all_safe"]
        assert any("REFUSED" in e for e in engine.event_log)


class TestFireSequencing:
    """
    POST /sequence/* is a pausable/seekable runner for fire.yaml. While
    stopped, seeking only moves the playhead (no hardware effect). While
    running, a forward seek fires every action it skips over; a backward seek is refused.

    Scheduled gamma 100s out so sequences stay active until explicitly stopped
    or aborted during tests.
    """

    FIRE_STEPS = {
        "post_record_seconds": 0,
        "steps": [
            {"name": "alpha", "actions": [[0.02, "lox_vent", 1]]},
            {"name": "beta",  "actions": [[0.06, "lox_purge", 1], [0.05, "fuel_vent", 1]]},
            {"name": "gamma", "actions": [[100.0, "fuel_purge", 1]]},
        ],
    }
    ABORT_STEPS = {
        "post_record_seconds": 0,
        "steps": [[0.0, "lox_vent", 0]],
    }

    @classmethod
    def _engine(cls, tmp_path, logger=None, fire_steps=None):
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir(exist_ok=True)
        (seq_dir / "fire.yaml").write_text(yaml.safe_dump(fire_steps or cls.FIRE_STEPS))
        (seq_dir / "abort.yaml").write_text(yaml.safe_dump(cls.ABORT_STEPS))
        return make_engine(sequence_dir=str(seq_dir), logger=logger)

    def test_dump_lists_named_steps_in_order_with_earliest_action_as_start(self, tmp_path):
        engine = self._engine(tmp_path)
        dump = engine.dump_fire_sequence()
        assert [s["name"] for s in dump] == ["alpha", "beta", "gamma"]
        # beta's two actions are 0.06 then 0.05 in file order - start_s must
        # be the smaller time, not "whichever action came first".
        beta = next(s for s in dump if s["name"] == "beta")
        assert beta["start_s"] == 0.05
        assert beta["actions"] == [[0.06, "lox_purge", 1], [0.05, "fuel_vent", 1]]

    def test_current_step_is_none_before_first_phase_start_time(self, tmp_path):
        engine = self._engine(tmp_path)
        assert engine.snapshot.sequence_step is None  # never started
        status = engine.sequence_status()
        assert status["step"] is None

    def test_jump_to_step_seeks_to_earliest_action_in_group(self, tmp_path):
        engine = self._engine(tmp_path)
        status = engine.jump_to_step("beta")
        # beta's actions are [0.06, 0.05] in file order - must seek to the
        # smaller time (0.05s), not the first-listed one (0.06s).
        assert status["step"] == "beta"
        assert status["sequence_time_ms"] == 50

    def test_backward_jump_refused_while_running(self, tmp_path):
        # engine.snapshot only updates via the acquisition loop (start()),
        # which isn't running here - check the live flag directly instead,
        # matching TestUnwiredActuatorGate's convention.
        engine = self._engine(tmp_path)
        engine.start_sequence()
        time.sleep(0.1)  # well past alpha/beta, clearly ahead of T+0
        try:
            assert engine._sequence_active is True
            with pytest.raises(RuntimeError):
                engine.set_sequence_time(0.0)
            with pytest.raises(RuntimeError):
                engine.jump_to_step("alpha")  # alpha is behind the current position
        finally:
            engine.stop_sequence()

    def test_forward_jump_while_running_replays_skipped_actions(self, tmp_path):
        engine = self._engine(tmp_path)
        calls = []
        orig_write = engine._device.write_actuator
        def spy(name, state):
            calls.append((name, state))
            return orig_write(name, state)
        engine._device.write_actuator = spy

        engine.start_sequence()
        status = engine.set_sequence_time(50.0)  # immediately, before alpha/beta tick naturally

        # alpha + beta's two actions fired as part of the jump; gamma (100s
        # out) did not.
        assert set(calls) == {("lox_vent", 1), ("lox_purge", 1), ("fuel_vent", 1)}
        assert status["step"] == "beta"
        # A few microseconds pass between the jump landing and this status
        # read, so it's ~50000ms.
        assert 50000 <= status["sequence_time_ms"] < 50050

        # Ticking resumed live from the new position, not left stopped.
        assert engine._sequence_active is True
        engine.stop_sequence()

    def test_step_unknown_name_raises_keyerror(self, tmp_path):
        engine = self._engine(tmp_path)
        with pytest.raises(KeyError):
            engine.jump_to_step("not_a_real_step")

    def test_stop_freezes_without_abort_or_all_safe_or_stopping_recording(self, tmp_path):
        logger = _StubLogger()
        engine = self._engine(tmp_path, logger=logger)
        all_safe_calls = []
        engine._device.all_safe = lambda: all_safe_calls.append("all_safe")

        engine.fire()
        time.sleep(0.15)  # let alpha + beta apply; gamma is 100s out
        status = engine.stop_sequence()

        assert engine._sequence_active is False
        assert all_safe_calls == []
        assert logger.stop_calls == 0
        assert status["step"] == "beta"

    def test_stop_then_start_resumes_without_replaying_earlier_actions(self, tmp_path):
        engine = self._engine(tmp_path)
        calls = []
        orig_write = engine._device.write_actuator
        def spy(name, state):
            calls.append((name, state))
            return orig_write(name, state)
        engine._device.write_actuator = spy

        engine.fire()
        time.sleep(0.15)
        engine.stop_sequence()
        first_round = list(calls)
        assert set(first_round) == {("lox_vent", 1), ("lox_purge", 1), ("fuel_vent", 1)}

        engine.start_sequence()
        time.sleep(0.15)
        engine.stop_sequence()

        assert calls == first_round, "resuming replayed an already-applied action"

    def test_abort_still_preempts_mid_run_fire_and_invalidates_position(self, tmp_path):
        engine = self._engine(tmp_path)
        engine.fire()
        time.sleep(0.15)  # alpha + beta applied, sitting on gamma (100s out)

        engine.abort()
        if engine._sequence_thread:
            engine._sequence_thread.join(timeout=2.0)

        assert any("ABORT" in e or "[abort]" in e for e in engine.event_log)
        # Frozen where it was (not reset to 0) and invalidated - resumable
        # only via a fresh fire().
        assert engine._fire_position_s == pytest.approx(0.06, abs=0.02)
        assert engine._fire_invalidated is True
        assert engine._sequence_active is False
        with pytest.raises(RuntimeError):
            engine.start_sequence()

    def test_fire_after_abort_clears_invalidation(self, tmp_path):
        engine = self._engine(tmp_path)
        engine.fire()
        time.sleep(0.15)
        engine.abort()
        if engine._sequence_thread:
            engine._sequence_thread.join(timeout=2.0)
        assert engine._fire_invalidated is True

        engine.fire()  # fresh fire clears the invalidation and starts at T=0
        assert engine._fire_invalidated is False
        time.sleep(0.05)
        assert engine._sequence_active is True
        engine.stop_sequence()

    def test_natural_completion_keeps_the_clock_running(self, tmp_path):
        """After the last scripted action, elapsed keeps counting up (like
        a real launch clock after liftoff) instead of freezing or
        resetting."""
        engine = self._engine(tmp_path, fire_steps={
            "post_record_seconds": 0,
            "steps": [{"name": "only", "actions": [[0.01, "lox_vent", 1]]}],
        })
        engine.fire()
        time.sleep(0.1)  # completes within this window
        assert engine._sequence_active is False
        assert engine._fire_invalidated is False

        status1 = engine.sequence_status()
        time.sleep(0.1)
        status2 = engine.sequence_status()
        assert status2["sequence_time_ms"] > status1["sequence_time_ms"]


class _StubLogger:
    """Minimal Logger stand-in for asserting recording start/stop calls."""

    def __init__(self):
        self._recording = False
        self.stop_calls = 0
        self.start_calls = 0

    @property
    def is_recording(self):
        return self._recording

    def start_recording(self, prefix="data"):
        self.start_calls += 1
        self._recording = True

    def stop_recording(self):
        self.stop_calls += 1
        self._recording = False

    def write_row(self, row_vals):
        pass
