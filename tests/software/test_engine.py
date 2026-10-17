"""
tests/software/test_engine.py

daq/engine.py against MockLabJack: snapshots, actuators, reconnect,
thresholds, staleness, sequences and the CJC poll.
"""

import json
import threading
import time

import pytest
import yaml

from daq.calculations import psi_to_pa
from daq.engine import (
    FIRE_ABORTED,
    FIRE_COMPLETED,
    FIRE_IDLE,
    FIRE_PAUSED,
    FIRE_RUNNING,
    Engine,
    SequenceRefused,
)
from daq.manifest import load_actuators

from tests.software._helpers import (
    FULL_ACTUATORS,
    FULL_CHANNELS,
    REPO_ACTUATORS,
    REPO_CONFIG,
    UNWIRED_ACTUATORS,
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
        # Mock pt0 is ~200 psi at the default 400 psi/V.
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
        """A stepper is moving for the burst; a solenoid never is."""
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

    def test_tare_zeroes_a_loaded_cell(self):
        # The mock LC rests near 0, so offset it first or tare proves nothing.
        self.engine.update_calibration("lc0", slope=440.92, intercept=100.0)
        time.sleep(0.25)
        before = self.engine.snapshot.channels["lc0"].value
        assert before is not None and before > 50.0

        self.engine.tare()
        time.sleep(0.25)
        after = self.engine.snapshot.channels["lc0"].value
        assert abs(after) < 5.0, f"lc0 after tare: {after}"

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
        self.engine.abort()   # the fixture's abort.yaml is runnable and takes ~10 s
        assert self.engine._sequence_active
        with pytest.raises(RuntimeError):
            self.engine.write_actuator("lox_main", 1)

    def test_update_calibration_changes_snapshot(self):
        """Doubling pt0's slope should roughly double its reading."""
        before = self.engine.snapshot.channels["pt0"].value
        assert before is not None

        self.engine.update_calibration("pt0", slope=800.0, intercept=0.0)
        time.sleep(0.3)
        after = self.engine.snapshot.channels["pt0"].value

        assert after is not None
        # Wide tolerance: the mock drifts and adds noise between batches.
        assert 1.8 < after / before < 2.2, f"{before} -> {after}"

    def test_stop_and_snapshot_streaming_false(self):
        self.engine.stop()
        time.sleep(0.1)
        assert self.engine.snapshot.streaming is False
        # Recreate so teardown can stop it.
        self.engine = make_engine()
        self.engine.start()


# -- Reconnection After Comms Loss -----------------------------

class TestEngineReconnect:
    """After repeated read failures the engine reconnects and streams again."""

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

        # Time to fail, back off, reconnect and stream again.
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
        """streaming goes False as soon as reads fail."""
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
    """psi bands for pressure channels become Pa; other channels pass through."""

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
    """Status is computed here so Dashboard can't drift from config. Display only."""

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
        # No batch before start(), so stale.
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


# -- First-batch wait ----------------------------------------

class TestFirstBatchWait:
    """Wait for a populated snapshot, not a fixed sleep (that flaked under load)."""

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
    A sequence naming an unwired actuator is refused, not run to a fake completion.
    """

    @staticmethod
    def _seq_engine(tmp_path, steps, actuators_path=UNWIRED_ACTUATORS):
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
        """Join the sequence thread if any; a refused sequence starts none."""
        thread = engine._sequence_thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        assert not thread.is_alive(), "sequence never finished"

    def test_unfilled_manifest_reports_every_actuator_unwired(self):
        engine = make_engine(actuators_path=UNWIRED_ACTUATORS)
        assert set(engine.unwired_actuators) == {
            spec.id for spec in engine.actuator_specs
        }, "an actuators.yaml with no pins assigned"

    def test_fixture_manifest_reports_nothing_unwired(self):
        assert make_engine().unwired_actuators == []

    def test_sequence_commanding_an_unwired_actuator_is_refused(self, tmp_path):
        engine = self._seq_engine(tmp_path, [[0.0, "tank_depress", 0]])
        engine.abort()
        self._wait_for_sequence_end(engine)

        log = engine.event_log
        assert any("REFUSED" in e and "tank_depress" in e for e in log), log
        assert not any("T+0.00s" in e for e in log), (
            "a step ran despite the actuator having no pin assignment"
        )

    def test_refused_abort_still_drives_the_hardware_safe(self, tmp_path):
        """An ordered closure is impossible, so fall back to de-energising."""
        engine = self._seq_engine(tmp_path, [[0.0, "tank_depress", 0]])
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
            [[0.0, "tank_depress", 1], [0.0, "fuel_vent", 1]],
            actuators_path=None,
        )
        engine.abort()
        self._wait_for_sequence_end(engine)

        log = engine.event_log
        assert not any("REFUSED" in e for e in log), log
        assert engine._device.read_actuator("tank_depress") == 1
        assert engine._device.read_actuator("fuel_vent") == 1


class TestActuatorNormalState:
    """`normal` stays null unless the manifest sets it."""

    @staticmethod
    def _manifest(tmp_path, **extra):
        entry = {"id": "tank_depress", "type": "binary_dio", "dio": "CIO1", **extra}
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
    """Mass flow and O/F from the dPTs and config.yaml's `derived` block."""

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
        with open(REPO_CONFIG) as f:
            cfg = yaml.safe_load(f)["derived"]
        assert cfg["lox"]["dp_channel"] == "dpt0"
        assert cfg["fuel"]["dp_channel"] == "dpt1"


class TestFireRefusalReachesTheCaller:
    """A refused fire raises on the caller's thread, so /fire can return 409."""

    @staticmethod
    def _engine(tmp_path, steps, name="fire.yaml", actuators_path=UNWIRED_ACTUATORS):
        """
        Engine with sequences/`name` built from flat triples (one step each in
        fire.yaml).
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
        engine = self._engine(tmp_path, [[0.0, "tank_depress", 1]])
        with pytest.raises(SequenceRefused, match="not wired"):
            engine.fire()

    def test_refused_fire_starts_nothing(self, tmp_path):
        engine = self._engine(tmp_path, [[0.0, "tank_depress", 1]])
        with pytest.raises(SequenceRefused):
            engine.fire()
        assert engine._sequence_thread is None
        assert engine.snapshot.sequence_active is False

    def test_unreadable_sequence_file_also_raises(self, tmp_path):
        """The other silent-success path: a malformed fire.yaml."""
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir()
        (seq_dir / "fire.yaml").write_text("steps: [[0.0, 'tank_depress']]")
        engine = make_engine(sequence_dir=str(seq_dir))
        with pytest.raises(SequenceRefused, match="could not be loaded"):
            engine.fire()

    def test_missing_sequence_file_raises(self, tmp_path):
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir()
        engine = make_engine(sequence_dir=str(seq_dir))
        with pytest.raises(SequenceRefused, match="could not be loaded"):
            engine.fire()

    def test_dump_lists_steps_even_when_an_actuator_is_unwired(self, tmp_path):
        """GET /sequence is read-only; only fire() refuses unwired steps."""
        engine = self._engine(tmp_path, [[0.0, "tank_depress", 1]])
        assert [s["name"] for s in engine.dump_fire_sequence()] == ["step_0"]
        with pytest.raises(SequenceRefused, match="not wired"):
            engine.fire()

    def test_a_drivable_fire_still_starts(self, tmp_path):
        engine = self._engine(tmp_path, [[0.0, "tank_depress", 1]],
                              actuators_path=None)
        engine.fire()
        assert engine._sequence_thread is not None
        engine._sequence_thread.join(timeout=5.0)

    def test_abort_never_raises_and_still_safes_the_hardware(self, tmp_path):
        """abort() swallows the refusal and still de-energises."""
        engine = self._engine(tmp_path, [[0.0, "tank_depress", 0]],
                              name="abort.yaml")
        calls = []
        engine._device.all_safe = lambda: calls.append("all_safe")

        engine.abort()   # must not raise

        assert calls == ["all_safe"]
        assert any("REFUSED" in e for e in engine.event_log)

    @pytest.mark.parametrize("state", [2, -1, 0.5, True, "1"])
    def test_fire_refuses_a_state_other_than_0_or_1(self, tmp_path, state):
        engine = self._engine(tmp_path, [[0.0, "tank_depress", state]],
                              actuators_path=None)
        with pytest.raises(SequenceRefused, match="state must be 0 or 1"):
            engine.fire()

    def test_abort_with_a_bad_state_still_safes_the_hardware(self, tmp_path):
        engine = self._engine(tmp_path, [[0.0, "tank_depress", 2]],
                              name="abort.yaml", actuators_path=None)
        calls = []
        engine._device.all_safe = lambda: calls.append("all_safe")

        engine.abort()

        assert calls == ["all_safe"]
        assert any("state must be 0 or 1" in e for e in engine.event_log)


class TestFireSequencing:
    """
    Pause, resume and seek for fire.yaml. gamma sits at T+100 s so a run stays active.
    """

    FIRE_STEPS = {
        "post_record_seconds": 0,
        "steps": [
            {"name": "alpha", "actions": [[0.02, "tank_depress", 1]]},
            {"name": "beta",  "actions": [[0.06, "lox_purge", 1], [0.05, "fuel_vent", 1]]},
            {"name": "gamma", "actions": [[100.0, "fuel_purge", 1]]},
        ],
    }
    ABORT_STEPS = {
        "post_record_seconds": 0,
        "steps": [[0.0, "tank_depress", 0]],
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
        # beta lists 0.06 then 0.05; start_s is the earlier time.
        beta = next(s for s in dump if s["name"] == "beta")
        assert beta["start_s"] == 0.05
        assert beta["actions"] == [[0.06, "lox_purge", 1], [0.05, "fuel_vent", 1]]

    def test_current_step_is_none_before_first_phase_start_time(self, tmp_path):
        engine = self._engine(tmp_path)
        assert engine.snapshot.sequence_step is None  # never started
        assert engine.snapshot.sequence_state == FIRE_IDLE
        status = engine.sequence_status()
        assert status["step"] is None
        assert status["state"] == FIRE_IDLE

    def test_jump_to_step_seeks_to_earliest_action_in_group(self, tmp_path):
        engine = self._engine(tmp_path)
        status = engine.jump_to_step("beta")
        # Seeks to beta's earlier action (0.05 s), not its first-listed one.
        assert status["step"] == "beta"
        assert status["sequence_time_ms"] == 50

    def test_backward_jump_refused_while_running(self, tmp_path):
        # Engine not started, so check the live flag, not the snapshot.
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

        # The jump fired alpha and beta, not gamma (T+100 s).
        assert set(calls) == {("tank_depress", 1), ("lox_purge", 1), ("fuel_vent", 1)}
        assert status["step"] == "beta"
        # A few microseconds pass before this read.
        assert 50000 <= status["sequence_time_ms"] < 50050

        # Still ticking from the new position.
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
        assert engine._fire_state == FIRE_PAUSED
        assert all_safe_calls == []
        assert logger.stop_calls == 0
        assert status["step"] == "beta"
        assert status["state"] == FIRE_PAUSED

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
        assert set(first_round) == {("tank_depress", 1), ("lox_purge", 1), ("fuel_vent", 1)}

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
        # Frozen, not reset; only a fresh fire() can resume.
        assert engine._fire_position_s == pytest.approx(0.06, abs=0.02)
        assert engine._fire_invalidated is True
        assert engine._sequence_active is False
        assert engine._fire_state == FIRE_ABORTED
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
        assert engine._fire_state == FIRE_ABORTED

        engine.fire()  # fresh fire clears the invalidation and starts at T=0
        assert engine._fire_invalidated is False
        assert engine._fire_state == FIRE_RUNNING
        time.sleep(0.05)
        assert engine._sequence_active is True
        engine.stop_sequence()

    def test_natural_completion_keeps_the_clock_running(self, tmp_path):
        """After the last action the clock keeps counting."""
        engine = self._engine(tmp_path, fire_steps={
            "post_record_seconds": 0,
            "steps": [{"name": "only", "actions": [[0.01, "tank_depress", 1]]}],
        })
        engine.fire()
        time.sleep(0.1)  # completes within this window
        assert engine._sequence_active is False
        assert engine._fire_invalidated is False
        assert engine._fire_state == FIRE_COMPLETED

        status1 = engine.sequence_status()
        time.sleep(0.1)
        status2 = engine.sequence_status()
        assert status2["sequence_time_ms"] > status1["sequence_time_ms"]
        assert status1["state"] == FIRE_COMPLETED == status2["state"]


    def test_fire_while_running_raises_and_keeps_position(self, tmp_path):
        engine = self._engine(tmp_path)
        engine.fire()
        time.sleep(0.15)  # alpha and beta applied
        try:
            before = (engine._fire_next_idx, engine._fire_position_s)
            with pytest.raises(RuntimeError, match="already running"):
                engine.fire()
            assert (engine._fire_next_idx, engine._fire_position_s) == before
        finally:
            engine.stop_sequence()

    def test_fire_waits_out_a_finishing_thread(self, tmp_path):
        """A thread already marked inactive but not yet exited must not block /fire."""
        engine = self._engine(tmp_path)
        finishing = threading.Thread(target=time.sleep, args=(0.2,))
        finishing.start()
        engine._sequence_thread = finishing
        engine.fire()
        try:
            assert engine._sequence_active
            assert engine._sequence_thread is not finishing
        finally:
            engine.stop_sequence()

    def test_abort_of_a_paused_fire_reports_aborted(self, tmp_path):
        engine = self._engine(tmp_path)
        engine.fire()
        time.sleep(0.15)
        engine.stop_sequence()
        assert engine.sequence_status()["state"] == FIRE_PAUSED

        engine.abort()
        assert engine.sequence_status()["state"] == FIRE_ABORTED
        with pytest.raises(RuntimeError, match="aborted"):
            engine.start_sequence()

    POST_WINDOW_STEPS = {
        "post_record_seconds": 5.0,
        "steps": [{"name": "only", "actions": [[0.0, "tank_depress", 1]]}],
    }

    def _fire_into_post_window(self, tmp_path, logger):
        engine = self._engine(
            tmp_path, logger=logger, fire_steps=self.POST_WINDOW_STEPS
        )
        engine.fire()
        time.sleep(0.2)   # past the only action, inside the 5 s post window
        assert engine._sequence_active
        return engine

    @staticmethod
    def _finish_abort(engine):
        engine.abort()
        engine._sequence_thread.join(timeout=5.0)

    def test_abort_in_post_window_reports_aborted_and_keeps_recording(self, tmp_path):
        logger = _StubLogger()
        engine = self._fire_into_post_window(tmp_path, logger)

        self._finish_abort(engine)

        assert engine.sequence_status()["state"] == FIRE_ABORTED
        # Only abort.yaml stops the recording, after it has run.
        assert logger.stop_calls == 1

    def test_stop_and_seek_in_post_window_are_refused(self, tmp_path):
        engine = self._fire_into_post_window(tmp_path, _StubLogger())
        try:
            t0 = time.perf_counter()
            with pytest.raises(RuntimeError, match="post-record"):
                engine.stop_sequence()
            assert time.perf_counter() - t0 < 0.5, "stop blocked instead of refusing"
            with pytest.raises(RuntimeError, match="post-record"):
                engine.set_sequence_time(10.0)
            assert engine._sequence_active
        finally:
            self._finish_abort(engine)


class TestRepeatedAbort:
    """A second abort while abort.yaml runs neither blocks nor starts a second thread."""

    def test_second_abort_is_a_quick_no_op(self, tmp_path):
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir()
        (seq_dir / "abort.yaml").write_text(yaml.safe_dump({
            "post_record_seconds": 0,
            "steps": [[0.0, "tank_depress", 0], [1.0, "lox_purge", 0]],
        }))
        engine = make_engine(sequence_dir=str(seq_dir))

        engine.abort()
        first = engine._sequence_thread
        t0 = time.perf_counter()
        engine.abort()
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.2, f"second abort blocked for {elapsed:.2f} s"
        assert engine._sequence_thread is first
        assert any("already in progress" in e for e in engine.event_log)

        first.join(timeout=5.0)
        assert not engine._sequence_active


class TestCalibrationFile:
    """Updates must name a real channel; saving keeps the rest of the file."""

    @staticmethod
    def _engine(tmp_path, content):
        cal = tmp_path / "calibration.json"
        cal.write_text(json.dumps(content))
        engine = Engine(
            cal_path=str(cal), channels_path=FULL_CHANNELS,
            actuators_path=FULL_ACTUATORS,
        )
        return engine, cal

    def test_unknown_tag_is_refused(self, tmp_path):
        engine, _ = self._engine(tmp_path, {"PT": {}, "LC": {}})
        with pytest.raises(KeyError):
            engine.update_calibration("pt0_typo", 1.0, 0.0)

    def test_unknown_tag_in_file_is_logged(self, tmp_path):
        engine, _ = self._engine(
            tmp_path, {"PT": {"ptX": {"slope": 1.0, "intercept": 0.0}}}
        )
        assert any("not in the manifest" in e and "ptX" in e for e in engine.event_log)

    def test_save_keeps_other_keys(self, tmp_path):
        engine, cal = self._engine(tmp_path, {
            "note": "keep me",
            "PT": {"ptX": {"slope": 1.0, "intercept": 0.0}},
            "LC": {},
        })
        engine.update_calibration("pt0", 410.0, -1.0)
        engine.save_calibration()

        saved = json.loads(cal.read_text())
        assert saved["note"] == "keep me"
        assert saved["PT"]["ptX"] == {"slope": 1.0, "intercept": 0.0}
        assert saved["PT"]["pt0"] == {"slope": 410.0, "intercept": -1.0}

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


class TestColdJunctionPoll:
    """A failed CJC/counter read is logged and blanks the value, never a guessed CJC."""

    @staticmethod
    def _tc_spec(engine):
        return next(s for s in engine._channels if s.type == "tc_differential")

    def test_no_tc_value_before_the_first_cjc_read(self):
        engine = make_engine()
        assert engine._scale_channel(self._tc_spec(engine), [0.0], engine._cjc_celsius) == [None]

    def test_failed_cjc_read_is_logged_once_and_blanks_the_tc(self):
        engine = make_engine()
        engine._device.read_cjc_celsius = lambda: 21.0
        engine._poll_out_of_band()
        assert engine._cjc_celsius == 21.0

        def boom():
            raise RuntimeError("LJM timeout")
        engine._device.read_cjc_celsius = boom
        engine._poll_out_of_band()
        engine._poll_out_of_band()

        assert engine._cjc_celsius is None
        failures = [e for e in engine.event_log if "CJC read FAILED" in e]
        assert len(failures) == 1 and "LJM timeout" in failures[0]
        assert engine._scale_channel(self._tc_spec(engine), [0.0], engine._cjc_celsius) == [None]

        engine._device.read_cjc_celsius = lambda: 21.0
        engine._poll_out_of_band()
        assert engine._cjc_celsius == 21.0
        assert any("CJC read recovered" in e for e in engine.event_log)

    def test_failed_counter_read_is_logged_and_clears_counts(self):
        engine = make_engine()
        engine._counters = {"pos_lox_main": 12.0}

        def boom():
            raise RuntimeError("LJM timeout")
        engine._device.read_counters = boom
        engine._poll_out_of_band()

        assert engine._counters == {}
        assert any("counter read FAILED" in e for e in engine.event_log)
