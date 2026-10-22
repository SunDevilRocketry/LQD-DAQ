"""tests/software/test_hardware_mock.py: MockLabJack timing, data and actuators."""

import math
import time

import pytest

from daq.hardware.mock import MockLabJack
from daq.manifest import BINARY_DIO, ActuatorSpec

from tests.software._helpers import full_actuators, full_channels

# 0.4 MPa (58.015 psi) over 0-10 V; seller listing, UNVERIFIED until the label
# is read.
_DPT_FULL_SCALE_PSI = 58.015
_DPT_PSI_PER_V      = _DPT_FULL_SCALE_PSI / 10.0


class TestMockLabJack:

    def setup_method(self):
        self.channels  = full_channels()
        self.actuators = full_actuators()
        self.device = MockLabJack(self.channels, self.actuators)
        self.device.open()
        self.device.start_stream()

    def teardown_method(self):
        self.device.stop_stream()
        self.device.close()

    def test_stream_returns_every_active_streamed_channel(self):
        batch = self.device.stream_read()
        expected = {
            spec.id for spec in self.channels
            if spec.active and spec.is_streamed
        }
        assert expected.issubset(set(batch.keys()))
        assert "scan_times" in batch

    def test_stream_omits_inactive_channels(self):
        """pt7 is declared but marked inactive - it must not be sampled."""
        batch = self.device.stream_read()
        assert "pt7" not in batch

    def test_stream_omits_out_of_band_channels(self):
        """Photogate counters are polled separately, not scanned."""
        batch = self.device.stream_read()
        assert "pos_lox_main" not in batch

    def test_batch_has_correct_scan_count(self):
        batch = self.device.stream_read()
        n = len(batch["scan_times"])
        assert n == 25, f"Expected 25 scans/batch (20 batches/s), got {n}"

    def test_all_channels_have_matching_length(self):
        batch = self.device.stream_read()
        n = len(batch["scan_times"])
        for tag in self.device.sensor_tags:
            values = batch[tag]
            assert len(values) == n, f"{tag} length mismatch: {len(values)} != {n}"

    def test_scan_times_are_monotonically_increasing(self):
        batch = self.device.stream_read()
        times = batch["scan_times"]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1], f"Non-monotonic at index {i}"

    def test_pt_voltages_in_plausible_range(self):
        batch = self.device.stream_read()
        for tag in ("pt0", "pt1", "pt2"):
            for v in batch[tag]:
                assert 0.0 < v < 5.0, f"{tag} voltage {v} out of range"

    def test_tc_voltages_are_small(self):
        # TC differential voltages should be millivolt-scale
        batch = self.device.stream_read()
        for v in batch["tc0"]:
            assert -0.05 < v < 0.05, f"tc0 TC voltage {v} out of range"

    def test_lc_voltages_near_zero_at_rest(self):
        batch = self.device.stream_read()
        for tag in ("lc0", "lc1"):
            avg = sum(batch[tag]) / len(batch[tag])
            assert abs(avg) < 0.1, f"{tag} resting voltage {avg} too large"

    def test_lc_voltages_increase_during_fire(self):
        self.device.write_actuator("lox_main",  1)
        self.device.write_actuator("fuel_main", 1)
        # Allow the ramp to run past _LC_RAMP_S
        for _ in range(8):
            batch = self.device.stream_read()
        avg = sum(batch["lc0"]) / len(batch["lc0"])
        assert avg > 0.5, f"lc0 didn't ramp up during fire: avg={avg}"

    def test_scan_time_keeps_counting_after_a_restart(self):
        first = self.device.stream_read()["scan_times"][-1]
        self.device.stop_stream()
        self.device.close()
        self.device.open()
        self.device.start_stream()
        second = self.device.stream_read()["scan_times"][0]
        assert second > first, "time_s restarted after a reconnect"

    def test_cjc_near_room_temp(self):
        cjc_c = self.device.read_cjc_celsius()
        assert 22.0 < cjc_c < 28.0, f"CJC temperature {cjc_c} unexpected"

    def test_actuator_write_and_read(self):
        self.device.write_actuator("tank_depress", 1)
        assert self.device.read_actuator("tank_depress") == 1
        self.device.write_actuator("tank_depress", 0)
        assert self.device.read_actuator("tank_depress") == 0

    def test_all_safe_closes_solenoids_and_stops_all_motion(self):
        for name in ("lox_main", "fuel_main", "ignition_arm"):
            self.device.write_actuator(name, 1)
        self.device.all_safe()

        states = self.device.actuator_states()
        assert states["ignition_arm"].state == 0, "solenoid not cleared by all_safe()"
        for name, reading in states.items():
            assert reading.moving is False, f"{name} still moving after all_safe()"

    def test_all_safe_does_not_claim_the_stepper_mains_are_closed(self):
        """A cancelled burst leaves the main where it is, so it keeps state 1."""
        self.device.write_actuator("lox_main", 1)
        self.device.all_safe()

        reading = self.device.actuator_states()["lox_main"]
        assert reading.state == 1, "stepper main falsely reported closed by all_safe()"
        assert reading.moving is False, "burst was not cancelled"

    def test_unknown_actuator_raises(self):
        with pytest.raises(KeyError):
            self.device.write_actuator("nonexistent_valve", 1)

    def test_stream_read_before_start_raises(self):
        d = MockLabJack(self.channels, self.actuators)
        d.open()
        with pytest.raises(RuntimeError):
            d.stream_read()
        d.close()

    def test_batch_timing_is_approximately_correct(self):
        # One batch at 500 Hz / 25 scans = 0.05 s, like the T7
        t0 = time.perf_counter()
        self.device.stream_read()
        elapsed = time.perf_counter() - t0
        assert 0.03 < elapsed < 0.09, (
            f"Batch timing {elapsed:.3f} s out of expected 0.05 s range"
        )

    def test_tc_and_lc_samples_stay_finite(self):
        """The dashboard once saw NaN from the mock TC/LC."""
        self.device.write_actuator("lox_main", 1)
        self.device.write_actuator("fuel_main", 1)   # fire: LCs ramp
        for _ in range(10):
            batch = self.device.stream_read()
            for tag in ("tc0", "lc0", "lc1"):
                assert all(math.isfinite(v) for v in batch[tag]), tag

    def test_igniter_sense_follows_the_igniter_lines(self):
        def level():
            values = self.device.stream_read()["ign_sense"]
            return sum(values) / len(values)
        assert level() == pytest.approx(4.4, abs=0.05)      # safe, no e-match
        self.device.write_actuator("ignition_arm", 1)
        assert level() == pytest.approx(9.0, abs=0.05)      # armed
        self.device.write_actuator("ignition_fire", 1)
        assert level() == pytest.approx(2.0, abs=0.05)      # firing

    def test_unwired_actuator_is_refused_like_the_t7(self):
        spec = ActuatorSpec(id="spare", type=BINARY_DIO, dio=None)
        device = MockLabJack(self.channels, [spec])
        with pytest.raises(RuntimeError, match="not fully specified"):
            device.write_actuator("spare", 1)

    def test_close_safes_the_outputs(self):
        self.device.write_actuator("fuel_vent", 1)
        self.device.close()
        assert self.device.read_actuator("fuel_vent") == 0
        self.device.open()

    def test_sensor_tags_property(self):
        tags = self.device.sensor_tags
        assert tags == ["pt0", "pt1", "pt2", "dpt0", "dpt1", "tc0", "lc0", "lc1",
                        "ign_sense"]

    def test_noise_is_not_constant(self):
        # Two batches should not be identical (PRNG is running)
        b1 = self.device.stream_read()
        b2 = self.device.stream_read()
        assert b1["pt0"] != b2["pt0"], "pt0 values identical across batches — noise broken"


# -- Stepper Actuators ----------------------------------------

class TestMockStepper:
    """A stepper move takes time; the mock must report moving=True meanwhile."""

    def setup_method(self):
        self.device = MockLabJack(full_channels(), full_actuators())
        self.device.open()
        self.device.start_stream()

    def teardown_method(self):
        self.device.stop_stream()
        self.device.close()

    def test_stepper_is_moving_immediately_after_command(self):
        self.device.write_actuator("lox_main", 1)
        reading = self.device.actuator_states()["lox_main"]
        assert reading.state == 1
        assert reading.moving is True

    def test_stepper_stops_moving_after_burst_duration(self):
        # 400 steps at 2000 Hz = 0.2 s per the fixture manifest.
        self.device.write_actuator("lox_main", 1)
        time.sleep(0.3)
        assert self.device.actuator_states()["lox_main"].moving is False

    def test_solenoid_never_reports_moving(self):
        self.device.write_actuator("lox_purge", 1)
        assert self.device.actuator_states()["lox_purge"].moving is False

    def test_counters_advance_only_while_moving(self):
        self.device.read_counters()          # establish the time baseline
        before = self.device.read_counters()["pos_lox_main"]

        self.device.write_actuator("lox_main", 1)
        time.sleep(0.1)                      # inside the 0.2 s burst
        during = self.device.read_counters()["pos_lox_main"]
        assert during > before, "counter did not advance during a move"

        time.sleep(0.3)                      # burst finished
        self.device.read_counters()
        settled = self.device.read_counters()["pos_lox_main"]
        after_idle = self.device.read_counters()["pos_lox_main"]
        assert after_idle == settled, "counter kept advancing while idle"


class TestDifferentialPtSimulation:
    """
    The mock dP signal must stay inside the dPT's span and small next to the
    PT signal. Span 0-0.4 MPa over 0-10 V (seller listing; UNVERIFIED).
    """

    def test_orifice_dp_stays_in_a_plausible_band(self):
        device = MockLabJack(full_channels(), full_actuators())
        device.open()
        device.start_stream()
        try:
            batch = device.stream_read()
        finally:
            device.stop_stream()
            device.close()

        for tag in ("dpt0", "dpt1"):
            psi = [v * _DPT_PSI_PER_V for v in batch[tag]]
            assert all(0.0 < p < _DPT_FULL_SCALE_PSI for p in psi), (
                f"{tag} simulated outside a plausible orifice dP band: "
                f"min={min(psi):.1f} max={max(psi):.1f} psi"
            )

    def test_orifice_dp_is_well_below_tank_pressure(self):
        """Orifice dP is a fraction of tank pressure."""
        device = MockLabJack(full_channels(), full_actuators())
        device.open()
        device.start_stream()
        try:
            batch = device.stream_read()
        finally:
            device.stop_stream()
            device.close()

        assert max(batch["dpt0"]) < min(batch["pt0"])
