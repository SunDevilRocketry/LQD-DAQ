"""
tests/software/test_hardware_mock.py

Tests for MockLabJack itself: stream timing, data shape, and actuator
control of the simulator (daq/hardware/mock.py).

No real hardware or network required.
Run with: python -m pytest tests/software/test_hardware_mock.py -v
"""

import time

import pytest

from daq.hardware.mock import MockLabJack

from tests.software._helpers import full_actuators, full_channels


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

    def test_cjc_voltage_near_room_temp(self):
        cjc_v = self.device.read_cjc()
        # 0.770 V = 25°C; allow ±0.05 V (~3°C)
        assert 0.720 < cjc_v < 0.820, f"CJC voltage {cjc_v} unexpected"

    def test_actuator_write_and_read(self):
        self.device.write_actuator("lox_vent", 1)
        assert self.device.read_actuator("lox_vent") == 1
        self.device.write_actuator("lox_vent", 0)
        assert self.device.read_actuator("lox_vent") == 0

    def test_all_safe_clears_all_actuators(self):
        for name in ("lox_main", "fuel_main", "ignition"):
            self.device.write_actuator(name, 1)
        self.device.all_safe()
        for name, reading in self.device.actuator_states().items():
            assert reading.state == 0, f"{name} not cleared by all_safe()"
            assert reading.moving is False, f"{name} still moving after all_safe()"

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
        # One batch at 500 Hz / 50 scans = 0.1 s
        t0 = time.perf_counter()
        self.device.stream_read()
        elapsed = time.perf_counter() - t0
        assert 0.08 < elapsed < 0.15, (
            f"Batch timing {elapsed:.3f} s out of expected 0.1 s range"
        )

    def test_sensor_tags_property(self):
        tags = self.device.sensor_tags
        assert tags == ["pt0", "pt1", "pt2", "tc0", "lc0", "lc1"]

    def test_noise_is_not_constant(self):
        # Two batches should not be identical (PRNG is running)
        b1 = self.device.stream_read()
        b2 = self.device.stream_read()
        assert b1["pt0"] != b2["pt0"], "pt0 values identical across batches — noise broken"


# -- Stepper Actuators ----------------------------------------

class TestMockStepper:
    """
    Stepper mains are commanded, not instant. The mock has to reproduce the
    gap between "commanded open" and "physically open" that solenoids don't
    have, because that's what the `moving` flag exists to express.
    """

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
