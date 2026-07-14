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
