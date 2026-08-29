"""
tests/software/test_logger.py

Tests for the non-blocking CSV logger (daq/logger.py).

No real hardware or network required.
Run with: python -m pytest tests/software/test_logger.py -v
"""

import csv
import os
import tempfile
import time

from daq.logger import Logger

from tests.software._helpers import full_channels


class TestLogger:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.logger = Logger(output_dir=self.tmp, channels=full_channels())
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
        path = self.logger.start_recording("test")
        self.logger.write_row(["0.001", "1.0", "150.0"])
        self.logger.stop_recording()
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "time_s" in header
        # CSV schema follows channels.yaml, so column names are channel ids
        # and their declared units.
        assert "pt0_raw_V"  in header
        assert "pt0_eng_Pa" in header
        assert "tc0_eng_celsius" in header
        assert "lc0_eng_lbf"     in header

    def test_header_excludes_inactive_and_out_of_band_channels(self):
        path = self.logger.start_recording("test")
        self.logger.stop_recording()
        with open(path) as f:
            header = next(csv.reader(f))
        # pt7 is inactive; photogate counters aren't sampled per scan.
        assert not any(c.startswith("pt7_") for c in header)
        assert not any(c.startswith("pos_") for c in header)

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
