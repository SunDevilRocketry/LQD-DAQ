"""
daq/logger.py

Non-blocking background CSV logging engine for real-time telemetry.

Design Constraints:
  - Disk I/O runs entirely on a dedicated writer thread to prevent stream stalling.
  - Thread-safe queue (deque) drops the oldest rows if disk latency spikes.
"""


from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional


# Buffer polling and disk flushing intervals
_FLUSH_INTERVAL_S  = 0.05     # Thread sleep duration (20 Hz drain rate)
_FSYNC_EVERY_N     = 20       # Frequency of file system syncs (~1 second interval)
_MAX_BUFFER        = 250_000  # Thread-safe ring buffer maximum capacity


class Logger:
    """
    Non-blocking CSV logger.

    Usage:
        logger = Logger(output_dir="data")
        logger.open()                          # starts writer thread

        logger.start_recording("hotfire")      # opens CSV file
        for row in rows:
            logger.write_row(row)              # non-blocking
        logger.stop_recording()                # flush + close file

        logger.close()                         # stops writer thread

    Lifecycle contract:
        open() and close() are managed by the application owner (__main__.py).
        start_recording() and stop_recording() are driven dynamically by the engine.
    """

    def __init__(self, output_dir: str = ".") -> None:
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._buffer: deque[list] = deque(maxlen=_MAX_BUFFER)
        self._lock   = threading.Lock()

        self._writer_thread: Optional[threading.Thread] = None
        self._running  = False
        self._recording = False

        self._csv_file:   Optional[object] = None
        self._csv_writer: Optional[csv.writer] = None
        self._current_path: str = ""
        self._flush_count  = 0

        # Operational metrics
        self._rows_written  = 0
        self._rows_dropped  = 0

    # --------------------------------------------------------
    # Lifecycle Management
    # --------------------------------------------------------

    def open(self) -> None:
        """Start the background writer thread. Call once at startup."""
        if self._running:
            return
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="daq-logger",
            daemon=True,
        )
        self._writer_thread.start()

    def close(self) -> None:
        """Flushes the remaining queue and stops the background writer thread."""
        if self._recording:
            self.stop_recording()
        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)

    # --------------------------------------------------------
    # Recording Control
    # --------------------------------------------------------

    def start_recording(self, prefix: str = "data") -> str:
        """
        Initializes a new CSV file and starts logging telemetry.

        If a recording is active, it is stopped and flushed first.

        Args:
            prefix: Filename prefix (e.g. "hotfire", "manual_log").

        Returns:
            Full path to the opened CSV file.
        """
        if self._recording:
            self.stop_recording()

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._output_dir, f"{prefix}_{ts}.csv")

        # Open with large write buffer; newline="" required for csv module
        fh = open(path, "w", newline="", buffering=1 << 17)
        writer = csv.writer(fh)
        writer.writerow(self._header())

        with self._lock:
            self._buffer.clear()
            self._csv_file    = fh
            self._csv_writer  = writer
            self._current_path = path
            self._recording   = True
            self._flush_count = 0
            self._rows_written = 0
            self._rows_dropped = 0

        print(f"[LOGGER] Recording: {path}")
        return path

    def stop_recording(self) -> None:
        """Flushes all remaining buffered rows, fsyncs, and closes the file."""
        with self._lock:
            self._recording = False

        time.sleep(_FLUSH_INTERVAL_S * 2)

        with self._lock:
            self._drain_locked()
            if self._csv_file:
                try:
                    self._csv_file.flush()
                    os.fsync(self._csv_file.fileno())
                    self._csv_file.close()
                except OSError:
                    pass
                self._csv_file   = None
                self._csv_writer = None

        print(f"[LOGGER] Stopped. {self._rows_written} rows written to "
              f"{self._current_path}")

    # --------------------------------------------------------
    # Hot Path (High-Frequency Input)
    # --------------------------------------------------------

    def write_row(self, row: list) -> None:
        """
        Enqueues a data row for non-blocking background writing.

        If the buffer is full (disk stall), the oldest row is silently
        dropped (deque maxlen handles this automatically).

        Args:
            row: List of values matching the CSV column order.
        """
        if not self._recording:
            return

        if len(self._buffer) >= _MAX_BUFFER:
            self._rows_dropped += 1

        self._buffer.append(row)

    # --------------------------------------------------------
    # Status & Metrics API
    # --------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_path(self) -> str:
        return self._current_path

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def rows_dropped(self) -> int:
        return self._rows_dropped

    @property
    def buffer_depth(self) -> int:
        return len(self._buffer)

    # --------------------------------------------------------
    # Writer thread
    # --------------------------------------------------------

    def _writer_loop(self) -> None:
        """
        Background thread: drains the buffer to disk at _FLUSH_INTERVAL_S.
        Runs independently of the stream thread.
        """
        while self._running:
            time.sleep(_FLUSH_INTERVAL_S)
            with self._lock:
                self._drain_locked()

        # Final drain after stop()
        with self._lock:
            self._drain_locked()

    def _drain_locked(self) -> None:
        """Drains the queue and writes buffered rows to the active file handle."""
        if not self._csv_writer or not self._buffer:
            return

        batch = []
        while self._buffer:
            try:
                batch.append(self._buffer.popleft())
            except IndexError:
                break

        if not batch:
            return

        try:
            for row in batch:
                self._csv_writer.writerow(row)
            self._rows_written += len(batch)
            self._flush_count  += 1

            if self._flush_count % _FSYNC_EVERY_N == 0:
                self._csv_file.flush()
        except OSError as exc:
            print(f"[LOGGER] Write error: {exc} - closing file handle")
            try:
                self._csv_file.close()
            except OSError:
                pass
            self._csv_file   = None
            self._csv_writer = None
            self._recording  = False

    # --------------------------------------------------------
    # CSV Schema Definition
    # --------------------------------------------------------

    @staticmethod
    def _header() -> list[str]:
        """Returns the list of column headers matching the telemetry row schema."""
        cols = ["time_s"]

        pt_tags = ["POT", "PFT", "POI", "PFI", "PFO", "PC", "PNS", "PNP"]
        tc_tags = ["TOI", "TFI", "TFO"]
        lc_tags = ["LC_1", "LC_2"]

        for tag in pt_tags:
            cols.append(f"{tag}_raw_V")
            cols.append(f"{tag}_eng_Pa")
        for tag in tc_tags:
            cols.append(f"{tag}_raw_V")
            cols.append(f"{tag}_eng_C")
        for tag in lc_tags:
            cols.append(f"{tag}_raw_V")
            cols.append(f"{tag}_eng_lbf")

        cols.append("impulse_ns")
        cols.append("lox_mdot_kg_s")
        cols.append("fuel_mdot_kg_s")

        return cols
