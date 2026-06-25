from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional


# Drain the write buffer every this many seconds
_FLUSH_INTERVAL_S  = 0.05    # 20 Hz drain rate
# fsync every N drain cycles (~1 s)
_FSYNC_EVERY_N     = 20
# Maximum buffered rows before we start dropping
_MAX_BUFFER        = 250_000


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
    """

    def __init__(self, output_dir: str = ".") -> None:
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._buffer: deque[list] = deque(maxlen=_MAX_BUFFER)
        self._lock   = threading.Lock()

        self._writer_thread: Optional[threading.Thread] = None
        self._running  = False
        self._recording = False

        self._csv_file:   Optional[object] = None   # file handle
        self._csv_writer: Optional[csv.writer] = None
        self._current_path: str = ""
        self._flush_count  = 0

        # Stats exposed to API
        self._rows_written  = 0
        self._rows_dropped  = 0

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def open(self) -> None:
        """Start the background writer thread. Call once at startup."""
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="daq-logger",
            daemon=True,
        )
        self._writer_thread.start()

    def close(self) -> None:
        """
        Stop the writer thread, flushing any remaining buffered rows first.
        Blocks until the writer exits (max ~1 s).
        """
        if self._recording:
            self.stop_recording()
        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)

    # --------------------------------------------------------
    # Recording control (called by engine.py)
    # --------------------------------------------------------

    def start_recording(self, prefix: str = "data") -> str:
        """
        Open a new CSV file and begin recording.

        If a recording is already active it is stopped first.

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
            # Clear any stale/leftover data from previous sessions
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
        """Flush remaining buffer rows, fsync, and close the current file."""
        with self._lock:
            self._recording = False

        # Give the writer thread one drain cycle to flush the buffer
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
    # Hot path (called from stream thread at 500 Hz)
    # --------------------------------------------------------

    def write_row(self, row: list) -> None:
        """
        Enqueue a row for writing. Returns immediately - never blocks.

        If the buffer is full (disk stall), the oldest row is silently
        dropped.
        
        Args:
            row: List of values matching the CSV column order.
        """
        if not self._recording:
            return
        
        # Track dropped rows before append (atomic operations in CPython)
        if len(self._buffer) >= _MAX_BUFFER:
            self._rows_dropped += 1
            
        self._buffer.append(row)

    # --------------------------------------------------------
    # Status (read by api.py)
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
        """
        Write all buffered rows to the CSV file.
        Must be called with self._lock held.
        """
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
            print(f"[LOGGER] Write error: {exc}")

    # --------------------------------------------------------
    # CSV header
    # --------------------------------------------------------

    @staticmethod
    def _header() -> list[str]:
        """
        Column headers matching the row layout produced by engine.py.
        Order must stay in sync with _process_batch in engine.py.
        """
        cols = ["time_s"]

        pt_tags = ["POT", "PFT", "POI", "PFI", "PFO", "PC", "PNS", "PNP"]
        tc_tags = ["TOI", "TFI", "TFO"]
        lc_tags = ["LC_1", "LC_2"]

        for tag in pt_tags + tc_tags + lc_tags:
            cols.append(f"{tag}_raw_V")
            cols.append(f"{tag}_eng")

        cols.append("impulse_ns")
        cols.append("lox_mdot_kg_s")
        cols.append("fuel_mdot_kg_s")

        return cols