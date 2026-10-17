"""
daq/logger.py

CSV recorder. Disk writes run on their own thread so the stream never
waits; if the disk stalls, the buffer drops the oldest rows.
"""


from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, Sequence

from daq.manifest import ChannelSpec


_FLUSH_INTERVAL_S  = 0.05     # drain every 50 ms
_FSYNC_EVERY_N     = 20       # flush() (not fsync) every N drains, ~1 s
_MAX_BUFFER        = 250_000  # rows (~500 s at 500 Hz)


class Logger:
    """
    Non-blocking CSV logger. __main__.py owns open()/close(); the engine and
    API start and stop recordings.
    """

    def __init__(
        self,
        output_dir: str = ".",
        channels: Optional[Sequence[ChannelSpec]] = None,
    ) -> None:
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Columns: active streamed channels only (counters aren't per scan).
        self._channels: list[ChannelSpec] = [
            spec for spec in (channels or [])
            if spec.active and spec.is_streamed
        ]

        self._buffer: deque[list] = deque(maxlen=_MAX_BUFFER)
        self._lock   = threading.Lock()

        self._writer_thread: Optional[threading.Thread] = None
        self._running  = False
        self._recording = False

        self._csv_file:   Optional[object] = None
        self._csv_writer: Optional[csv.writer] = None
        self._current_path: str = ""
        self._flush_count  = 0

        self._rows_written  = 0
        self._rows_dropped  = 0

    # --------------------------------------------------------
    # Lifecycle Management
    # --------------------------------------------------------

    def open(self) -> None:
        """Start the writer thread."""
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
        """Stop any recording and the writer thread."""
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
        Open <prefix>_<timestamp>.csv and start recording; returns its path.

        Stops an active recording first. Prefix isn't sanitised.
        """
        if self._recording:
            self.stop_recording()

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._output_dir, f"{prefix}_{ts}.csv")

        # 128 KiB buffer; csv needs newline="".
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
        """Drain, flush, fsync and close the file."""
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
        Queue a row (no-op when not recording). A full buffer drops the
        oldest row; rows_dropped counts it but nothing reports it.
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
        """Drain the buffer every _FLUSH_INTERVAL_S until close()."""
        while self._running:
            time.sleep(_FLUSH_INTERVAL_S)
            with self._lock:
                self._drain_locked()

        with self._lock:
            self._drain_locked()

    def _drain_locked(self) -> None:
        """
        Write buffered rows. On a disk error, stop recording with a print
        only; the API and GUI aren't told.
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

    def _header(self) -> list[str]:
        """time_s, then <id>_raw_V and <id>_eng_<unit> per channel, in the same
        order as Engine._process_batch() builds rows."""
        cols = ["time_s"]
        for spec in self._channels:
            cols.append(f"{spec.id}_raw_V")
            cols.append(f"{spec.id}_eng_{spec.unit}")
        return cols
