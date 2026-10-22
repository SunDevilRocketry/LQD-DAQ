"""
daq/broadcast.py

SSE messages and fan-out. The stream thread hands frames to the event loop
via call_soon_threadsafe; each client has its own queue of one, latest frame
wins, so a slow dashboard drops stale frames and never delays the others.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Optional

from daq.manifest import (
    PT_DIRECT,
    PT_DIFFERENTIAL,
    TC_DIFFERENTIAL,
    LC_DIRECT,
    PHOTOGATE_COUNTER,
)


ORIGIN  = "LQD-DAQ"
VERSION = "1.0"

# Channel type -> Dashboard sensor type.
_SENSOR_TYPES = {
    PT_DIRECT:         "pressure",
    PT_DIFFERENTIAL:   "pressure",
    TC_DIFFERENTIAL:   "temperature",
    LC_DIRECT:         "force",
    PHOTOGATE_COUNTER: "position",
}


def envelope(message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a payload in the message envelope."""
    return {
        "origin":      ORIGIN,
        "version":     VERSION,
        "timestamp":   time.time(),
        "messageType": message_type,
        "payload":     payload,
    }


def manifest_message(engine) -> dict[str, Any]:
    """Channel and actuator inventory, sent once on connect."""
    return envelope("manifest", {
        "sensors": [
            {
                "id":     spec.id,
                "type":   _SENSOR_TYPES.get(spec.type, spec.type),
                "unit":   spec.unit,
                "active": spec.active,
            }
            for spec in engine.channel_specs
        ],
        "valves": [
            {"id": spec.id, "type": spec.type}
            for spec in engine.actuator_specs
        ],
    })


def system_state_message(engine) -> dict[str, Any]:
    """
    One system-state frame, per batch and per error tick. On an error tick
    the values are the last known ones, with streaming/is_stale false/true,
    so the operator sees where the cart was when it went quiet.
    """
    snap    = engine.snapshot
    types   = {spec.id: spec.type for spec in engine.channel_specs}
    normals = {spec.id: spec.normal for spec in engine.actuator_specs}

    sensors = {
        cid: {
            "readout": reading.value,
            "unit":    reading.unit,   # same as the manifest's, as a cross-check
            "type":    _SENSOR_TYPES.get(types.get(cid), "unknown"),
            "status":  reading.status,
        }
        for cid, reading in (snap.channels or {}).items()
    }

    valves = {
        aid: {
            "normal": normals.get(aid),
            # "open" means energised.
            "status": "open" if reading.state == 1 else "closed",
            "moving": reading.moving,
        }
        for aid, reading in (snap.actuators or {}).items()
    }

    fire = engine.sequence_status()

    return envelope("system-state", {
        "streaming": bool(snap.streaming),
        "is_stale":  engine.is_data_stale,
        "backlog":   snap.backlog,   # scans the host hasn't read yet
        "sequence": {
            "active":    bool(snap.sequence_active),
            "name":      snap.sequence_name or None,
            "elapsed_s": engine.sequence_elapsed_s,
            "state":     snap.sequence_state,  # fire lifecycle
            # Fire clock, same as the /sequence/* responses: runs on after
            # completion, freezes on pause or abort.
            "step":           fire["step"],
            "time_ms":        fire["sequence_time_ms"],
            "server_time_ms": fire["server_time_ms"],
        },
        "sensors": sensors,
        "valves":  valves,
        # Calculated: mass flows (kg/s) and O/F; null when an input is missing.
        "derived": dict(snap.derived or {}),
    })


def format_sse(message: dict[str, Any]) -> str:
    """One SSE `data:` frame."""
    return f"data: {json.dumps(message)}\n\n"


# SSE comment: clients ignore it, but it keeps idle connections alive.
SSE_KEEPALIVE = ": keep-alive\n\n"


class Broadcaster:
    """Fan-out to /stream clients; publish() runs on the stream thread."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queues: set[asyncio.Queue] = set()
        self._lock = threading.Lock()

    def register(self) -> asyncio.Queue:
        """New client queue. Call on the event loop, which it captures."""
        self._loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        with self._lock:
            self._queues.add(queue)
        return queue

    def unregister(self, queue: asyncio.Queue) -> None:
        """Drop a disconnected client's queue."""
        with self._lock:
            self._queues.discard(queue)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._queues)

    def publish(self, message: dict[str, Any]) -> None:
        """Send to every client, from any thread; no-op without a loop."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._deliver, message)
        except RuntimeError:
            # Loop closed: the last client is gone.
            self._loop = None

    def _deliver(self, message: dict[str, Any]) -> None:
        """Replace each client's queued frame with this one (event loop)."""
        with self._lock:
            queues = list(self._queues)

        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()      # drop the stale frame
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass
