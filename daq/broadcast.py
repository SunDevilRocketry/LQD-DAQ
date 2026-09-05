"""
daq/broadcast.py

Server-sent-events plumbing for the legacy cart: the thread -> asyncio
hand-off, per-client fan-out, and the message envelopes themselves.

Threading model:
    engine.py's stream loop is a plain threading.Thread; the /stream
    endpoint is asyncio. The bridge is one asyncio.Queue(maxsize=1) per
    connected client, written to via loop.call_soon_threadsafe.

    maxsize=1 with drop-and-replace on put is deliberate. (Desire for current
    state), client falling behind should not need stale frames.

    Fan-out is per-client queues rather than one shared queue, so one
    slow dashboard can never delay delivery to any other. In-process
    only, no external broker - the audience is a handful of dashboards
    during a hotfire.
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

# Manifest channel type -> the sensor `type` Dashboard renders by.
_SENSOR_TYPES = {
    PT_DIRECT:         "pressure",
    PT_DIFFERENTIAL:   "pressure",
    TC_DIFFERENTIAL:   "temperature",
    LC_DIRECT:         "force",
    PHOTOGATE_COUNTER: "position",
}


def envelope(message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Wraps a payload in the standard outer envelope."""
    return {
        "origin":      ORIGIN,
        "version":     VERSION,
        "timestamp":   time.time(),
        "messageType": message_type,
        "payload":     payload,
    }


def manifest_message(engine) -> dict[str, Any]:
    """
    Full channel/actuator inventory, pushed once on connect.

    Lets Dashboard render generically off whatever this cart actually has.
    """
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
    One frame of live system state.

    Pushed once per processed batch (~10 Hz, the natural cadence of
    500 Hz / 50 scans-per-read) and again on an error tick when a comms
    failure means no batch arrived. On that error path `streaming` is
    false and the sensor values are the last known ones: they are kept
    rather than blanked so an operator still sees the state the cart was
    in when it went quiet, with `streaming`/`is_stale` saying plainly that
    it is no longer live.

    `status` is NOMINAL | CAUTION | WARNING per Dashboard's readingStatus.ts,
    plus UNCONFIGURED for a channel with no threshold bands configured yet.

    `derived` carries computed quantities (mass flow, mixture ratio) rather
    than measured ones, so its entries have values but no status band.
    """
    snap    = engine.snapshot
    types   = {spec.id: spec.type for spec in engine.channel_specs}
    normals = {spec.id: spec.normal for spec in engine.actuator_specs}

    sensors = {
        cid: {
            "readout": reading.value,
            "type":    _SENSOR_TYPES.get(types.get(cid), "unknown"),
            "status":  reading.status,
        }
        for cid, reading in (snap.channels or {}).items()
    }

    valves = {
        aid: {
            # Resting state, straight from actuators.yaml. null until the
            # cart's P&ID confirms it.
            "normal": normals.get(aid),
            "status": "open" if reading.state == 1 else "closed",
            "moving": reading.moving,
        }
        for aid, reading in (snap.actuators or {}).items()
    }

    return envelope("system-state", {
        "streaming": bool(snap.streaming),
        "is_stale":  engine.is_data_stale,
        "sequence": {
            "active":    bool(snap.sequence_active),
            "name":      snap.sequence_name or None,
            "elapsed_s": engine.sequence_elapsed_s,
        },
        "sensors": sensors,
        "valves":  valves,
        # Injector mass flows in kg/s and mixture ratio (O/F, dimensionless),
        # computed per batch from the differential PTs. SI on the wire, as
        # pressures are. An entry is null when an input is missing - most
        # often the LOX inlet temperature, which the density lookup needs.
        "derived": dict(snap.derived or {}),
    })


def format_sse(message: dict[str, Any]) -> str:
    """Renders one message as an SSE `data:` frame."""
    return f"data: {json.dumps(message)}\n\n"


# An SSE comment: carries no data and every client ignores it, but it is
# traffic, which is what keeps an idle connection from being dropped by an
# intervening proxy or a browser timeout.
SSE_KEEPALIVE = ": keep-alive\n\n"


class Broadcaster:
    """
    Fans state messages out to every connected /stream client.

    Lives in the API process alongside the engine. `publish()` is called
    from the acquisition thread; everything else runs on the event loop.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queues: set[asyncio.Queue] = set()
        self._lock = threading.Lock()

    def register(self) -> asyncio.Queue:
        """
        Creates a queue for one newly connected client.

        Must be called from the event loop thread: the loop reference the
        publisher needs is captured here, which also means a fresh loop
        (a test client's, say) rebinds correctly on the next connect.
        """
        self._loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        with self._lock:
            self._queues.add(queue)
        return queue

    def unregister(self, queue: asyncio.Queue) -> None:
        """Drops a disconnected client's queue."""
        with self._lock:
            self._queues.discard(queue)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._queues)

    def publish(self, message: dict[str, Any]) -> None:
        """
        Hands a message to every connected client. Safe to call from any
        thread; a no-op when nobody is connected or the loop has gone away.
        """
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._deliver, message)
        except RuntimeError:
            # Loop already closed - the last client is gone.
            self._loop = None

    def _deliver(self, message: dict[str, Any]) -> None:
        """Latest-value-wins delivery. Runs on the event loop."""
        with self._lock:
            queues = list(self._queues)

        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()      # discard the superseded frame
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass
