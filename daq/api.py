"""
daq/api.py

FastAPI microservice for the Liquids DAQ system.

Exposes the Engine state and control interfaces over HTTP to decouple the 
GUI process (SR 3.4.1) and support concurrent client connections (SR 3.3.1).

Client Polling Guideline (SR 3.3.2):
  To minimize network overhead, clients should poll on a closed-loop schedule 
  (send request, wait for response, sleep before the next request).

Endpoints:
  GET  /status          - connection and system health
  GET  /stream          - server-sent events: manifest on connect, then
                          live system-state pushed per acquisition batch
  GET  /snapshot        - latest engineering values (all sensors)
  GET  /actuators       - current actuator states
  GET  /events          - recent event log (SR 3.5 debug console)
  GET  /logger          - logger status and current file path
  POST /actuator        - send a single manual actuator command
  POST /safe            - all safe (de-energise everything)
  POST /fire            - start the fire autosequence
  POST /abort           - abort any running sequence
  POST /tare            - tare load cells
  POST /calibration     - update a sensor calibration coefficient
  POST /log/start       - start manual CSV recording
  POST /log/stop        - stop manual CSV recording

All endpoints return JSON. Errors return {"error": "description"}.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from daq.broadcast import (
    SSE_KEEPALIVE,
    Broadcaster,
    format_sse,
    manifest_message,
    system_state_message,
)

# Seconds of silence on /stream before a comment frame is sent instead.
_KEEPALIVE_INTERVAL_S = 5.0

# Engine and Logger instances injected at application startup.
_engine = None
_logger = None

_broadcaster = Broadcaster()


def set_engine(engine, logger=None) -> None:
    """
    Injects runtime engine and logger instances at startup.

    Also subscribes the SSE broadcaster to the engine's state updates, so
    /stream clients are fed straight off the acquisition thread.
    """
    global _engine, _logger
    _engine = engine
    _logger = logger

    if engine is not None:
        engine.set_state_listener(lambda: _publish_state(engine))


def _publish_state(engine) -> None:
    """
    Build and publish one state frame, unless nobody is listening.

    Runs on the acquisition thread ~10 Hz. Broadcaster.publish() already
    no-ops with no clients, so the client check comes first.
    """
    if _broadcaster.client_count == 0:
        return
    _broadcaster.publish(system_state_message(engine))


# --------------------------------------------------------
# Application Configuration
# --------------------------------------------------------

app = FastAPI(
    title="Liquids DAQ API",
    description="Ground controller data acquisition and sequencing service",
    version="1.0.0",
)

# CORS middleware for local network communication
# Restrict this if the system is ever exposed beyond the test-stand LAN.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------
# Request / Response Models
# --------------------------------------------------------

class ActuatorCommand(BaseModel):
    name:  str
    state: int   # 1 = Open, 0 = Closed


class CalibrationUpdate(BaseModel):
    """
    slope/intercept remain in the psi calibration domain (psi/V, psi)
    """
    tag:       str
    slope:     float
    intercept: float


class LogStartRequest(BaseModel):
    prefix: str = "manual_log"


# --------------------------------------------------------
# Internal Helpers
# --------------------------------------------------------

def _require_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return _engine


def _snap_to_dict(snap) -> dict[str, Any]:
    """Convert an EngineState snapshot to a JSON-serializable dictionary."""
    result: dict[str, Any] = {}
    for slot in snap.__slots__:
        result[slot] = getattr(snap, slot, None)
    return result


def _data_age_s(eng) -> Optional[float]:
    """
    Rounded seconds since the last processed batch, or None if infinite
    (no data has ever been received). JSON has no infinity literal, so
    None is the wire representation of "never received data".
    """
    age = eng.data_age_seconds
    return None if age == float("inf") else round(age, 3)

# --------------------------------------------------------
# Read Endpoints
# --------------------------------------------------------

@app.get("/status")
def get_status():
    """Returns system connectivity, active sequence status, and server time."""
    eng = _require_engine()
    snap = eng.snapshot
    return {
        "ok":                 snap.streaming,
        "using_mock":         snap.using_mock,
        "stream_hz":          snap.stream_hz,
        "sequence_active":    snap.sequence_active,
        "sequence_name":      snap.sequence_name,
        "unwired_actuators":  eng.unwired_actuators,
        "stale":              eng.is_data_stale,
        "data_age_s":         _data_age_s(eng),
        "server_time":        time.time(),
    }


@app.get("/stream")
async def get_stream():
    """
    Live telemetry as server-sent events.

    On connect the client receives a `manifest` message (the cart's full
    channel/actuator inventory) followed immediately by one `system-state`
    frame, so a dashboard can render w/o waiting for the next batch.
    After that, one `system-state` per processed acquisition batch, plus
    an SSE comment frame every few seconds whenever no state has been
    pushed in that window.

    Cadence is the natural batch rate, ~10 Hz (500 Hz / 50 scans per read),
    which already sits under Dashboard's render ceiling - no down-sampling
    is applied here.

    /snapshot is kept alongside this for a client that doesn't want a persistent
    connection.
    """
    eng = _require_engine()

    async def event_source():
        queue = _broadcaster.register()
        try:
            yield format_sse(manifest_message(eng))
            yield format_sse(system_state_message(eng))
            while True:
                try:
                    message = await asyncio.wait_for(
                        queue.get(), timeout=_KEEPALIVE_INTERVAL_S
                    )
                except asyncio.TimeoutError:
                    yield SSE_KEEPALIVE
                    continue
                yield format_sse(message)
        finally:
            _broadcaster.unregister(queue)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/snapshot")
def get_snapshot():
    """
    Latest processed sensor snapshot.

    `channels` is keyed by channel id from channels.yaml; each entry
    carries value / unit / status / last_updated. Units are whatever the
    manifest declares (pressures in Pa). Always represents the last
    completed 500 Hz batch.

    Recommended polling pattern (SR 3.3.2):
        t0 = now(); GET /snapshot; sleep(max(0, 1/display_hz - (now()-t0)))

    Raises:
        500: If the underlying data is older than the staleness threshold
             e.g. hardware disconnected and a reconnect is in progress.

	     Prevents a dashboard from displaying frozen readings as if 
	     they were live. Poll /status first if you want to distinguish
             "stale" from "engine not initialised" ahead of time.
    """
    eng = _require_engine()
    if eng.is_data_stale:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Snapshot data is stale ({eng.data_age_seconds:.2f}s old, "
                f"threshold {eng.DATA_STALE_THRESHOLD_S}s) - hardware may be "
                f"disconnected. Check /status."
            ),
        )
    result = _snap_to_dict(eng.snapshot)
    result["data_age_s"] = _data_age_s(eng)
    return result


@app.get("/actuators")
def get_actuators():
    """
    Current state of every system actuator.

    Each entry is {"state": 1|0, "moving": bool}. `moving` is only ever
    true for stepper-driven valves, where a commanded move takes real time
    to complete; for a solenoid the DIO write is the state.
    """
    eng = _require_engine()
    return eng.snapshot.actuators or {}


@app.get("/events")
def get_events(last_n: int = 100):
    """
    Recent event log entries for the debug console (SR 3.5).

    Args:
        last_n: Maximum number of entries to return (default 100, max 500).
    """
    eng = _require_engine()
    events = eng.event_log
    clamped_n = max(1, min(last_n, 500))
    return {"events": events[-clamped_n:]}


@app.get("/logger")
def get_logger_status():
    """Returns current logger activity, active filepath, and write metrics."""
    if _logger is None:
        return {"recording": False, "path": None, "rows_written": 0, "buffer_depth": 0}
    return {
        "recording":    _logger.is_recording,
        "path":         _logger.current_path,
        "rows_written": _logger.rows_written,
        "buffer_depth": _logger.buffer_depth,
    }


# --------------------------------------------------------
# Control Endpoints
# --------------------------------------------------------

@app.post("/actuator")
def post_actuator(cmd: ActuatorCommand):
    """
    Commands a single actuator state if no autosequence is active.

    Blocked while a sequence is active (returns 409 Conflict).
    The GUI should reflect the blocked state immediately per SR 3.2.4.

    Body: {"name": "LOx Main", "state": 1}
    """
    eng = _require_engine()
    if cmd.state not in (0, 1):
        raise HTTPException(status_code=422, detail="state must be 0 or 1")
    try:
        eng.write_actuator(cmd.name, cmd.state)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown actuator: {exc}")
    return {"ok": True, "name": cmd.name, "state": cmd.state}


@app.post("/safe")
def post_safe():
    """Immediately aborts active sequences and de-energizes all hardware outputs."""
    eng = _require_engine()
    eng.all_safe()
    return {"ok": True}


@app.post("/fire")
def post_fire():
    """
    Triggers the fire autosequence if the system is idle.
    Returns 409 if a sequence is already running.
    """
    eng = _require_engine()
    snap = eng.snapshot
    if snap.sequence_active:
        raise HTTPException(
            status_code=409,
            detail=f"Sequence '{snap.sequence_name}' already running"
        )
    eng.fire()
    return {"ok": True, "sequence": "fire"}


@app.post("/abort")
def post_abort():
    """Aborts active sequences and triggers the safe shutdown sequence."""
    eng = _require_engine()
    eng.abort()
    return {"ok": True}


@app.post("/tare")
def post_tare():
    """Captures load cell baseline readings as tare offsets."""
    eng = _require_engine()
    eng.tare()
    return {"ok": True}


@app.post("/calibration")
def post_calibration(update: CalibrationUpdate):
    """
    Update a sensor calibration coefficient at runtime.

    Changes take effect on the next processed scan batch.
    Does not persist to calibration.json - call /calibration/save for that.

    Body: {"tag": "pt0", "slope": 128.0, "intercept": -62.8}
    """
    eng = _require_engine()
    eng.update_calibration(update.tag, update.slope, update.intercept)
    return {"ok": True, "tag": update.tag}


@app.post("/calibration/save")
def post_calibration_save():
    """Saves current active calibrations to calibration.json."""
    eng = _require_engine()
    try:
        eng.save_calibration()
    except AttributeError:
        raise HTTPException(
            status_code=501,
            detail="Engine does not have save_calibration - no cal_path set"
        )
    return {"ok": True}


@app.post("/log/start")
def post_log_start(req: LogStartRequest):
    """
    Starts a manual CSV logging session if the logger is idle.
    Returns 409 if a recording is already active.
    """
    if _logger is None:
        raise HTTPException(status_code=503, detail="Logger not configured")
    if _logger.is_recording:
        raise HTTPException(
            status_code=409,
            detail=f"Already recording: {_logger.current_path}"
        )
    path = _logger.start_recording(prefix=req.prefix)
    return {"ok": True, "path": path}


@app.post("/log/stop")
def post_log_stop():
    """Stops the active manual CSV logging session and flushes the buffer."""
    if _logger is None:
        raise HTTPException(status_code=503, detail="Logger not configured")
    if not _logger.is_recording:
        return {"ok": True, "message": "Not recording"}
    _logger.stop_recording()
    return {"ok": True, "rows_written": _logger.rows_written}
