"""
daq/api.py

HTTP API over the Engine (FastAPI).

  GET  /status            health, sequence state, unwired actuators
  GET  /stream            SSE: manifest on connect, then system-state per batch
  GET  /snapshot          latest values (500 if stale)
  GET  /actuators         actuator states
  GET  /events            event log
  GET  /logger            recording status
  GET  /sequence          fire.yaml phases
  POST /actuator          manual command (409 mid-sequence)
  POST /deenergize        halt any sequence, de-energise everything (/safe: old name)
  POST /fire              fire.yaml from T=0
  POST /abort             abort.yaml, or all-off if it can't run
  POST /sequence/start    resume fire
  POST /sequence/stop     pause fire in place
  POST /sequence/time     seek the fire clock (forward-only while running)
  POST /sequence/step     seek to a named step
  POST /tare              tare load cells
  POST /calibration       set a slope/intercept in memory
  POST /calibration/save  write calibration.json
  POST /log/start         start a CSV recording
  POST /log/stop          stop it

Errors are FastAPI's {"detail": ...}. /actuator, /fire and /sequence/*
return 423 {"detail": "busy"} if the control lock is busy; /deenergize and
/abort never take it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from daq.engine import SequenceRefused
from daq.broadcast import (
    SSE_KEEPALIVE,
    Broadcaster,
    format_sse,
    manifest_message,
    system_state_message,
)

# Seconds of /stream silence before a keep-alive comment.
_KEEPALIVE_INTERVAL_S = 5.0

# Set by set_engine() at startup.
_engine = None
_logger = None

_broadcaster = Broadcaster()


# Serialises control routes. Always taken before Engine._lock (lock order).
_control_lock = threading.Lock()


class _ControlLockBusy(Exception):
    """Non-blocking acquire of _control_lock failed (-> 423)."""


@contextmanager
def _control_guard():
    """Hold _control_lock; raises _ControlLockBusy if it's taken."""
    if not _control_lock.acquire(blocking=False):
        raise _ControlLockBusy()
    try:
        yield
    finally:
        _control_lock.release()


def _busy() -> JSONResponse:
    """423 for a busy control lock."""
    return JSONResponse(status_code=423, content={"detail": "busy"})


def set_engine(engine, logger=None) -> None:
    """Inject the engine and logger, and feed /stream from engine updates."""
    global _engine, _logger
    _engine = engine
    _logger = logger

    if engine is not None:
        engine.set_state_listener(lambda: _publish_state(engine))


def _publish_state(engine) -> None:
    """Publish one state frame; skipped with no clients (runs on the stream thread)."""
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

# Open CORS: test-stand LAN only.
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
    state: int   # 1 = energise, 0 = de-energise


class CalibrationUpdate(BaseModel):
    """slope/intercept in the psi or lbf domain (psi/V, lbf/V)."""
    tag:       str
    slope:     float
    intercept: float


class LogStartRequest(BaseModel):
    prefix: str = "manual_log"


class SequenceTimeRequest(BaseModel):
    seconds: float


class SequenceStepRequest(BaseModel):
    name: str


# --------------------------------------------------------
# Internal Helpers
# --------------------------------------------------------

def _require_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return _engine


def _snap_to_dict(snap) -> dict[str, Any]:
    """EngineState -> dict for JSON."""
    result: dict[str, Any] = {}
    for slot in snap.__slots__:
        result[slot] = getattr(snap, slot, None)
    return result


def _data_age_s(eng) -> Optional[float]:
    """Data age in s, or None if no batch yet (JSON has no infinity)."""
    age = eng.data_age_seconds
    return None if age == float("inf") else round(age, 3)

# --------------------------------------------------------
# Read Endpoints
# --------------------------------------------------------

@app.get("/status")
def get_status():
    """Health and sequence state. Lags a transition by up to one batch."""
    eng = _require_engine()
    snap = eng.snapshot
    return {
        "ok":                 snap.streaming,
        "using_mock":         snap.using_mock,
        "stream_hz":          snap.stream_hz,
        "backlog":            snap.backlog,
        "sequence_active":    snap.sequence_active,
        "sequence_name":      snap.sequence_name or None,
        "sequence_step":      snap.sequence_step,
        "sequence_state":     snap.sequence_state,
        "unwired_actuators":  eng.unwired_actuators,
        "stale":              eng.is_data_stale,
        "data_age_s":         _data_age_s(eng),
        "server_time":        time.time(),
    }


@app.get("/stream")
async def get_stream():
    """
    SSE: manifest + one state frame on connect, then a frame per batch
    (~20 Hz on the T7, 10 Hz mock), keep-alive comments when idle.
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
    Latest snapshot, channels keyed by manifest id in manifest units.

    Raises:
        500: Data is stale (e.g. reconnecting), so frozen values never
             look live.
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
    """{id: {"state": 1|0, "moving": bool}}; moving is stepper-only."""
    eng = _require_engine()
    return eng.snapshot.actuators or {}


@app.get("/events")
def get_events(last_n: int = 100):
    """Last `last_n` event log entries (1-500, SR 3.5)."""
    eng = _require_engine()
    events = eng.event_log
    clamped_n = max(1, min(last_n, 500))
    return {"events": events[-clamped_n:]}


@app.get("/logger")
def get_logger_status():
    """Recording flag, file path, row counts, buffer depth, last disk error."""
    if _logger is None:
        return {"recording": False, "path": None, "rows_written": 0,
                "rows_dropped": 0, "buffer_depth": 0, "error": None}
    return {
        "recording":    _logger.is_recording,
        "path":         _logger.current_path,
        "rows_written": _logger.rows_written,
        "rows_dropped": _logger.rows_dropped,
        "buffer_depth": _logger.buffer_depth,
        "error":        _logger.last_error,
    }


@app.get("/sequence")
def get_sequence():
    """fire.yaml phases (409 if it can't be parsed)."""
    eng = _require_engine()
    try:
        return {"steps": eng.dump_fire_sequence()}
    except SequenceRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# --------------------------------------------------------
# Control Endpoints
# --------------------------------------------------------

@app.post("/actuator")
def post_actuator(cmd: ActuatorCommand):
    """
    Body {"name": id, "state": 0|1}. 409 mid-sequence, unwired or stepper
    moving; 422 unknown id; 423 if the control lock is busy.
    """
    eng = _require_engine()
    if cmd.state not in (0, 1):
        raise HTTPException(status_code=422, detail="state must be 0 or 1")
    try:
        with _control_guard():
            eng.write_actuator(cmd.name, cmd.state)
    except _ControlLockBusy:
        return _busy()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown actuator: {exc}")
    return {"ok": True, "name": cmd.name, "state": cmd.state}


@app.post("/deenergize")
def post_deenergize():
    """Halt any sequence and de-energise every output. Never waits for the lock."""
    _require_engine().deenergize()
    return {"ok": True}


@app.post("/safe")
def post_safe():
    """Old name for /deenergize; kept for one release."""
    return post_deenergize()


@app.post("/fire")
def post_fire():
    """
    Start fire.yaml. 409 if a sequence is running or fire.yaml is refused,
    423 if the control lock is busy.
    """
    eng = _require_engine()
    try:
        with _control_guard():
            eng.fire()
    except _ControlLockBusy:
        return _busy()
    except (SequenceRefused, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "sequence": "fire"}


@app.post("/abort")
def post_abort():
    """
    abort.yaml, or all-off if it can't run; /events says which. Always 200.
    Never waits for the control lock.
    """
    _require_engine().abort()
    return {"ok": True}


@app.post("/sequence/start")
def post_sequence_start():
    """Resume fire from its position. 409 after an abort (use /fire), 423 if busy."""
    eng = _require_engine()
    try:
        with _control_guard():
            return eng.start_sequence()
    except _ControlLockBusy:
        return _busy()
    except (SequenceRefused, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/sequence/stop")
def post_sequence_stop():
    """Pause fire: igniter off, other outputs and recording kept. 423 if busy."""
    eng = _require_engine()
    try:
        with _control_guard():
            return eng.stop_sequence()
    except _ControlLockBusy:
        return _busy()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/sequence/time")
def post_sequence_time(req: SequenceTimeRequest):
    """
    Seek the fire clock. Stopped: no hardware effect. Running: forward only
    (skipped actions fire; 409 backward). 423 if busy.
    """
    eng = _require_engine()
    try:
        with _control_guard():
            return eng.set_sequence_time(req.seconds)
    except _ControlLockBusy:
        return _busy()
    except (SequenceRefused, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/sequence/step")
def post_sequence_step(req: SequenceStepRequest):
    """/sequence/time to a named step's start. 422 unknown step, 423 if busy."""
    eng = _require_engine()
    try:
        with _control_guard():
            return eng.jump_to_step(req.name)
    except _ControlLockBusy:
        return _busy()
    except (SequenceRefused, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown step: {exc}")


@app.post("/tare")
def post_tare():
    """Tare load cells on their last batch's mean; 409 during a fire."""
    eng = _require_engine()
    try:
        eng.tare()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True}


@app.post("/calibration")
def post_calibration(update: CalibrationUpdate):
    """
    Set a slope/intercept from the next batch on (/calibration/save persists
    it). 422 for a tag that isn't a calibrated channel.
    """
    eng = _require_engine()
    try:
        eng.update_calibration(update.tag, update.slope, update.intercept)
    except KeyError:
        raise HTTPException(
            status_code=422, detail=f"Unknown calibration tag: {update.tag}"
        )
    return {"ok": True, "tag": update.tag}


@app.post("/calibration/save")
def post_calibration_save():
    """Update calibration.json's PT/LC entries. 501 without a path."""
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
    """Start a CSV recording; 409 if one is running, 422 for a bad prefix."""
    if _logger is None:
        raise HTTPException(status_code=503, detail="Logger not configured")
    if _logger.is_recording:
        raise HTTPException(
            status_code=409,
            detail=f"Already recording: {_logger.current_path}"
        )
    try:
        path = _logger.start_recording(prefix=req.prefix)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True, "path": path}


@app.post("/log/stop")
def post_log_stop():
    """Stop and flush the recording."""
    if _logger is None:
        raise HTTPException(status_code=503, detail="Logger not configured")
    if not _logger.is_recording:
        return {"ok": True, "message": "Not recording"}
    _logger.stop_recording()
    return {"ok": True, "rows_written": _logger.rows_written}
