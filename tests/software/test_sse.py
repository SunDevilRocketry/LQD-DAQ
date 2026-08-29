"""
tests/software/test_sse.py

Tests for the /stream SSE endpoint and the Broadcaster behind it.

The rest of the API suite uses FastAPI's TestClient. That does not work
here: TestClient runs the ASGI app to completion and buffers the whole
body before returning (starlette/testclient.py's `portal.call(self.app,
...)`). This is an expected casualty of the persistent-connection
model.

These tests therefore run the app under a real uvicorn server on an
ephemeral port and connect over real HTTP - which also means the
disconnect path gets exercised.

No real hardware required (mock LabJack); binds to 127.0.0.1 only.
Run with: python -m pytest tests/software/test_sse.py -v
"""

import asyncio
import json
import threading
import time

import httpx
import pytest
import uvicorn

import daq.api as api_module
from daq.api import app
from daq.broadcast import Broadcaster, manifest_message, system_state_message

from tests.software._helpers import make_engine


_FRAME_TIMEOUT_S  = 10.0
_SERVER_BOOT_S    = 10.0


def iter_messages(response):
    """
    Yield decoded SSE messages off a streaming response.

    A response body can only be iterated once, so a test opens exactly one
    of these per connection and pulls from it.

    Raises:
        httpx.ReadTimeout: If the stream stalls (the client timeout, so a
                           broken endpoint fails the test rather than
                           hanging it).
    """
    for line in response.iter_lines():
        if line.startswith("data: "):
            yield json.loads(line[len("data: "):])


def read_frames(response, count: int) -> list[dict]:
    """
    Pull the first `count` decoded messages off a streaming response.

    Raises:
        AssertionError: If the stream ends before `count` frames arrive.
    """
    frames = []
    for message in iter_messages(response):
        frames.append(message)
        if len(frames) >= count:
            return frames
    raise AssertionError(f"Stream ended after {len(frames)}/{count} frames")


@pytest.fixture
def sse_server():
    """
    A running engine served by a real uvicorn instance on a free port.

    Yields (base_url, engine). api.py's engine reference is a module-level
    singleton shared with the rest of the API suite, so it is saved and
    restored around the test.
    """
    prev_engine, prev_logger = api_module._engine, api_module._logger

    engine = make_engine()
    engine.start()
    api_module.set_engine(engine)

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + _SERVER_BOOT_S
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"

    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", engine
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        engine.stop()
        api_module.set_engine(prev_engine, prev_logger)


@pytest.fixture
def sse_client(sse_server):
    """An httpx client pointed at the live server, with a read timeout."""
    base_url, engine = sse_server
    with httpx.Client(base_url=base_url, timeout=_FRAME_TIMEOUT_S) as client:
        yield client, engine


# -- Envelope & payload shape ---------------------------------

class TestEnvelope:
    """
    The outer envelope matches the convention Dashboard already consumes
    for flight (sseCommon.ts).
    """

    def test_manifest_envelope_fields(self):
        engine = make_engine()
        msg = manifest_message(engine)
        assert msg["origin"] == "LQD-DAQ"
        assert msg["version"] == "1.0"
        assert msg["messageType"] == "manifest"
        assert isinstance(msg["timestamp"], float)

    def test_manifest_lists_every_channel_and_actuator(self):
        engine = make_engine()
        payload = manifest_message(engine)["payload"]
        assert {s["id"] for s in payload["sensors"]} == {
            spec.id for spec in engine.channel_specs
        }
        assert {v["id"] for v in payload["valves"]} == {
            spec.id for spec in engine.actuator_specs
        }

    def test_manifest_maps_channel_types_to_dashboard_types(self):
        engine = make_engine()
        sensors = {s["id"]: s for s in manifest_message(engine)["payload"]["sensors"]}
        assert sensors["pt0"]["type"] == "pressure"
        assert sensors["tc0"]["type"] == "temperature"
        assert sensors["lc0"]["type"] == "force"
        assert sensors["pos_lox_main"]["type"] == "position"

    def test_valves_report_manifest_actuator_type(self):
        engine = make_engine()
        valves = {v["id"]: v for v in manifest_message(engine)["payload"]["valves"]}
        assert valves["lox_main"]["type"] == "pulse_stepper"
        assert valves["lox_purge"]["type"] == "binary_dio"

    def test_system_state_payload_shape(self):
        engine = make_engine()
        payload = system_state_message(engine)["payload"]
        for key in ("streaming", "is_stale", "sequence", "sensors", "valves", "derived"):
            assert key in payload, f"missing payload key: {key}"
        assert set(payload["sequence"]) == {"active", "name", "elapsed_s"}

    def test_derived_is_present_but_empty(self):
        """Kept for shape consistency; this cart has no orifice geometry."""
        engine = make_engine()
        assert system_state_message(engine)["payload"]["derived"] == {}

    def test_system_state_is_json_serializable(self):
        engine = make_engine()
        json.dumps(system_state_message(engine))


# -- Live endpoint --------------------------------------------

class TestStreamEndpoint:

    def test_first_frame_is_the_manifest(self, sse_client):
        client, _ = sse_client
        with client.stream("GET", "/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            first = read_frames(response, 1)[0]
        assert first["messageType"] == "manifest"

    def test_state_follows_manifest_without_waiting_for_a_batch(self, sse_client):
        client, _ = sse_client
        with client.stream("GET", "/stream") as response:
            frames = read_frames(response, 2)
        assert [f["messageType"] for f in frames] == ["manifest", "system-state"]

    def test_state_frames_keep_arriving(self, sse_client):
        """Batches are pushed as they're processed, ~10 Hz."""
        client, _ = sse_client
        with client.stream("GET", "/stream") as response:
            frames = read_frames(response, 6)
        states = [f for f in frames if f["messageType"] == "system-state"]
        assert len(states) >= 4

    def test_streamed_sensors_carry_live_readings(self, sse_client):
        client, _ = sse_client
        with client.stream("GET", "/stream") as response:
            frames = read_frames(response, 3)
        payload = frames[-1]["payload"]
        assert payload["streaming"] is True
        assert payload["sensors"]["pt0"]["readout"] is not None
        assert payload["sensors"]["pt0"]["type"] == "pressure"

    def test_valves_report_moving(self, sse_client):
        client, engine = sse_client
        engine.write_actuator("lox_main", 1)
        with client.stream("GET", "/stream") as response:
            frames = read_frames(response, 3)
        valve = frames[-1]["payload"]["valves"]["lox_main"]
        assert valve["status"] == "open"
        assert valve["moving"] is True

    def test_client_unregisters_on_disconnect(self, sse_client):
        client, _ = sse_client
        with client.stream("GET", "/stream") as response:
            read_frames(response, 2)
        # Give the generator's finally block a moment to run.
        deadline = time.time() + 2.0
        while time.time() < deadline and api_module._broadcaster.client_count:
            time.sleep(0.05)
        assert api_module._broadcaster.client_count == 0


# -- Error-path push (handoff sec 4's known gap) ---------------

class TestErrorPathBroadcast:
    """
    A comms failure skips _process_batch entirely, so w/o an explicit
    push on the error path the exact moment the cart goes quiet is the one
    moment nothing is sent - Dashboard would have to infer the outage from
    silence.
    """

    def test_outage_pushes_a_not_streaming_frame(self, sse_client):
        client, engine = sse_client

        with client.stream("GET", "/stream") as response:
            messages = iter_messages(response)
            next(messages)      # manifest
            next(messages)      # first state

            def always_fail():
                raise RuntimeError("simulated comms loss")

            engine._device.stream_read = always_fail

            saw_outage = False
            for _ in range(20):
                frame = next(messages)
                if (frame["messageType"] == "system-state"
                        and frame["payload"]["streaming"] is False):
                    saw_outage = True
                    break

        assert saw_outage, "no frame was pushed when the stream started failing"

    def test_outage_frame_keeps_last_known_readings(self, sse_client):
        """
        Values are kept, not blanked: an operator should still see the
        state the cart was in when it went quiet. `streaming`/`is_stale`
        are what say it is no longer live.
        """
        client, engine = sse_client

        with client.stream("GET", "/stream") as response:
            messages = iter_messages(response)
            next(messages)      # manifest
            next(messages)      # first state

            def always_fail():
                raise RuntimeError("simulated comms loss")

            engine._device.stream_read = always_fail

            outage = None
            for _ in range(20):
                frame = next(messages)
                if frame["payload"].get("streaming") is False:
                    outage = frame
                    break

        assert outage is not None
        assert outage["payload"]["sensors"]["pt0"]["readout"] is not None


# -- Broadcaster fan-out --------------------------------------

class TestBroadcaster:

    def test_publish_without_a_loop_is_a_noop(self):
        """Nothing connected yet - publishing must not raise."""
        Broadcaster().publish({"messageType": "system-state"})

    def test_latest_value_wins_on_a_full_queue(self):
        """
        Telemetry is not an event log. A client that falls behind should
        get current state on its next read, not a backlog.
        """
        broadcaster = Broadcaster()
        results = {}

        async def scenario():
            queue = broadcaster.register()
            broadcaster._deliver({"n": 1})
            broadcaster._deliver({"n": 2})
            broadcaster._deliver({"n": 3})
            results["depth"] = queue.qsize()
            results["value"] = await queue.get()

        asyncio.run(scenario())

        assert results["depth"] == 1
        assert results["value"] == {"n": 3}

    def test_each_client_gets_its_own_frame(self):
        broadcaster = Broadcaster()
        results = {}

        async def scenario():
            a = broadcaster.register()
            b = broadcaster.register()
            broadcaster._deliver({"n": 1})
            results["a"] = await a.get()
            results["b"] = await b.get()

        asyncio.run(scenario())

        assert results["a"] == {"n": 1}
        assert results["b"] == {"n": 1}

    def test_a_stalled_client_does_not_block_the_others(self):
        """
        Per-client queues exist precisely so one hung dashboard can't hold
        up delivery to any other.
        """
        broadcaster = Broadcaster()
        results = {}

        async def scenario():
            stalled = broadcaster.register()   # never read from
            healthy = broadcaster.register()
            for n in range(5):
                broadcaster._deliver({"n": n})
            results["healthy"] = await healthy.get()
            results["stalled_depth"] = stalled.qsize()

        asyncio.run(scenario())

        assert results["healthy"] == {"n": 4}
        assert results["stalled_depth"] == 1

    def test_unregister_stops_delivery(self):
        broadcaster = Broadcaster()
        results = {}

        async def scenario():
            queue = broadcaster.register()
            broadcaster.unregister(queue)
            broadcaster._deliver({"n": 1})
            results["depth"] = queue.qsize()

        asyncio.run(scenario())

        assert results["depth"] == 0
