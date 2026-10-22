"""
tests/software/test_sse.py

/stream and the Broadcaster. TestClient buffers a whole response, so these
run a real uvicorn on a free 127.0.0.1 port, which also covers disconnects.
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
from daq.broadcast import (
    SSE_KEEPALIVE,
    Broadcaster,
    manifest_message,
    system_state_message,
)

from tests.software._helpers import make_engine, wait_for_first_batch


_FRAME_TIMEOUT_S  = 10.0
_SERVER_BOOT_S    = 10.0


def iter_messages(response):
    """
    Decoded SSE messages; use one per connection (a body iterates once).

    Raises:
        httpx.ReadTimeout: If the stream stalls, so the test fails, not hangs.
    """
    for line in response.iter_lines():
        if line.startswith("data: "):
            yield json.loads(line[len("data: "):])


def read_frames(response, count: int) -> list[dict]:
    """First `count` messages; AssertionError if the stream ends first."""
    frames = []
    for message in iter_messages(response):
        frames.append(message)
        if len(frames) >= count:
            return frames
    raise AssertionError(f"Stream ended after {len(frames)}/{count} frames")


@pytest.fixture
def sse_server():
    """
    Yields (base_url, engine) on a real uvicorn. Saves and restores api.py's
    module-level engine, which the other API tests share.
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
    """The envelope matches Dashboard's sseCommon.ts."""

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
        for key in ("streaming", "is_stale", "backlog", "sequence", "sensors",
                    "valves", "derived"):
            assert key in payload, f"missing payload key: {key}"
        assert set(payload["sequence"]) == {"active", "name", "elapsed_s", "state"}
        assert payload["sequence"]["state"] == "idle"  # fresh engine, fire never run

    def test_sensors_carry_their_unit(self):
        engine = make_engine()
        engine.start()
        try:
            wait_for_first_batch(engine)
            sensors = system_state_message(engine)["payload"]["sensors"]
        finally:
            engine.stop()
        assert sensors["pt0"]["unit"] == "Pa"
        assert sensors["tc0"]["unit"] == "celsius"

    def test_derived_lists_mass_flows_and_ratio(self):
        engine = make_engine()
        engine.start()
        try:
            wait_for_first_batch(engine)
            derived = system_state_message(engine)["payload"]["derived"]
        finally:
            engine.stop()
        assert set(derived) == {
            "lox_mass_flow_rate_kg_s", "fuel_mass_flow_rate_kg_s", "mixture_ratio",
        }

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

    def test_valves_report_normal_resting_state(self, sse_client):
        client, _ = sse_client
        with client.stream("GET", "/stream") as response:
            frames = read_frames(response, 3)
        valves = frames[-1]["payload"]["valves"]
        assert valves["tank_depress"]["normal"] == "open"
        assert valves["lox_main"]["normal"] == "closed"
        # Omitted in the manifest - reaches Dashboard as null, not defaulted.
        assert valves["ignition_arm"]["normal"] is None

    def test_client_unregisters_on_disconnect(self, sse_client):
        client, _ = sse_client
        with client.stream("GET", "/stream") as response:
            read_frames(response, 2)
        # Give the generator's finally block a moment to run.
        deadline = time.time() + 2.0
        while time.time() < deadline and api_module._broadcaster.client_count:
            time.sleep(0.05)
        assert api_module._broadcaster.client_count == 0


# -- Error-path push ------------------------------------------

class TestErrorPathBroadcast:
    """A comms failure skips _process_batch, so the error path must push a frame."""

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
        """Values stay (last known state); streaming/is_stale mark them not live."""
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
        """A client that falls behind gets the current frame, not a backlog."""
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
        """One hung client can't hold up the others."""
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


# -- Keep-alive on an idle stream ------------------------------

class TestKeepAlive:
    """With the engine stopped, only keep-alive comments follow the first two frames."""

    def test_idle_stream_sends_comment_frames(self, sse_client, monkeypatch):
        client, engine = sse_client
        monkeypatch.setattr(api_module, "_KEEPALIVE_INTERVAL_S", 0.2)
        engine.stop()   # no more batches, so nothing is ever pushed

        with client.stream("GET", "/stream") as response:
            saw_keepalive = False
            for line in response.iter_lines():
                if line.startswith(":"):
                    saw_keepalive = True
                    break
        assert saw_keepalive, "an idle stream sent no keep-alive"

    def test_keepalive_carries_no_data_field(self):
        """Clients ignore comment frames; it must not decode as a message."""
        assert SSE_KEEPALIVE.startswith(":")
        assert "data:" not in SSE_KEEPALIVE
        assert SSE_KEEPALIVE.endswith("\n\n")

    def test_state_frames_still_arrive_while_keepalives_are_armed(
        self, sse_client, monkeypatch
    ):
        """The keep-alive timeout must not swallow real frames."""
        client, _ = sse_client
        monkeypatch.setattr(api_module, "_KEEPALIVE_INTERVAL_S", 0.2)
        with client.stream("GET", "/stream") as response:
            frames = read_frames(response, 6)
        states = [f for f in frames if f["messageType"] == "system-state"]
        assert len(states) >= 4


# -- Listener cost with nobody connected -----------------------

class TestListenerCost:
    """With no clients, the stream thread must not build a state frame at all."""

    def test_no_state_message_is_built_with_nobody_connected(self, monkeypatch):
        assert api_module._broadcaster.client_count == 0, (
            "a previous test left a client registered"
        )
        prev_engine, prev_logger = api_module._engine, api_module._logger

        built = []
        monkeypatch.setattr(
            api_module,
            "system_state_message",
            lambda eng: built.append(eng) or {},
        )

        engine = make_engine()
        api_module.set_engine(engine)
        try:
            engine.start()
            time.sleep(0.5)     # several batches at ~10 Hz
        finally:
            engine.stop()
            api_module.set_engine(prev_engine, prev_logger)

        assert built == [], f"built {len(built)} frames for nobody"
