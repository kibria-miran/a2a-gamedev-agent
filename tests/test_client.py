"""Client tests (section 8.8 + R20 fetch routes, E-70)."""

import hashlib

import httpx
import pytest
from tests.test_server import (
    COUNTERED,
    CRITIC,
    DESIGNER,
    ScriptedResponder,
    counter_decision,
    fake_build_result,
)

from src import client as client_mod
from src import config, decision, server
from src import store as store_mod
from src.transport import (
    message_dict,
    offer_parts,
    spec_signed_parts,
)

OFFER = {
    "genre": "invaders",
    "grid_w": 40,
    "grid_h": 20,
    "tick_ms": 100,
    "enemy_count": 8,
    "win_condition": "survive",
}


@pytest.fixture
def store(tmp_path):
    s = store_mod.Store(str(tmp_path / "coder.db"))
    yield s
    s.close()


def asgi_client(app, token=DESIGNER, *, headers=None):
    merged = {"A2A-Version": config.A2A_PROTOCOL_VERSION}
    if headers:
        merged.update(headers)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers=merged,
        timeout=10,
    )


async def test_fetch_card(store):
    app = server.create_app(store, responder=ScriptedResponder(),
                            designer_token=DESIGNER, critic_token=CRITIC)
    http_client = asgi_client(app, token=None)
    async with http_client:
        card = await client_mod.A2AClient(
            "http://testserver", token=None, http_client=http_client
        ).fetch_card()
    assert card["name"] == "A2A Game Dev Coder"
    assert card["skills"][0]["id"] == "build_game"


async def test_send_message_returns_task(store):
    responder = ScriptedResponder(counter_decision(round_number=1))
    app = server.create_app(store, responder=responder,
                            designer_token=DESIGNER, critic_token=CRITIC)
    http_client = asgi_client(app, token=DESIGNER)
    msg = message_dict(role="user", parts=offer_parts(0, OFFER))
    async with http_client:
        a2a = client_mod.A2AClient("http://testserver", token=DESIGNER, http_client=http_client)
        task = await a2a.send_message(msg)
    assert task["id"]
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"


async def test_full_lifecycle_via_client_fetch_build(store):
    build_result = fake_build_result()
    responder = ScriptedResponder(
        counter_decision(round_number=1),
        decision.Decision(kind="build", round=3, build_result=build_result, source="code"),
    )
    app = server.create_app(store, responder=responder,
                            designer_token=DESIGNER, critic_token=CRITIC)

    # designer side: offer -> counter -> spec_signed -> build
    designer_http = asgi_client(app, token=DESIGNER)
    async with designer_http:
        designer = client_mod.A2AClient("http://testserver", token=DESIGNER, http_client=designer_http)
        offer = message_dict(role="user", parts=offer_parts(0, OFFER))
        task = await designer.send_message(offer)
        task_id = task["id"]
        signed = message_dict(role="user", parts=spec_signed_parts(2, COUNTERED), task_id=task_id)
        task = await designer.send_message(signed)

        agent_messages = [m for m in task["history"] if m["role"] == "ROLE_AGENT"]
        build_data = [
            p["data"] for p in agent_messages[-1]["parts"]
            if p.get("data", {}).get("type") == "build"
        ][0]
        artifact = build_data["artifact"]

    # critic side: fetch build via R20 route (sha verified inside the client)
    critic_http = asgi_client(app, token=CRITIC)
    async with critic_http:
        critic = client_mod.A2AClient("http://testserver", token=CRITIC, http_client=critic_http)
        blob, sha256, size = await critic.fetch_build(task_id, artifact["artifactId"])
    assert blob == build_result.binary_bytes
    assert sha256 == hashlib.sha256(blob).hexdigest() == artifact["sha256"]
    assert size == len(blob)


async def test_fetch_build_unknown_artifact(store):
    app = server.create_app(store, responder=ScriptedResponder(),
                            designer_token=DESIGNER, critic_token=CRITIC)
    http_client = asgi_client(app, token=CRITIC)
    async with http_client:
        a2a = client_mod.A2AClient("http://testserver", token=CRITIC, http_client=http_client)
        with pytest.raises(client_mod.A2AClientError) as excinfo:
            await a2a.fetch_build("task-nope", "artifact-nope")
    assert excinfo.value.code == config.A2A_TASK_NOT_FOUND


async def test_fetch_build_sha_mismatch(store):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-the-real-bytes",
            headers={"X-Build-Sha256": "0" * 64,
                     "Content-Type": "application/octet-stream"},
        )

    transport = httpx.MockTransport(handler)
    mock_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    a2a = client_mod.A2AClient("http://testserver", token=CRITIC, http_client=mock_client)
    with pytest.raises(client_mod.ShaMismatch):
        await a2a.fetch_build("t", "a")
    await mock_client.aclose()


async def test_rpc_unauthorized_raises(store):
    app = server.create_app(store, responder=ScriptedResponder(),
                            designer_token=DESIGNER, critic_token=CRITIC)
    http_client = asgi_client(app, token="wrong-token")
    async with http_client:
        a2a = client_mod.A2AClient("http://testserver", token="wrong-token", http_client=http_client)
        with pytest.raises(client_mod.AuthError):
            await a2a.get_task("whatever")


async def test_rpc_error_surfaces_jsonrpc_code(store):
    app = server.create_app(store, responder=ScriptedResponder(),
                            designer_token=DESIGNER, critic_token=CRITIC)
    http_client = asgi_client(app, token=DESIGNER)
    async with http_client:
        a2a = client_mod.A2AClient("http://testserver", token=DESIGNER, http_client=http_client)
        with pytest.raises(client_mod.A2AClientError) as excinfo:
            await a2a.get_task("nope")
    assert excinfo.value.code == config.A2A_TASK_NOT_FOUND


async def test_card_missing_fields_raises_incompatible():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": "incomplete"})

    transport = httpx.MockTransport(handler)
    mock_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    a2a = client_mod.A2AClient("http://testserver", token=None, http_client=mock_client)
    with pytest.raises(client_mod.IncompatibleAgent):
        await a2a.fetch_card()
    await mock_client.aclose()
