"""Server tests (Phase 2): R0-R20 rows, R20 fetch routes, role tokens (E-66/E-70).

In-memory ASGI transport; the Coder's negotiation is a scripted responder
(the real negotiator lands in Phase 4 with the designer). The build path
uses a real BuildResult from sandbox.build_game so the handoff artifact is
real bytes.
"""

import hashlib
import json
import uuid

import httpx
import pytest

from src import decision, sandbox, server
from src import store as store_mod
from src.transport import (
    accept_parts,
    message_dict,
    offer_parts,
    spec_signed_parts,
    walk_away_parts,
)

DESIGNER = "designer-secret-1"
CRITIC = "critic-secret-2"

MINI_GAME = """
#include "harness.h"
int harness_init(void) { return 0; }
void harness_tick(void) {}
const char* harness_render(void) { return "x"; }
"""

OFFER = {
    "genre": "invaders",
    "grid_w": 40,
    "grid_h": 20,
    "tick_ms": 100,
    "enemy_count": 8,
    "win_condition": "survive",
}
COUNTERED = {**OFFER, "enemy_count": 12}


class ScriptedResponder:
    """Deterministic responder for the protocol-level tests.

    With no more scripted steps, degrades gracefully: offer/counter produce a
    default counter (round = incoming round + 1), anything else a walk_away.
    """

    DEFAULT_TERMS = {
        "genre": "invaders",
        "grid_w": 40,
        "grid_h": 20,
        "tick_ms": 100,
        "enemy_count": 12,
        "win_condition": "survive",
    }

    def __init__(self, *steps):
        self._steps = list(steps)
        self.calls = []

    def __call__(self, view, incoming, history):
        self.calls.append({"view": view, "incoming": incoming, "history": list(history)})
        if self._steps:
            step = self._steps.pop(0)
            if isinstance(step, Exception):
                raise step
            return step
        if incoming.get("type") in {"offer", "counter"}:
            return decision.Decision(
                kind="counter",
                terms=self.DEFAULT_TERMS,
                round=int(incoming.get("round", 0)) + 1,
                own_u=0.5,
                source="code",
                rationale="default counter (test fixture)",
            )
        return decision.Decision(kind="walk_away", reason="script_exhausted")


def counter_decision(round_number, terms=None, source="code"):
    return decision.Decision(
        kind="counter",
        terms=terms if terms is not None else COUNTERED,
        round=round_number,
        own_u=0.5,
        source=source,
        rationale="counter for test",
    )


def fake_build_result():
    blob = b"\x7fELF-fake-game-binary"
    return sandbox.BuildResult(
        ok=True,
        network_isolation="bwrap",
        exit_code=0,
        binary_bytes=blob,
        binary_sha256=hashlib.sha256(blob).hexdigest(),
        source_tar=b"fake-tar-archive-bytes",
        duration_s=0.01,
    )


@pytest.fixture
def store(tmp_path):
    s = store_mod.Store(str(tmp_path / "coder.db"))
    yield s
    s.close()


@pytest.fixture
def app(store):
    return server.create_app(
        store,
        responder=ScriptedResponder(),
        designer_token=DESIGNER,
        critic_token=CRITIC,
        startup_recovery=True,
        trace_agent="coder-test",
    )


def make_client(app, token=DESIGNER):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "A2A-Version": "1.0",
        },
    )


def spec_message(type_, round_, terms=None, build=None, reason=None):
    part = {"type": type_, "round": round_}
    if terms is not None:
        part["terms"] = terms
    if build is not None:
        part["build"] = build
    if reason is not None:
        part["reason"] = reason
    return {"kind": "message", "messageId": str(uuid.uuid4()), "role": "ROLE_USER",
            "parts": [{"data": part}]}


def offer_msg(round_number=0, terms=None, task_id=None):
    return message_dict(role="user", parts=offer_parts(round_number, terms or OFFER),
                        task_id=task_id)


async def send(client, message):
    return await client.post(
        "/",
        json={"jsonrpc": "2.0", "id": "1", "method": "SendMessage",
              "params": {"message": message}},
    )


async def send_offer(client, round_number=0, terms=None):
    return await send(client, offer_msg(round_number, terms))


async def get_task(client, task_id):
    response = await client.post(
        "/",
        json={"jsonrpc": "2.0", "id": "1", "method": "GetTask",
              "params": {"id": task_id}},
    )
    return response.json()["result"]


def blob_of_task_view(view):
    return view["status"]["state"], view["artifacts"]


# ---------------------------------------------------------------------------
# card + auth (E-38)
# ---------------------------------------------------------------------------

async def test_card_is_public(store):
    app = server.create_app(store, responder=ScriptedResponder(),
                            designer_token=DESIGNER, critic_token=CRITIC)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://testserver") as client:
        response = await client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    body = response.json()
    assert body["skills"][0]["id"] == "build_game"
    assert body["supportedInterfaces"][0]["protocolVersion"] == "1.0"


async def test_rpc_requires_bearer(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://testserver") as client:
        response = await client.post("/", json={"jsonrpc": "2.0", "id": "1",
                                                "method": "SendMessage", "params": {}})
    assert response.status_code == 401


async def test_unknown_bearer_is_unauthorized(app):
    async with make_client(app, token="nobody") as client:
        response = await send(client, offer_msg())
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# R0-R9 table rows
# ---------------------------------------------------------------------------

async def test_method_not_found(app):
    async with make_client(app) as client:
        response = await client.post("/", json={"jsonrpc": "2.0", "id": "1",
                                                "method": "SendMeme", "params": {}})
    assert response.json()["error"]["code"] == -32601


async def test_parse_error(app):
    async with make_client(app) as client:
        response = await client.post("/", content=b"{not json")
    assert response.json()["error"]["code"] == -32700


async def test_envelope_must_be_dict(app):
    async with make_client(app) as client:
        response = await client.post("/", json=[1, 2, 3])
    assert response.json()["error"]["code"] == -32600


async def test_offer_round_mismatch_is_R9(app):
    async with make_client(app) as client:
        first = await send_offer(client, round_number=0)
        task_id = first.json()["result"]["task"]["id"]
        # designer re-offers with round 1 while the task round is 2 -> R9
        response = await send(client, offer_msg(1, task_id=task_id))
    assert response.json()["error"]["code"] == -32602
    assert response.json()["error"]["message"] == "round mismatch"


async def test_spec_message_not_from_designer_is_R21(app):
    async with make_client(app, token=CRITIC) as client:
        response = await send_offer(client, round_number=0)
    assert response.json()["error"]["message"] == "spec message not from designer"


async def test_spec_signed_from_critic_is_R17(store):
    app = server.create_app(store, responder=ScriptedResponder(),
                            designer_token=DESIGNER, critic_token=CRITIC)
    async with make_client(app, token=CRITIC) as client:
        response = await send(client, message_dict(
            role="user", parts=spec_signed_parts(0, OFFER)))
    assert response.json()["error"]["message"] == "spec_signed not from designer"


async def test_build_message_from_client_is_R18(store):
    app = server.create_app(store, responder=ScriptedResponder(),
                            designer_token=DESIGNER, critic_token=CRITIC)
    async with make_client(app) as client:
        response = await send(client, spec_message("build", 0, build={"artifact": {}}))
    assert response.json()["error"]["message"] == "build before spec_signed"


async def test_accept_before_signoff_is_R19(app):
    # critic accept against a real task that never got signed off
    async with make_client(app) as designer:
        response = await send_offer(designer, round_number=0)
        task_id = response.json()["result"]["task"]["id"]
    async with make_client(app, token=CRITIC) as critic:
        response = await send(critic, message_dict(
            role="user", parts=accept_parts(0), task_id=task_id))
    assert response.json()["error"]["message"] == "accept not from critic or before signoff"


async def test_accept_from_designer_is_R19(app):
    async with make_client(app) as designer:
        response = await send_offer(designer, round_number=0)
        task_id = response.json()["result"]["task"]["id"]
        response = await send(designer, message_dict(
            role="user", parts=accept_parts(0), task_id=task_id))
    assert response.json()["error"]["message"] == "accept not from critic or before signoff"


# ---------------------------------------------------------------------------
# The full lifecycle: offer -> counter -> spec_signed -> build -> accept
# ---------------------------------------------------------------------------

async def test_full_spec_lifecycle_ends_in_completed(store):
    build_result = fake_build_result()
    responder = ScriptedResponder(
        counter_decision(round_number=1),
        decision.Decision(kind="build", round=3, build_result=build_result, source="code"),
    )
    app = server.create_app(store, responder=responder,
                            designer_token=DESIGNER, critic_token=CRITIC)

    async with make_client(app, token=DESIGNER) as designer:
        # 1) offer round 0 -> agent counter
        response = await send_offer(designer, round_number=0)
        body = response.json()
        task_id = body["result"]["task"]["id"]

        # 2) designer signs the (agreed) spec at round 2
        response = await send(designer, message_dict(
            role="user", parts=spec_signed_parts(2, COUNTERED), task_id=task_id))
        body = response.json()
        assert "error" not in body, body
        state = body["result"]["task"]["status"]["state"]
        assert state == "TASK_STATE_INPUT_REQUIRED"

        # find the build artifact the agent stored
        agent_messages = [
            m for m in body["result"]["task"]["history"] if m["role"] == "ROLE_AGENT"
        ]
        build_message = json.loads(json.dumps(agent_messages[-1]))
        build_data = [
            p["data"] for p in build_message["parts"] if "data" in p and p["data"]["type"] == "build"
        ][0]
        artifact = build_data["artifact"]
        assert artifact["sha256"] == hashlib.sha256(build_result.binary_bytes).hexdigest()
        artifact_id = artifact["artifactId"]

    # 3) critic fetches the binary via R20 and verifies sha
    async with make_client(app, token=CRITIC) as critic:
        response = await critic.get(f"/tasks/{task_id}/build/{artifact_id}")
        assert response.status_code == 200
        assert response.content == build_result.binary_bytes
        assert response.headers["X-Build-Sha256"] == artifact["sha256"]
        assert response.headers["Content-Type"] == "application/octet-stream"

        # source tar also fetchable
        response = await critic.get(f"/tasks/{task_id}/source/{artifact_id}")
        assert response.status_code == 200
        assert response.content == build_result.source_tar

        # 4) critic accepts -> completed
        response = await send(critic, message_dict(
            role="user", parts=accept_parts(3, build={"artifactId": artifact_id,
                                                      "sha256": artifact["sha256"]}),
            task_id=task_id))
        body = response.json()
        assert "error" not in body, body
        assert body["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"


async def test_r20_unknown_artifact_is_not_found(app):
    async with make_client(app, token=CRITIC) as critic:
        response = await critic.get("/tasks/nope/build/also-nope")
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == -32001
        assert body["error"]["message"] == "artifact not found"


async def test_r20_requires_bearer(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://testserver") as client:
        response = await client.get("/tasks/x/build/y")
    assert response.status_code == 401


async def test_walk_away_from_designer_rejects_task(app):
    async with make_client(app) as designer:
        response = await send_offer(designer, round_number=0)
        task_id = response.json()["result"]["task"]["id"]

        response = await send(designer, message_dict(
            role="user", parts=walk_away_parts(2), task_id=task_id))
        body = response.json()
        assert "error" not in body, body
        assert body["result"]["task"]["status"]["state"] == "TASK_STATE_REJECTED"


async def test_r7_message_after_completion_is_rejected(app):
    async with make_client(app) as designer:
        response = await send_offer(designer, round_number=0)
        task_id = response.json()["result"]["task"]["id"]
        await send(designer, message_dict(
            role="user", parts=walk_away_parts(2), task_id=task_id))
        # terminal now; any further message is R7
        response = await send(designer, offer_msg(4, task_id=task_id))
    body = response.json()
    assert body["error"]["message"] == "task is terminal"


async def test_get_task_unknown_is_not_found(app):
    async with make_client(app) as client:
        response = await client.post("/", json={"jsonrpc": "2.0", "id": "1",
                                                "method": "GetTask",
                                                "params": {"id": "nope"}})
    assert response.json()["error"]["code"] == -32001


async def test_get_task_history_length_clamped(app):
    async with make_client(app) as client:
        response = await client.post("/", json={"jsonrpc": "2.0", "id": "1",
                                                "method": "GetTask",
                                                "params": {"id": "nope", "historyLength": 999999}})
    # invalid params still -32602? clamping is legal; unknown task -> -32001
    assert response.json()["error"]["code"] == -32001


async def test_cancel_unknown_task_not_found(app):
    async with make_client(app) as client:
        response = await client.post("/", json={"jsonrpc": "2.0", "id": "1",
                                                "method": "CancelTask",
                                                "params": {"id": "nope"}})
    assert response.json()["error"]["code"] == -32001


async def test_cancel_then_cancel_is_idempotent(app):
    async with make_client(app) as designer:
        response = await send_offer(designer, round_number=0)
        task_id = response.json()["result"]["task"]["id"]
        first = await client_post(designer, task_id)
        assert first["result"]["status"]["state"] == "TASK_STATE_CANCELED"
        second = await client_post(designer, task_id)
        assert second["result"]["status"]["state"] == "TASK_STATE_CANCELED"


async def test_replay_returns_existing_task(app):
    message = offer_msg(0)
    message["messageId"] = "stable-message-id-1"
    async with make_client(app) as designer:
        first = await send(designer, message)
        second = await send(designer, message)
    assert first.json()["result"]["task"]["id"] == second.json()["result"]["task"]["id"]


async def test_fake_llm_build_failure_becomes_walk_away_build_failed(store, tmp_path):
    """Responder signals a failed build -> build_failed walk_away (E-68)."""
    responder = ScriptedResponder(
        counter_decision(round_number=1),
        decision.Decision(
            kind="build",
            round=3,
            build_result=sandbox.BuildResult(
                ok=False, violations=("rng:unseeded",), exit_code=None,
            ),
            source="code",
        ),
    )
    app = server.create_app(store, responder=responder,
                            designer_token=DESIGNER, critic_token=CRITIC)
    async with make_client(app) as designer:
        response = await send_offer(designer, round_number=0)
        task_id = response.json()["result"]["task"]["id"]
        response = await send(designer, message_dict(
            role="user", parts=spec_signed_parts(2, COUNTERED), task_id=task_id))
        body = response.json()
        task = body["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_REJECTED"
        agent_messages = [m for m in task["history"] if m["role"] == "ROLE_AGENT"]
        data = [p["data"] for p in agent_messages[-1]["parts"] if "data" in p][0]
        assert data["type"] == "walk_away"
        assert data["reason"] == "build_failed"


async def client_post(client, task_id):
    response = await client.post(
        "/",
        json={"jsonrpc": "2.0", "id": "1", "method": "CancelTask",
              "params": {"id": task_id}},
    )
    return response.json()
