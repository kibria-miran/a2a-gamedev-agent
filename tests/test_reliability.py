"""Phase 3 reliability: startup recovery, resume/drift, replay, cancel, LLMError."""

import asyncio
import threading

import httpx
import pytest

from src import config, decision, llm
from src import store as store_mod
from src.server import create_app
from src.transport import message_dict, offer_parts

DESIGNER = "designer-secret-1"
CRITIC = "critic-secret-2"

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


def offer_msg(round_number=0, terms=None):
    return message_dict(role="user", parts=offer_parts(round_number, terms or OFFER))


def make_client(app, token=DESIGNER):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )


async def send(client, message, method="SendMessage", params=None):
    if params is None:
        params = {"message": message}
    return await client.post(
        "/", json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
    )


def counter_decision(round_number=1):
    return decision.Decision(
        kind="counter", terms=OFFER, round=round_number, own_u=0.9, source="code"
    )


async def test_startup_recovery_on_app_create(store):
    task = store.create_task()
    create_app(
        store, responder=lambda *a: counter_decision(),
        designer_token=DESIGNER, critic_token=CRITIC,
    )
    assert store.get_task(task["id"])["state"] == "input-required"


async def test_resume_after_startup_recovery(store):
    view = offer_msg(0)
    task = store.create_task()
    store.begin_turn(task["id"], view["messageId"], view["parts"])
    assert store.startup_recovery() == 1
    recovered = store.get_task(task["id"])
    assert recovered["state"] == "input-required"
    # The pre-increment message round, not the post-increment task round.
    assert recovered["round_at_turn_start"] == 0

    calls = []

    def responder(task_view, incoming, own_history):
        calls.append(incoming)
        return counter_decision(1)

    app = create_app(
        store, responder=responder, designer_token=DESIGNER,
        critic_token=CRITIC, startup_recovery=False,
    )
    async with make_client(app) as client:
        response = await send(client, view)
    body = response.json()
    assert "error" not in body, body
    assert body["result"]["task"]["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert len(calls) == 1
    assert calls[0]["type"] == "offer"
    assert store.get_task(task["id"])["round"] == 2


async def test_resume_drift_is_rejected(store):
    view = offer_msg(0)
    task = store.create_task()
    store.begin_turn(task["id"], view["messageId"], view["parts"])
    store.startup_recovery()
    with store.conn:
        store.conn.execute(
            "UPDATE tasks SET round_at_turn_start = 7 WHERE id = ?", (task["id"],)
        )

    app = create_app(
        store, responder=lambda *a: counter_decision(), designer_token=DESIGNER,
        critic_token=CRITIC, startup_recovery=False,
    )
    async with make_client(app) as client:
        response = await send(client, view)
    assert response.json()["error"]["message"] == config.REASONS["RESUME_DRIFT"]


async def test_replay_does_not_rerun_the_turn(store):
    calls = []

    def responder(task_view, incoming, own_history):
        calls.append(incoming)
        return counter_decision(1)

    app = create_app(
        store, responder=responder, designer_token=DESIGNER, critic_token=CRITIC
    )
    message = offer_msg(0)
    message["messageId"] = "stable-replay-1"
    async with make_client(app) as client:
        first = await send(client, message)
        second = await send(client, message)
    assert first.json()["result"]["task"]["id"] == second.json()["result"]["task"]["id"]
    assert len(calls) == 1


async def test_cancel_running_turn_discards_the_late_result(store, monkeypatch):
    monkeypatch.setattr(config, "BLOCKING_TIMEOUT", 0.05)
    started = threading.Event()
    release = threading.Event()

    def slow_responder(task_view, incoming, own_history):
        started.set()
        release.wait(5)
        return counter_decision(1)

    app = create_app(
        store, responder=slow_responder, designer_token=DESIGNER, critic_token=CRITIC
    )
    async with make_client(app) as client:
        response = await send(client, offer_msg(0))
        task_id = response.json()["result"]["task"]["id"]
        # The turn is still running, so the blocking send returned "working".
        assert response.json()["result"]["task"]["status"]["state"] == "TASK_STATE_WORKING"
        assert started.wait(2)

        canceled = await send(client, None, method="CancelTask", params={"id": task_id})
        assert canceled.json()["result"]["status"]["state"] == "TASK_STATE_CANCELED"
        release.set()

        for _ in range(100):
            if not app.state.turn_tasks:
                break
            await asyncio.sleep(0.01)

        task = store.get_task(task_id)
        assert task["state"] == "canceled"
        # No agent message was committed after cancellation.
        assert store.last_message(task_id, "agent") is None


async def test_llm_error_in_a_turn_fails_the_task(store):
    def failing_responder(task_view, incoming, own_history):
        raise llm.LLMError("provider down")

    app = create_app(
        store, responder=failing_responder, designer_token=DESIGNER,
        critic_token=CRITIC,
    )
    async with make_client(app) as client:
        response = await send(client, offer_msg(0))
    assert response.json()["result"]["task"]["status"]["state"] == "TASK_STATE_FAILED"
