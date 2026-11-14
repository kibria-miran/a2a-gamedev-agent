"""Slow tier: real uvicorn Coder subprocess + real fake-NIM subprocess.

S9   kill -9 the Coder mid-build-turn; restart; the client retries the same
     messageId (R6c resume); the build completes with no duplicated messages.
S10  concurrency against the real Coder subprocess: (a) two simultaneous
     SendMessage with the SAME messageId produce exactly one task; (b) an
     accept posted while the build turn is in flight gets -32602 (R8).
S14  fake NIM sleeps past LLM_TIMEOUT; the build turn fails with llm_error
     and the task is rejected with no build — bounded, and honest.

These tests need g++ and a network-isolation backend for the game sandbox
(bwrap or unshare -n); when neither exists they skip. The slow tier is
opt-in: `make slow`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from src import config, protocol, sandbox

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
TERMS = {
    "genre": "invaders",
    "grid_w": 40,
    "grid_h": 20,
    "tick_ms": 100,
    "enemy_count": 8,
    "win_condition": "survive",
    "reliability": "deterministic",
}

GAME = """
#include "harness.h"
#include <cstdio>
namespace { int x = 0; }
int harness_init(void) { return 0; }
void harness_tick(void) { ++x; }
const char* harness_render(void) {
    static char buf[32];
    std::snprintf(buf, sizeof buf, "x=%d", x);
    return buf;
}
"""

DESIGNER_TOKEN = "slow-designer-token"
CRITIC_TOKEN = "slow-critic-token"


def _isolation_available() -> bool:
    backend, _ = sandbox.probe_network_isolation()
    return backend != "unavailable"


requires_sandbox = pytest.mark.skipif(
    not _isolation_available(), reason="no network isolation backend (bwrap or unshare -n)"
)


class FakeNIM:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "tests" / "fake_nim_server.py")],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self.proc.stdout is not None
        self.port = int(self.proc.stdout.readline())
        self.url = f"http://127.0.0.1:{self.port}"
        # The port is printed before uvicorn binds; wait for it to listen.
        for _ in range(100):
            try:
                with httpx.Client(base_url=self.url, timeout=1.0) as client:
                    client.get("/__state")
                break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise TimeoutError("fake NIM did not come up")

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(base_url=self.url, timeout=10.0) as client:
            return client.post(path, json=payload).json()

    def get(self, path: str) -> dict[str, Any]:
        with httpx.Client(base_url=self.url, timeout=10.0) as client:
            return client.get(path).json()

    def stage(self, responses: list[str]) -> None:
        self.post("/__stage", {"responses": responses})

    def control(self, sleep_calls: dict[int, float]) -> None:
        self.post("/__control", {"sleep_calls": {str(k): v for k, v in sleep_calls.items()}})

    def state(self) -> dict[str, Any]:
        return self.get("/__state")

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait(timeout=5)


@pytest.fixture
def fake_nim():
    fake = FakeNIM()
    yield fake
    fake.close()


def _free_port() -> int:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_coder(db: Path, fake_nim: FakeNIM, *, llm_timeout: float = 60.0) -> _CoderProcess:
    port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "NIM_BASE_URL": f"http://127.0.0.1:{fake_nim.port}/v1",
            "NIM_MODEL": "fake-model",
            "NVIDIA_API_KEY": "dummy",
            "A2A_CODER_HOST": "127.0.0.1",
            "A2A_CODER_PORT": str(port),
            "A2A_CODER_DB": str(db),
            "A2A_TRACE_PATH": str(db.parent / f"{db.stem}.trace.jsonl"),
            "A2A_LLM_TIMEOUT": str(llm_timeout),
            "A2A_TOKEN_DESIGNER": DESIGNER_TOKEN,
            "A2A_TOKEN_CRITIC": CRITIC_TOKEN,
        }
    )
    coder = _CoderProcess(port, env)
    _wait_for_card(coder.url, timeout=20.0)
    return coder


class _CoderProcess:
    def __init__(self, port: int, env: dict[str, str]) -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "src.coder_server"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait(timeout=5)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)


def _wait_for_card(url: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with httpx.Client(base_url=url, timeout=2.0) as client:
                response = client.get(config.AGENT_CARD_PATH)
                if response.status_code == 200:
                    return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise TimeoutError(f"coder did not come up at {url}")


def rpc(
    url: str,
    method: str,
    params: dict[str, Any],
    token: str,
    timeout: float = 90.0,
) -> dict[str, Any]:
    with httpx.Client(base_url=url, timeout=timeout) as client:
        response = client.post(
            config.A2A_RPC_PATH,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            headers={"Authorization": f"Bearer {token}"},
        )
        return response.json()


def message(
    message_id: str,
    round_number: int,
    terms: dict[str, Any],
    *,
    msg_type: str,
    task_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [protocol.data_part({"type": msg_type, "round": round_number, "terms": terms})],
    }
    if task_id:
        payload["taskId"] = task_id
    return payload


def build_parts(task: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    for item in task.get("history") or []:
        for part in item.get("parts") or []:
            data = part.get("data")
            if isinstance(data, dict) and data.get("type") == "build":
                found.append(data)
    return found


def accept_message(message_id: str, round_number: int, task_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [protocol.data_part({"type": "accept", "round": round_number})],
        "taskId": task_id,
    }
    return payload


def _open_task(coder_url: str) -> tuple[str, dict[str, Any]]:
    """Offer (code path, no LLM) and return (task_id, counter_body)."""
    offer = message("mA", 0, dict(TERMS), msg_type="offer")
    body = rpc(coder_url, "SendMessage", {"message": offer}, DESIGNER_TOKEN)
    assert "error" not in body, body
    task = body["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    return task["id"], body


# ---------------------------------------------------------------------------
# S9: kill -9 mid-build, restart, resume, build exactly once
# ---------------------------------------------------------------------------


@requires_sandbox
def test_s9_kill9_midbuild_restart_resumes(tmp_path: Path, fake_nim: FakeNIM) -> None:
    fake_nim.stage([GAME, GAME])  # first copy is lost to the kill
    fake_nim.control({1: 8.0})    # the build codegen call sleeps 8 s
    db = tmp_path / "coder.db"
    coder = start_coder(db, fake_nim)
    try:
        task_id, _ = _open_task(coder.url)

        signed = message("mB", 2, dict(TERMS), msg_type="spec_signed", task_id=task_id)
        second_body = {"message": signed}

        async def send_then_die() -> None:
            async with httpx.AsyncClient(base_url=coder.url, timeout=30.0) as client:
                with contextlib.suppress(httpx.HTTPError):
                    await client.post(
                        config.A2A_RPC_PATH,
                        json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": second_body},
                        headers={"Authorization": f"Bearer {DESIGNER_TOKEN}"},
                    )

        async def kill_soon() -> None:
            await asyncio.sleep(2.0)
            coder.kill()

        async def both() -> None:
            await asyncio.gather(send_then_die(), kill_soon())

        asyncio.run(both())
    finally:
        if coder.proc.poll() is None:
            coder.close()

    # Restart against the same DB; startup recovery (T8) runs at boot.
    coder = start_coder(db, fake_nim)
    try:
        with sqlite3.connect(db) as conn:
            state = conn.execute(
                "SELECT state, round_at_turn_start FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        assert state is not None and state[0] == "input-required", state

        resumed = rpc(coder.url, "SendMessage", second_body, DESIGNER_TOKEN)
        assert "error" not in resumed, resumed
        resumed_task = resumed["result"]["task"]
        assert resumed_task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        builds = build_parts(resumed_task)
        assert len(builds) == 1, builds

        accepted = rpc(
            coder.url,
            "SendMessage",
            {"message": accept_message("mC", 3, task_id)},
            CRITIC_TOKEN,
        )
        assert "error" not in accepted, accepted
        final = accepted["result"]["task"]
        assert final["status"]["state"] == "TASK_STATE_COMPLETED"

        artifact = builds[0]["artifact"]
        with httpx.Client(base_url=coder.url, timeout=15.0) as client:
            fetched = client.get(
                f"/tasks/{task_id}/build/{artifact['artifactId']}",
                headers={"Authorization": f"Bearer {CRITIC_TOKEN}"},
            )
        assert fetched.status_code == 200
        assert fetched.headers["X-Build-Sha256"] == artifact["sha256"]
        assert len(fetched.content) == artifact["size"]

        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT role, seq FROM messages WHERE task_id = ? ORDER BY seq", (task_id,)
            ).fetchall()
        user_ids = [row for row in rows if row[0] == "user"]
        assert len(user_ids) == 3  # mA, mB (resumed, no duplication), mC
    finally:
        coder.close()


# ---------------------------------------------------------------------------
# S10: concurrency — same messageId, and busy-turn rejection
# ---------------------------------------------------------------------------


@requires_sandbox
def test_s10a_concurrent_same_message_id_one_task(tmp_path: Path, fake_nim: FakeNIM) -> None:
    db = tmp_path / "coder.db"
    coder = start_coder(db, fake_nim)
    try:
        offer = message("dup-id", 0, dict(TERMS), msg_type="offer")

        async def run() -> list[dict[str, Any]]:
            async with httpx.AsyncClient(base_url=coder.url, timeout=30.0) as client:
                body = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "SendMessage",
                    "params": {"message": offer},
                }
                return list(await asyncio.gather(
                    client.post(config.A2A_RPC_PATH, json=body,
                                headers={"Authorization": f"Bearer {DESIGNER_TOKEN}"}),
                    client.post(config.A2A_RPC_PATH, json=body,
                                headers={"Authorization": f"Bearer {DESIGNER_TOKEN}"}),
                ))

        responses = asyncio.run(run())
        bodies = [r.json() for r in responses]
        oks = [b for b in bodies if "error" not in b]
        errors = [b for b in bodies if "error" in b]
        assert oks, bodies
        task_ids = {b["result"]["task"]["id"] for b in oks}
        assert len(task_ids) == 1
        for b in errors:
            assert b["error"]["code"] == config.JSONRPC_INVALID_PARAMS
        with sqlite3.connect(db) as conn:
            agent_rows = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role = 'agent'"
            ).fetchone()[0]
            user_rows = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role = 'user'"
            ).fetchone()[0]
        assert agent_rows == 1  # one counter, not two
        assert user_rows == 1   # the offer is stored once
    finally:
        coder.close()


@requires_sandbox
def test_s10b_accept_during_build_turn_gets_busy(tmp_path: Path, fake_nim: FakeNIM) -> None:
    fake_nim.stage([GAME])
    fake_nim.control({1: 6.0})  # the build codegen call sleeps 6 s
    db = tmp_path / "coder.db"
    coder = start_coder(db, fake_nim)
    try:
        task_id, _ = _open_task(coder.url)
        signed = message("mB", 2, dict(TERMS), msg_type="spec_signed", task_id=task_id)

        async def sign() -> None:
            async with httpx.AsyncClient(base_url=coder.url, timeout=30.0) as client:
                await client.post(
                    config.A2A_RPC_PATH,
                    json={"jsonrpc": "2.0", "id": 1,
                          "method": "SendMessage", "params": {"message": signed}},
                    headers={"Authorization": f"Bearer {DESIGNER_TOKEN}"},
                )

        async def accept_during_build() -> None:
            await asyncio.sleep(1.5)
            with httpx.Client(base_url=coder.url, timeout=10.0) as client:
                response = client.post(
                    config.A2A_RPC_PATH,
                    json={"jsonrpc": "2.0", "id": 2,
                          "method": "SendMessage",
                          "params": {"message": accept_message("mC", 3, task_id)}},
                    headers={"Authorization": f"Bearer {CRITIC_TOKEN}"},
                )
            body = response.json()
            assert "error" in body, body
            assert body["error"]["code"] == config.JSONRPC_INVALID_PARAMS

        async def both() -> None:
            await asyncio.gather(sign(), accept_during_build())

        asyncio.run(both())

        # The task is input-required once the build lands.
        resumed = rpc(
            coder.url,
            "SendMessage",
            {"message": accept_message("mD", 3, task_id)},
            CRITIC_TOKEN,
        )
        assert "error" not in resumed, resumed
        assert resumed["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    finally:
        coder.close()


# ---------------------------------------------------------------------------
# S14: fake NIM slower than LLM_TIMEOUT — build fails honestly, no artifact
# ---------------------------------------------------------------------------


@requires_sandbox
def test_s14_llm_timeout_yields_llm_error_rejection(tmp_path: Path, fake_nim: FakeNIM) -> None:
    fake_nim.stage([GAME])
    fake_nim.control({1: 30.0})  # far past the 3 s LLM_TIMEOUT we set below
    db = tmp_path / "coder.db"
    coder = start_coder(db, fake_nim, llm_timeout=3.0)
    try:
        task_id, _ = _open_task(coder.url)
        signed = message("mB", 2, dict(TERMS), msg_type="spec_signed", task_id=task_id)
        body = rpc(
            coder.url,
            "SendMessage",
            {"message": signed},
            DESIGNER_TOKEN,
            timeout=60.0,
        )
        assert "error" not in body, body
        task = body["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_REJECTED"
        assert build_parts(task) == []
        walk_aways = [
            (part.get("data") or {})
            for item in task.get("history") or []
            for part in item.get("parts") or []
            if (part.get("data") or {}).get("type") == "walk_away"
        ]
        assert walk_aways, task
        assert walk_aways[-1].get("reason") == "llm_error"
    finally:
        coder.close()
