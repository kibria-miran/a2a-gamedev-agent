"""SQLite persistence, per-task locks and the CAS commit (sections 7, 8.5).

One Store instance owns one SQLite connection. All methods are synchronous;
the server serializes per-task work with the asyncio locks exposed here.

§7.0 delta (E-70): the artifacts table gains ``binary``/``source_tar`` BLOB
columns; both are size-capped by ``config.MAX_ARTIFACT_BYTES`` and rejected
at write time. The task table gains ``spec_signed`` (E-66): set when a
``spec_signed`` part is committed, so R18/R19 can be checked with a plain
column read.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from typing import Any

from src import clock, config, state_machine

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    context_id           TEXT,
    state                TEXT NOT NULL,
    round                INTEGER NOT NULL DEFAULT 0,
    round_at_turn_start  INTEGER,
    own_utility_history  TEXT NOT NULL DEFAULT '[]',
    spec_signed          INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_open ON tasks(state)
    WHERE state NOT IN ('completed','canceled','failed','rejected');

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL REFERENCES tasks(id),
    message_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    parts_json  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE(task_id, seq)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_user_message_id
    ON messages(message_id) WHERE role = 'user';

CREATE TABLE IF NOT EXISTS artifacts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL REFERENCES tasks(id),
    artifact_id  TEXT NOT NULL,
    name         TEXT,
    parts_json   TEXT NOT NULL,
    binary       BLOB NULL,
    source_tar   BLOB NULL,
    created_at   TEXT NOT NULL
);
"""

# v1.1 agent (Coder service) part types. spec_signed arrives from the
# Designer (user role) and is deliberately NOT an agent type.
_AGENT_TYPES = {"counter", "build", "walk_away"}

# E-67: spec messages are the negotiation itself — offer/counter/spec_signed/
# walk_away — and each commits +1 to tasks.round (spec message index). Build,
# review/accept, and failed deliveries are not spec messages and commit +0.
_SPEC_MESSAGE_TYPES = {"offer", "counter", "spec_signed", "walk_away"}


class StoreError(Exception):
    pass


def _now_iso() -> str:
    return clock.to_iso(clock.now())


def _part_type(parts: list[dict[str, Any]]) -> str | None:
    for part in parts:
        data = part.get("data")
        if isinstance(data, dict) and isinstance(data.get("type"), str):
            return data["type"]
    return None


class Store:
    def __init__(self, path: str, *, journal_mode: str = "WAL") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(f"PRAGMA journal_mode={journal_mode}")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute(f"PRAGMA busy_timeout={config.SQLITE_BUSY_TIMEOUT_MS}")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._locks: dict[str, asyncio.Lock] = {}
        self.creation_lock = asyncio.Lock()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def lock_for(self, task_id: str) -> asyncio.Lock:
        lock = self._locks.get(task_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[task_id] = lock
        return lock

    # -- task rows ----------------------------------------------------------

    def create_task(self, context_id: str | None = None) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        now = _now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO tasks (id, context_id, state, round, round_at_turn_start,"
                " own_utility_history, spec_signed, created_at, updated_at)"
                " VALUES (?, ?, 'submitted', 0, NULL, '[]', 0, ?, ?)",
                (task_id, context_id, now, now),
            )
        task = self.get_task(task_id)
        assert task is not None
        return task

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def open_task_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE state NOT IN"
            " ('completed','canceled','failed','rejected')"
        ).fetchone()
        return int(row["n"])

    def own_utility_history(self, task_id: str) -> list[float]:
        task = self.get_task(task_id)
        if task is None:
            return []
        return [float(v) for v in json.loads(task["own_utility_history"])]

    def spec_signed(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        return bool(task and task["spec_signed"])

    # -- messages -----------------------------------------------------------

    @staticmethod
    def _check_agent_parts(parts: list[dict[str, Any]], role: str) -> None:
        if role != "agent":
            return
        data_parts = [p for p in parts if p.get("data") is not None]
        text_parts = [p for p in parts if p.get("text") is not None]
        if len(data_parts) != 1 or len(text_parts) > 1:
            raise StoreError("agent message must have one data part and at most one text part")
        data = data_parts[0]["data"]
        if not isinstance(data, dict):
            raise StoreError("agent data part must be an object")
        if data.get("type") not in _AGENT_TYPES:
            raise StoreError(
                f"agent message data type {data.get('type')!r} must be one of"
                f" {sorted(_AGENT_TYPES)}"
            )

    def _insert_message(
        self,
        task_id: str,
        message_id: str,
        role: str,
        parts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._check_agent_parts(parts, role)
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS seq FROM messages WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        seq = int(row["seq"]) + 1
        self.conn.execute(
            "INSERT INTO messages (task_id, message_id, role, parts_json, seq, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, message_id, role, json.dumps(parts), seq, _now_iso()),
        )
        stored = self.conn.execute(
            "SELECT * FROM messages WHERE task_id = ? AND seq = ?", (task_id, seq)
        ).fetchone()
        assert stored is not None
        return self._message_row(stored)

    def append_message(
        self,
        task_id: str,
        message_id: str,
        role: str,
        parts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if role not in config.ROLES:
            raise StoreError(f"unknown role {role!r}")
        with self.conn:
            return self._insert_message(task_id, message_id, role, parts)

    @staticmethod
    def _message_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "task_id": row["task_id"],
            "message_id": row["message_id"],
            "role": row["role"],
            "parts": json.loads(row["parts_json"]),
            "seq": row["seq"],
            "created_at": row["created_at"],
        }

    def find_message(self, message_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM messages WHERE message_id = ? ORDER BY id LIMIT 1",
            (message_id,),
        ).fetchone()
        return self._message_row(row) if row else None

    def messages_for_task(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE task_id = ? ORDER BY seq", (task_id,)
        ).fetchall()
        return [self._message_row(row) for row in rows]

    def last_message(self, task_id: str, role: str | None = None) -> dict[str, Any] | None:
        if role is None:
            row = self.conn.execute(
                "SELECT * FROM messages WHERE task_id = ? ORDER BY seq DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM messages WHERE task_id = ? AND role = ?"
                " ORDER BY seq DESC LIMIT 1",
                (task_id, role),
            ).fetchone()
        return self._message_row(row) if row else None

    @staticmethod
    def message_round(message: dict[str, Any]) -> int | None:
        for part in message["parts"]:
            data = part.get("data")
            if isinstance(data, dict) and "round" in data:
                value = data["round"]
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
        return None

    # -- turn lifecycle -----------------------------------------------------

    def begin_turn(
        self,
        task_id: str,
        message_id: str,
        parts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """R14: store the incoming spec message, set round_at_turn_start, go working.

        The incoming message IS a spec message (E-67): tasks.round advances
        by exactly 1 and round_at_turn_start keeps the pre-increment value
        (needed by T8 recovery / RESUME_DRIFT). If the incoming parts carry a
        ``spec_signed`` data part, the spec_signed flag is set in the same
        transaction (E-66). The flag is sticky — never cleared by the store.
        """

        with self.conn:
            task = self.get_task(task_id)
            if task is None:
                raise StoreError("task not found")
            state_machine.transition(task["state"], "working")
            self._insert_message(task_id, message_id, "user", parts)
            part_type = _part_type(parts)
            is_signoff = 1 if part_type == "spec_signed" else 0
            commits_round = 1 if part_type in _SPEC_MESSAGE_TYPES else 0
            self.conn.execute(
                "UPDATE tasks SET state = 'working', round_at_turn_start = round,"
                " round = round + ?, spec_signed = MAX(spec_signed, ?),"
                " updated_at = ? WHERE id = ?",
                (commits_round, is_signoff, _now_iso(), task_id),
            )
        updated = self.get_task(task_id)
        assert updated is not None
        return updated

    def resume_turn(self, task_id: str) -> dict[str, Any]:
        """R6c: re-enter working without inserting a message or touching round."""

        with self.conn:
            task = self.get_task(task_id)
            if task is None:
                raise StoreError("task not found")
            if task["round_at_turn_start"] is None:
                raise StoreError("resume requires round_at_turn_start")
            if task["state"] != "working":
                state_machine.transition(task["state"], "working")
                self.conn.execute(
                    "UPDATE tasks SET state = 'working', updated_at = ? WHERE id = ?",
                    (_now_iso(), task_id),
                )
        updated = self.get_task(task_id)
        assert updated is not None
        return updated

    def cas_commit(
        self,
        task_id: str,
        state: str,
        sent_kind: str,
        *,
        own_u: float | None = None,
        agent_message_id: str | None = None,
        agent_parts: list[dict[str, Any]] | None = None,
        agreement_terms: dict[str, int] | None = None,
    ) -> bool:
        """Compare-and-set the turn result (D-15, E-47, E-67).

        Spec messages (counter, walk_away, spec_signed) advance tasks.round by
        exactly 1 (E-67); build/accept/failed deliveries advance by 0. Round
        checks for spec messages remain the server's job (R9). ``spec_signed``
        also sets the spec_signed flag (E-66).
        """

        with self.conn:
            task = self.get_task(task_id)
            if task is None or task["state"] != "working":
                return False
            state_machine.transition("working", state)

            start = task["round"]
            if sent_kind in {"counter", "walk_away", "spec_signed"}:
                new_round = int(start) + 1
                history = [float(v) for v in json.loads(task["own_utility_history"])]
                if own_u is not None:
                    history.append(float(own_u))
                history_json = json.dumps(history)
            elif sent_kind in {"accept", "failed", "build"}:
                new_round = int(start)
                history_json = task["own_utility_history"]
            else:
                raise StoreError(f"unknown sent_kind {sent_kind!r}")

            sets_spec = 1 if sent_kind == "spec_signed" else 0
            cursor = self.conn.execute(
                "UPDATE tasks SET state = ?, round = ?, round_at_turn_start = NULL,"
                " own_utility_history = ?, spec_signed = MAX(spec_signed, ?),"
                " updated_at = ? WHERE id = ? AND state = 'working'",
                (state, new_round, history_json, sets_spec, _now_iso(), task_id),
            )
            if cursor.rowcount == 1 and agent_parts is not None:
                self._insert_message(
                    task_id,
                    agent_message_id or str(uuid.uuid4()),
                    "agent",
                    agent_parts,
                )
            if cursor.rowcount == 1 and agreement_terms is not None:
                self._insert_agreement_artifact(task_id, agreement_terms, now=_now_iso())
            return cursor.rowcount == 1

    def commit_user_walk_away(
        self,
        task_id: str,
        *,
        message_id: str | None = None,
        parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Terminal walk-away from a CLIENT (designer mid-negotiation, or critic).

        No agreement, no agent reply. input-required -> working -> rejected or
        (from working, after begin_turn stored the message) working -> rejected.
        """

        with self.conn:
            task = self.get_task(task_id)
            if task is None:
                raise StoreError("task not found")
            if task["state"] == "input-required":
                state_machine.transition("input-required", "working")
                state_machine.transition("working", "rejected")
            elif task["state"] != "rejected":
                state_machine.transition(task["state"], "rejected")
            if message_id is not None and parts is not None:
                self._insert_message(task_id, message_id, "user", parts)
            self.conn.execute(
                "UPDATE tasks SET state = 'rejected', updated_at = ? WHERE id = ?",
                (_now_iso(), task_id),
            )
        updated = self.get_task(task_id)
        assert updated is not None
        return updated

    def complete_with_accept(
        self,
        task_id: str,
        terms: dict[str, int] | None = None,
        *,
        message_id: str | None = None,
        parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Critic-accept path (R19/E-66): input-required -> working -> completed.

        Unlike the demo, the critic's accept carries no negotiation terms, so
        ``terms`` is optional; when omitted no agreement artifact is written
        (the build artifact is the artifact here, via store_binary_artifact).
        """

        with self.conn:
            task = self.get_task(task_id)
            if task is None:
                raise StoreError("task not found")
            if task["state"] == "input-required":
                state_machine.transition("input-required", "working")
                state_machine.transition("working", "completed")
            elif task["state"] != "completed":
                state_machine.transition(task["state"], "completed")
            now = _now_iso()
            if message_id is not None and parts is not None:
                self._insert_message(task_id, message_id, "user", parts)
            self.conn.execute(
                "UPDATE tasks SET state = 'completed', updated_at = ? WHERE id = ?",
                (now, task_id),
            )
            if terms is not None:
                self._insert_agreement_artifact(task_id, terms, now=now)
        updated = self.get_task(task_id)
        assert updated is not None
        return updated

    def cancel_task(self, task_id: str) -> tuple[dict[str, Any] | None, int | None]:
        """C3/C4/C5. Returns (task, error_code)."""

        with self.conn:
            task = self.get_task(task_id)
            if task is None:
                return None, config.A2A_TASK_NOT_FOUND
            if task["state"] == "canceled":
                return task, None
            if state_machine.is_terminal(task["state"]):
                return task, config.A2A_TASK_NOT_CANCELABLE
            state_machine.transition(task["state"], "canceled")
            self.conn.execute(
                "UPDATE tasks SET state = 'canceled', updated_at = ? WHERE id = ?",
                (_now_iso(), task_id),
            )
        updated = self.get_task(task_id)
        assert updated is not None
        return updated, None

    def startup_recovery(self) -> int:
        """T8: submitted/working -> input-required.

        ``round_at_turn_start`` is the round of the last stored user message,
        not the post-increment ``tasks.round``: E-67's begin_turn already
        advanced ``round`` past the message, so only the message round lets
        R6c resume the interrupted turn without drift (RESUME_DRIFT). A task
        with no message falls back to the current round.
        """

        rows = self.conn.execute(
            "SELECT id, round FROM tasks WHERE state IN ('submitted','working')"
        ).fetchall()
        recovered = 0
        with self.conn:
            for row in rows:
                last = self.last_message(row["id"], "user")
                stored_round = self.message_round(last) if last is not None else None
                start = stored_round if stored_round is not None else int(row["round"])
                self.conn.execute(
                    "UPDATE tasks SET state = 'input-required',"
                    " round_at_turn_start = ?, updated_at = ? WHERE id = ?",
                    (start, _now_iso(), row["id"]),
                )
                recovered += 1
        return recovered

    # -- artifacts ----------------------------------------------------------

    def _insert_agreement_artifact(
        self, task_id: str, terms: dict[str, int], *, now: str
    ) -> None:
        self.conn.execute(
            "INSERT INTO artifacts (task_id, artifact_id, name, parts_json, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                task_id,
                str(uuid.uuid4()),
                "agreement",
                json.dumps([{"data": {"type": "agreement", "terms": terms}}]),
                now,
            ),
        )

    def store_binary_artifact(
        self,
        task_id: str,
        *,
        binary: bytes,
        name: str = "binary",
        source_tar: bytes | None = None,
        parts: list[dict[str, Any]] | None = None,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """E-70: persist the built binary (and optional source tar) as a BLOB.

        Enforces MAX_ARTIFACT_BYTES on write. Returns the artifact descriptor
        (artifactId, name, parts, sha256) — never the blob bytes themselves.
        """
        if len(binary) > config.MAX_ARTIFACT_BYTES:
            raise StoreError(
                f"binary exceeds MAX_ARTIFACT_BYTES ({len(binary)} > {config.MAX_ARTIFACT_BYTES})"
            )
        if source_tar is not None and len(source_tar) > config.MAX_ARTIFACT_BYTES:
            raise StoreError(
                f"source_tar exceeds MAX_ARTIFACT_BYTES ({len(source_tar)} > {config.MAX_ARTIFACT_BYTES})"
            )
        sha256 = hashlib.sha256(binary).hexdigest()
        artifact_id = artifact_id or str(uuid.uuid4())
        if parts is None:
            parts = [
                {
                    "data": {
                        "type": "build",
                        "name": name,
                        "sha256": sha256,
                        "size": len(binary),
                    }
                }
            ]
        now = _now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO artifacts"
                " (task_id, artifact_id, name, parts_json, binary, source_tar, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, artifact_id, name, json.dumps(parts), binary, source_tar, now),
            )
        return {
            "artifactId": artifact_id,
            "name": name,
            "parts": parts,
            "sha256": sha256,
            "size": len(binary),
        }

    def get_binary_artifact(self, task_id: str, artifact_id: str) -> dict[str, Any] | None:
        """R20 fetch: raw binary bytes + recomputed sha256 for the caller."""

        row = self.conn.execute(
            "SELECT * FROM artifacts WHERE task_id = ? AND artifact_id = ?",
            (task_id, artifact_id),
        ).fetchone()
        if row is None:
            return None
        binary = bytes(row["binary"]) if row["binary"] is not None else None
        if binary is None:
            return None
        return {
            "artifactId": row["artifact_id"],
            "name": row["name"],
            "parts": json.loads(row["parts_json"]),
            "binary": binary,
            "sha256": hashlib.sha256(binary).hexdigest(),
            "size": len(binary),
        }

    def get_source_tar_artifact(self, task_id: str, artifact_id: str) -> bytes | None:
        row = self.conn.execute(
            "SELECT source_tar FROM artifacts WHERE task_id = ? AND artifact_id = ?",
            (task_id, artifact_id),
        ).fetchone()
        if row is None or row["source_tar"] is None:
            return None
        return bytes(row["source_tar"])

    # -- views --------------------------------------------------------------

    def _artifact_rows(self, task_id: str) -> list[dict[str, Any]]:
        """List artifacts WITHOUT blob payloads (parts only)."""

        rows = self.conn.execute(
            "SELECT * FROM artifacts WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
        return [
            {
                "artifactId": row["artifact_id"],
                "name": row["name"],
                "parts": json.loads(row["parts_json"]),
                "binaryPresent": row["binary"] is not None,
                "sourceTarPresent": row["source_tar"] is not None,
            }
            for row in rows
        ]

    def task_view(
        self, task_id: str, history_length: int = config.MAX_HISTORY_RESPONSE
    ) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if task is None:
            return None

        limit = max(0, min(int(history_length), config.MAX_HISTORY_RESPONSE))
        messages = self.messages_for_task(task_id)
        history = messages[-limit:] if limit else []

        last_agent = self.last_message(task_id, "agent")
        status: dict[str, Any] = {
            "state": config.A2A_STATE_BY_INTERNAL[task["state"]],
            "timestamp": task["updated_at"],
        }
        if last_agent is not None:
            status["message"] = self._message_wire(last_agent)

        return {
            "id": task["id"],
            "contextId": task["context_id"],
            "status": status,
            "history": [self._message_wire(m) for m in history],
            "artifacts": self._artifact_rows(task_id),
        }

    @staticmethod
    def _message_wire(message: dict[str, Any]) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "messageId": message["message_id"],
            "role": config.A2A_ROLE_BY_INTERNAL[message["role"]],
            "parts": message["parts"],
        }
        if message["task_id"]:
            wire["taskId"] = message["task_id"]
        return wire

    def artifacts_for_task(self, task_id: str) -> list[dict[str, Any]]:
        return self._artifact_rows(task_id)
