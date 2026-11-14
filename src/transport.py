"""Shared wire/transport helpers for the agents (payload builders)."""

from __future__ import annotations

import uuid
from typing import Any

from src import protocol


def message_dict(
    *,
    role: str,
    parts: list[dict[str, Any]],
    task_id: str | None = None,
    context_id: str | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    wire_role = "ROLE_USER" if role == "user" else "ROLE_AGENT"
    msg: dict[str, Any] = {
        "messageId": message_id or str(uuid.uuid4()),
        "role": wire_role,
        "parts": parts,
    }
    if task_id:
        msg["taskId"] = task_id
    if context_id:
        msg["contextId"] = context_id
    return msg


def offer_parts(round_number: int, terms: dict[str, Any]) -> list[dict[str, Any]]:
    return [protocol.data_part({"type": "offer", "round": round_number, "terms": terms})]


def counter_parts(round_number: int, terms: dict[str, Any]) -> list[dict[str, Any]]:
    return [protocol.data_part({"type": "counter", "round": round_number, "terms": terms})]


def spec_signed_parts(round_number: int, terms: dict[str, Any]) -> list[dict[str, Any]]:
    return [protocol.data_part({"type": "spec_signed", "round": round_number, "terms": terms})]


def accept_parts(round_number: int, build: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    part: dict[str, Any] = {"type": "accept", "round": round_number}
    if build is not None:
        part["build"] = build
    return [protocol.data_part(part)]


def walk_away_parts(round_number: int, reason: str | None = None) -> list[dict[str, Any]]:
    part: dict[str, Any] = {"type": "walk_away", "round": round_number}
    if reason:
        part["reason"] = reason
    return [protocol.data_part(part)]
