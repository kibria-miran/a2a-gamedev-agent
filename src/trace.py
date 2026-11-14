"""Append-only JSONL trace (section 8.9/E-35/E-51), game-dev kinds.

One line per event. Fields: ``ts``, ``agent``, ``task_id`` (omitted when
unknown), ``round``, ``direction`` (``in`` | ``out``), ``kind``,
``error_kind`` (only when ``severity == "error"``), ``severity``,
``terms`` (spec terms are not private, but quantized like the demo for
uniformity), ``own_U``, ``source``, ``tokens_in``, ``tokens_out``.

Tracing is disabled by default (``trace()`` is a no-op); entry points call
``configure()`` with ``config.TRACE_PATH``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src import clock, config

_ROUND = round  # builtin kept under an alias

MESSAGE_KINDS = frozenset({"offer", "counter", "accept", "walk_away", "spec_signed", "build"})
LIFECYCLE_KINDS = frozenset({"replay", "resume", "discarded", "turn_failed", "server_busy"})
KINDS = MESSAGE_KINDS | LIFECYCLE_KINDS

ERROR_KINDS = frozenset(
    {"bad_params", "bad_round", "turn_failed", "injection_attempt", "server_busy", "error"}
)

_PATH: str | None = None


def configure(path: str | os.PathLike[str] | None) -> None:
    """Enable (or, with ``None``, disable) the trace sink."""

    global _PATH
    _PATH = os.fspath(path) if path is not None else None
    if _PATH:
        parent = os.path.dirname(_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)


def reset() -> None:
    """Disable the sink and drop the path (tests)."""

    global _PATH
    _PATH = None


def enabled() -> bool:
    return _PATH is not None


def _quantize(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), config.TRACE_U_DECIMALS)


def trace(
    kind: str,
    *,
    direction: str,
    agent: str,
    task_id: str | None = None,
    round: int | None = None,  # noqa: A002 - wire field name
    severity: str = "info",
    error_kind: str | None = None,
    terms: dict[str, Any] | None = None,
    own_u: float | None = None,
    source: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    latency_ms: float | None = None,
) -> None:
    if _PATH is None:
        return
    if kind not in KINDS:
        raise ValueError(f"unknown trace kind: {kind!r}")
    if direction not in {"in", "out"}:
        raise ValueError(f"unknown trace direction: {direction!r}")
    if severity not in {"info", "error"}:
        raise ValueError(f"unknown trace severity: {severity!r}")

    event: dict[str, Any] = {
        "ts": clock.now().isoformat(),
        "agent": agent,
        "round": round,
        "direction": direction,
        "kind": kind,
        "severity": severity,
    }
    if task_id is not None:
        event["task_id"] = task_id
    if severity == "error":
        if error_kind is None:
            error_kind = "error"
        if error_kind not in ERROR_KINDS:
            raise ValueError(f"unknown error_kind: {error_kind!r}")
        event["error_kind"] = error_kind
    if terms is not None:
        event["terms"] = terms
    q_own = _quantize(own_u)
    if q_own is not None:
        event["own_U"] = q_own
    if source is not None:
        event["source"] = source
    if tokens_in is not None:
        event["tokens_in"] = int(tokens_in)
    if tokens_out is not None:
        event["tokens_out"] = int(tokens_out)
    if latency_ms is not None:
        event["latency_ms"] = _ROUND(float(latency_ms), 1)

    with open(_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, separators=(",", ":")) + "\n")
