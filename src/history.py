"""Read-only helpers for A2A task views.

Agents are stateless: every decision is derived from ``tasks/get`` (or the
task returned by ``SendMessage``). These helpers centralize the wire
introspection so the three agents agree on what a spec message, a build
message and the next round are (E-66/E-67).
"""

from __future__ import annotations

from typing import Any

from src import config


def state_of(view: dict[str, Any]) -> str:
    state = str(view.get("status", {}).get("state", "TASK_STATE_WORKING"))
    return config.INTERNAL_STATE_BY_A2A.get(state, state)


def role_of(message: dict[str, Any]) -> str:
    role = message.get("role")
    return "user" if role in {"ROLE_USER", "user"} else "agent"


def parts_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("parts")
    return parts if isinstance(parts, list) else []


def data_of(message: dict[str, Any]) -> dict[str, Any] | None:
    for part in parts_of(message):
        data = part.get("data")
        if isinstance(data, dict):
            return data
    return None


def text_of(message: dict[str, Any]) -> str | None:
    for part in parts_of(message):
        text = part.get("text")
        if isinstance(text, str):
            return text
    return None


def history(view: dict[str, Any]) -> list[dict[str, Any]]:
    entries = view.get("history")
    return entries if isinstance(entries, list) else []


def last_message(view: dict[str, Any]) -> dict[str, Any] | None:
    entries = history(view)
    return entries[-1] if entries else None


def spec_entries(view: dict[str, Any]) -> list[dict[str, Any]]:
    """All offer/counter/spec_signed messages, in order (E-67)."""

    found: list[dict[str, Any]] = []
    for message in history(view):
        data = data_of(message)
        if not isinstance(data, dict):
            continue
        if data.get("type") not in {"offer", "counter", "spec_signed"}:
            continue
        terms = data.get("terms")
        round_number = data.get("round")
        if not isinstance(terms, dict) or not isinstance(round_number, int):
            continue
        found.append(
            {
                "type": data["type"],
                "round": int(round_number),
                "terms": terms,
                "role": role_of(message),
                "text": text_of(message),
                "message": message,
            }
        )
    return found


def latest_spec(view: dict[str, Any]) -> dict[str, Any] | None:
    entries = spec_entries(view)
    return entries[-1] if entries else None


def next_round(view: dict[str, Any]) -> int:
    spec = latest_spec(view)
    return int(spec["round"]) + 1 if spec else 0


def latest_build(view: dict[str, Any]) -> dict[str, Any] | None:
    """The last agent build part, if any: {type, round, artifact}."""

    for message in reversed(history(view)):
        if role_of(message) != "agent":
            continue
        data = data_of(message)
        if isinstance(data, dict) and data.get("type") == "build":
            return data
    return None


def build_artifact(view: dict[str, Any]) -> dict[str, Any] | None:
    build = latest_build(view)
    artifact = build.get("artifact") if build else None
    return artifact if isinstance(artifact, dict) else None


def latest_spec_terms(view: dict[str, Any]) -> dict[str, Any] | None:
    spec = latest_spec(view)
    return spec["terms"] if spec else None


def own_spec_utilities(view: dict[str, Any], role: str, profile: Any) -> list[float]:
    """Utilities of our own prior spec messages (for monotonicity)."""

    utilities: list[float] = []
    for entry in spec_entries(view):
        if entry["role"] != role:
            continue
        try:
            utilities.append(float(profile.utility(entry["terms"])))
        except (ValueError, TypeError):
            continue
    return utilities


def sample_digests(
    digests: tuple[tuple[int, str], ...],
    limit: int = config.MAX_FRAME_DIGESTS_TO_LLM,
) -> list[tuple[int, str]]:
    """Evenly sample at most ``limit`` digests, always keeping the final one."""

    items = list(digests)
    if len(items) <= limit:
        return items
    if limit <= 0:
        return []
    step = max(1, len(items) // limit)
    sampled = items[::step][:limit]
    if sampled and sampled[-1] != items[-1]:
        sampled[-1] = items[-1]
    return sampled


__all__ = [
    "build_artifact",
    "data_of",
    "history",
    "last_message",
    "latest_build",
    "latest_spec",
    "latest_spec_terms",
    "next_round",
    "own_spec_utilities",
    "parts_of",
    "role_of",
    "sample_digests",
    "spec_entries",
    "state_of",
    "text_of",
]
