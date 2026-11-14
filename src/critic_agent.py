"""Critic strategy: fetch the binary, verify it, play it, accept or reject.

This is the only role that can accept (E-66). The Critic is a separate
process from the Coder, so it never trusts a local path: it fetches the
artifact over the R20 route, recomputes sha256, and rejects on mismatch
(E-70/S25/S27). The review is evidence-driven: exit code 0, at least one
parsed frame digest, and two independent runs that agree (E-74).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from src import client as client_mod
from src import config, history, sandbox
from src.decision import Decision

ROLE = "critic"

Runner = Callable[..., sandbox.RunResult]
Gate = Callable[..., sandbox.DeterminismGate]


def _artifact_id(artifact: dict[str, Any]) -> str | None:
    value = artifact.get("artifactId") or artifact.get("artifact_id")
    return str(value) if value else None


async def review(
    view: dict[str, Any],
    client: Any,
    *,
    tape: bytes | None = None,
    run_game: Runner = sandbox.run_game,
    gate: Gate = sandbox.determinism_gate,
) -> Decision:
    """Review the latest build and return ``accept`` or ``walk_away``."""

    state = history.state_of(view)
    if state == "completed":
        return Decision(kind="complete", rationale="already accepted")
    if state in {"rejected", "canceled", "failed"}:
        return Decision(kind="rejected", reason=f"task is {state}")

    artifact = history.build_artifact(view)
    if artifact is None:
        return Decision(kind="walk_away", reason="no_build")
    artifact_id = _artifact_id(artifact)
    expected_sha = str(artifact.get("sha256", ""))
    if not artifact_id or not expected_sha:
        return Decision(kind="walk_away", reason="bad_artifact_descriptor")

    terms = history.latest_spec_terms(view)
    if terms is None:
        return Decision(kind="walk_away", reason="no_signed_spec")
    canonical = json.dumps(terms, sort_keys=True, separators=(",", ":"))

    task_id = str(view.get("id", ""))
    try:
        blob, sha256, size = await client.fetch_build(task_id, artifact_id)
    except client_mod.ShaMismatch:
        return Decision(kind="walk_away", reason="sha256_mismatch")
    except client_mod.A2AClientError:
        return Decision(kind="walk_away", reason="artifact_unavailable")
    except client_mod.AuthError:
        return Decision(kind="walk_away", reason="artifact_unauthorized")
    except client_mod.SendFailed:
        return Decision(kind="walk_away", reason="artifact_unavailable")

    if sha256 != expected_sha:
        return Decision(kind="walk_away", reason="sha256_mismatch")
    if int(artifact.get("size", size)) != size:
        return Decision(kind="walk_away", reason="artifact_size_mismatch")

    extra_env = {"HARNESS_SPEC_JSON": canonical, "HARNESS_PLAY_TIMEOUT": "10"}
    played = run_game(blob, tape or config.CRITIC_TAPE, extra_env=extra_env)
    if played.sandbox_violation:
        return Decision(kind="walk_away", reason="sandbox_violation")
    if played.exit_code != 0:
        return Decision(kind="walk_away", reason="play_failed")
    if not played.digests:
        return Decision(kind="walk_away", reason="no_frame_digests")

    checked = gate(
        blob,
        tape=tape or config.CRITIC_TAPE,
        nondeterministic_source=False,
        extra_env=extra_env,
    )
    if not checked.ok:
        return Decision(kind="walk_away", reason="nondeterministic")

    build = history.latest_build(view) or {}
    review_round = int(build.get("round", history.next_round(view)) or 0)
    final_tick, final_digest = played.digests[-1]
    sampled = history.sample_digests(played.digests)
    return Decision(
        kind="accept",
        round=review_round,
        source="code",
        rationale=(
            f"Verified sha256 and played {len(played.digests)} frames "
            f"({len(sampled)} sampled); final tick {final_tick} {final_digest}."
        ),
        message={
            "artifact": {
                "artifactId": artifact_id,
                "sha256": sha256,
                "size": size,
            }
        },
    )


__all__ = ["ROLE", "review"]
