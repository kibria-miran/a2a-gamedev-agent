"""Standalone Coder agent (Phase 1): spec in, binary out. No server, no A2A.

``generate_and_build`` is the whole loop: one LLM codegen call, then up to
``config.COMPILE_RETRIES`` compile attempts inside the single build turn
(E-68: exactly one build turn, never a renegotiated spec). A failed attempt
feeds the stable build log — never the LLM's rejection of it — back into a
fix prompt (E-13).

Returns ``(source, BuildResult)``; callers (Phase 4) walk away with
``build_failed`` if the result is not ok.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src import config, llm, negotiator, prompts, sandbox, trace
from src.decision import Decision
from src.utility import Profile, load_profile

logger = logging.getLogger(__name__)

Chat = Callable[..., str]
Codegen = Callable[[dict[str, Any]], tuple[str, sandbox.BuildResult]]

ROLE = "coder"
BUILD_LOG_CHARS = 6000

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE_PATH = ROOT / "evals" / "reference_games" / "invaders" / "main.cpp"

_PROFILE_CACHE: Profile | None = None


def _build_log(result: sandbox.BuildResult) -> str:
    """Stable, model-facing failure summary (E-13)."""
    parts: list[str] = []
    if result.violations:
        parts.append("validator violations:\n" + "\n".join(f"- {v}" for v in result.violations))
    if result.stderr.strip():
        parts.append(result.stderr.strip()[-BUILD_LOG_CHARS:])
    if result.exit_code is not None:
        parts.append(f"exit_code={result.exit_code}")
    return "\n".join(parts) if parts else "build produced no diagnostic output"


def _canonicalize(spec: dict[str, Any]) -> dict[str, Any]:
    """Canonical spec JSON: sorted keys, compact separators (the seed source)."""
    return json.loads(json.dumps(spec, sort_keys=True, separators=(",", ":")))


def generate_and_build(
    spec: dict[str, Any],
    chat: Chat | None = None,
) -> tuple[str, sandbox.BuildResult]:
    """Generate main.cpp from ``spec`` and build it in the sandbox.

    ``chat`` injects the LLM call (tests pass FakeLLM); when omitted, the
    real NIM client is used via ``llm.default_chat()`` and an unconfigured
    endpoint raises ``LLMError`` (E-50).
    """
    canonical = _canonicalize(spec)
    client: Chat | None = chat or llm.default_chat()
    if client is None:
        raise llm.LLMError("NIM_MODEL/NVIDIA_API_KEY not configured; cannot codegen offline")

    source = ""
    messages = prompts.build_coder_messages(canonical)
    last_result: sandbox.BuildResult | None = None

    for attempt in range(1, config.COMPILE_RETRIES + 1):
        raw = client(messages, role=ROLE, temperature=config.LLM_TEMPERATURE)
        extracted = prompts.extract_source(raw)
        if extracted is None:
            build_log = "the reply did not contain extractable C++ source"
            source = raw
            last_result = sandbox.BuildResult(ok=False, violations=("extract:no source in reply",))
            logger.info("coder attempt %d: unparseable reply", attempt)
            messages = prompts.build_fix_messages(canonical, source, build_log)
            continue
        source = extracted
        result = sandbox.build_game({"main.cpp": source})
        last_result = result
        if result.ok:
            logger.info("coder attempt %d: build ok", attempt)
            return source, result
        logger.info(
            "coder attempt %d: build failed (%s)",
            attempt,
            "; ".join(result.violations) or f"exit_code={result.exit_code}",
        )
        build_log = _build_log(result)
        messages = prompts.build_fix_messages(canonical, source, build_log)

    assert last_result is not None
    return source, last_result


# ---------------------------------------------------------------------------
# Offline oracle + the A2A responder (Phase 4)
# ---------------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_coder_profile(path: str | Path | None = None) -> Profile:
    """Load the Coder's private profile (cached when using the default path)."""

    global _PROFILE_CACHE
    if path is None:
        if _PROFILE_CACHE is None:
            _PROFILE_CACHE = load_profile(_resolve(config.CODER_PROFILE_PATH))
        return _PROFILE_CACHE
    return load_profile(_resolve(path))


def default_codegen(
    spec: dict[str, Any],
    *,
    chat: Chat | None = None,
) -> tuple[str, sandbox.BuildResult]:
    """Codegen entry point used by the service.

    * a real chat client (or a configured NIM endpoint) generates the game;
    * offline, the checked-in reference invaders source is the oracle for
      ``genre == "invaders"`` — exactly like ``src.builder_demo``;
    * anything else without an LLM raises ``LLMError`` so the Coder walks
      away instead of inventing a build.
    """

    client = chat if chat is not None else llm.default_chat()
    if client is not None:
        return generate_and_build(spec, chat=client)
    if str(spec.get("genre", "")) == "invaders":
        source = REFERENCE_SOURCE_PATH.read_text(encoding="utf-8")
        result = sandbox.build_game({"main.cpp": source})
        return source, result
    raise llm.LLMError(
        "NIM_MODEL/NVIDIA_API_KEY not configured and the offline oracle only knows invaders"
    )


def _llm_counter(
    chat: Chat,
    profile: Profile,
    round_number: int,
    own_history: list[float],
    peer_terms: dict[str, Any],
    peer_text: str | None,
    retry_reason: str | None,
) -> dict[str, Any] | None:
    raw = chat(
        prompts.build_proposer_messages(
            profile, round_number, own_history, peer_terms, peer_text, retry_reason
        ),
        role="proposer",
        temperature=config.LLM_TEMPERATURE,
    )
    return prompts.parse_proposal(raw)


def respond(
    view: dict[str, Any],
    incoming: dict[str, Any],
    own_history: list[float] | None = None,
    *,
    profile: Profile | None = None,
    chat: Chat | None = None,
    codegen: Codegen | None = None,
) -> Decision:
    """One Coder turn: counter a spec message, or build after ``spec_signed``.

    This is the responder plugged into the service. It is synchronous (the
    server runs it in a thread) and stateless: all history comes from the
    task view. The build runs exactly once per task, after signoff (E-66/68).
    """

    profile = profile or load_coder_profile()
    kind = incoming.get("type")
    round_number = int(incoming.get("round", 0) or 0)
    history_values = list(own_history or [])

    if kind in {"offer", "counter"}:
        peer_terms = incoming.get("terms")
        if not isinstance(peer_terms, dict) or not negotiator.is_canonical_terms(peer_terms):
            return Decision(kind="walk_away", reason="bad_terms", round=round_number + 1)
        peer_terms = negotiator.canonical_terms(peer_terms)
        response_round = round_number + 1

        # The Coder never terminal-accepts: if the peer's spec is acceptable,
        # it counters with exactly those terms, and the Designer signs off
        # (E-66: only the Designer signs, only the Critic accepts).
        if negotiator.decide_accept(peer_terms, profile, response_round):
            return Decision(
                kind="counter",
                terms=peer_terms,
                round=response_round,
                own_u=profile.utility(peer_terms),
                source="code",
                rationale="These terms are acceptable; confirming them for signoff.",
            )

        if response_round > config.MAX_ROUNDS:
            return Decision(
                kind="walk_away", reason="max_rounds", round=response_round
            )

        peer_text = incoming.get("_peer_text")
        if peer_text and negotiator.contains_instruction(str(peer_text)):
            trace.trace(
                "counter", direction="in", agent=ROLE,
                task_id=view.get("id"), round=round_number,
                severity="error", error_kind="injection_attempt",
            )

        if chat is not None:
            retry_reason: str | None = None
            for _ in range(config.LLM_RETRY_ATTEMPTS + 1):
                try:
                    proposal = _llm_counter(
                        chat, profile, response_round, history_values,
                        peer_terms, peer_text, retry_reason,
                    )
                except llm.LLMError:
                    proposal = None
                    retry_reason = llm.REASON_LLM_ERROR
                    continue
                ok, reason = negotiator.validate_proposal(
                    proposal, profile, response_round, history_values
                )
                if ok and proposal is not None:
                    terms = negotiator.canonical_terms(proposal["terms"])
                    return Decision(
                        kind="counter",
                        terms=terms,
                        round=response_round,
                        own_u=profile.utility(terms),
                        source="llm",
                        rationale=str(proposal.get("rationale", "")),
                    )
                retry_reason = reason

        terms = negotiator.fallback_offer(profile, response_round, history_values)
        return Decision(
            kind="counter",
            terms=terms,
            round=response_round,
            own_u=profile.utility(terms),
            source="fallback",
            rationale=negotiator.FALLBACK_RATIONALE,
        )

    if kind == "spec_signed":
        signed_terms = incoming.get("terms")
        if not isinstance(signed_terms, dict) or not negotiator.is_canonical_terms(signed_terms):
            return Decision(kind="walk_away", reason="bad_terms", round=round_number + 1)
        signed = negotiator.canonical_terms(signed_terms)
        build_round = round_number + 1
        try:
            _source, result = (codegen or default_codegen)(signed)
        except llm.LLMError:
            return Decision(
                kind="walk_away", reason="llm_error", round=build_round, source="code"
            )
        if not result.ok:
            reason = "sandbox_violation" if result.sandbox_violation else "build_failed"
            return Decision(
                kind="walk_away", reason=reason, round=build_round, source="code"
            )
        return Decision(
            kind="build",
            build_result=result,
            round=build_round,
            source="code",
            rationale="Built once after signoff (E-68).",
        )

    return Decision(kind="walk_away", reason="unexpected_message", round=round_number)


def make_responder(
    *,
    profile: Profile | None = None,
    chat: Chat | None = None,
    codegen: Codegen | None = None,
) -> Callable[..., Decision]:
    """Bind a Coder profile/codegen into the server's responder signature."""

    resolved = profile or load_coder_profile()

    def responder(view: dict[str, Any], incoming: dict[str, Any], own: list[float]) -> Decision:
        return respond(view, incoming, own, profile=resolved, chat=chat, codegen=codegen)

    return responder


__all__ = [
    "Codegen",
    "ROLE",
    "default_codegen",
    "generate_and_build",
    "load_coder_profile",
    "make_responder",
    "respond",
]
