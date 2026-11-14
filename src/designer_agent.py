"""Designer strategy: opening spec, concede, sign off; never accept (E-66).

Stateless: the task view is the whole memory, so a restarted CLI can resume
from ``tasks/get``. The Designer negotiates with the Coder, then sends
``spec_signed`` — a spec message that sets the build flag but does NOT
complete the task. Only the Critic can accept.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src import config, history, llm, negotiator, prompts, trace
from src.decision import Decision
from src.utility import Profile, load_profile

Chat = Callable[..., str]
Proposer = Callable[[dict[str, Any], int], dict[str, Any] | None]

ROLE = "designer"
ROOT = Path(__file__).resolve().parents[1]

_PROFILE_CACHE: Profile | None = None


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_designer_profile(path: str | Path | None = None) -> Profile:
    global _PROFILE_CACHE
    if path is None:
        if _PROFILE_CACHE is None:
            _PROFILE_CACHE = load_profile(_resolve(config.DESIGNER_PROFILE_PATH))
        return _PROFILE_CACHE
    return load_profile(_resolve(path))


def opening_terms(profile: Profile, brief: dict[str, Any] | None = None) -> dict[str, Any]:
    """Round 0 terms: the user's brief when it is a complete spec, else the
    profile's deterministic opening offer (E-54)."""

    if isinstance(brief, dict) and negotiator.is_canonical_terms(brief):
        return negotiator.canonical_terms(brief)
    return negotiator.fallback_offer(profile, 0, [])


def _llm_offer(
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


def next_move(
    view: dict[str, Any],
    profile: Profile | None = None,
    *,
    brief: dict[str, Any] | None = None,
    chat: Chat | None = None,
    propose: Proposer | None = None,
) -> Decision:
    """Decide the Designer's next message from the task view alone."""

    profile = profile or load_designer_profile()
    state = history.state_of(view)
    messages = history.history(view)

    if state == "completed":
        return Decision(kind="complete", rationale="the Critic accepted the build")
    if state == "rejected":
        return Decision(kind="rejected", reason="task was rejected")
    if state == "canceled":
        return Decision(kind="canceled", reason="task was canceled")
    if state == "failed":
        return Decision(kind="failed", reason="turn failed")

    # Opening offer (round 0). The brief is data; the profile is private.
    if not messages:
        terms = opening_terms(profile, brief)
        return Decision(
            kind="offer",
            terms=terms,
            round=0,
            own_u=profile.utility(terms),
            source="code",
            rationale="Opening the spec from the brief.",
        )

    last = history.last_message(view)
    spec = history.latest_spec(view)

    # Our own last spec message has no reply yet: resend it (D-17/E-16).
    if last is not None and history.role_of(last) == "user":
        data = history.data_of(last)
        if isinstance(data, dict) and data.get("type") in {"offer", "counter", "spec_signed"}:
            return Decision(kind="resend", message=last)

    if spec is None or spec["role"] != "agent":
        return Decision(kind="wait", reason="no peer counter yet")

    peer_terms = negotiator.canonical_terms(spec["terms"])
    response_round = int(spec["round"]) + 1
    own_history = history.own_spec_utilities(view, "user", profile)

    # Code decides acceptance (D-5/E-12): sign off if the peer's terms meet
    # the concession target. spec_signed is a spec message at response_round.
    if negotiator.decide_accept(peer_terms, profile, response_round):
        return Decision(
            kind="spec_signed",
            terms=peer_terms,
            round=response_round,
            own_u=profile.utility(peer_terms),
            source="code",
            rationale="Terms meet the agreed target; signing off for the build.",
        )

    if response_round > config.MAX_ROUNDS:
        return Decision(kind="walk_away", reason="max_rounds", round=response_round)

    peer_text = spec.get("text")
    if peer_text and negotiator.contains_instruction(str(peer_text)):
        trace.trace(
            "counter", direction="in", agent=ROLE, task_id=view.get("id"),
            round=int(spec["round"]), severity="error",
            error_kind="injection_attempt",
        )

    attempt: Callable[[str | None], dict[str, Any] | None] | None = None
    if propose is not None:
        attempt = lambda _reason: propose(view, response_round)  # noqa: E731
    elif chat is not None:
        attempt = lambda reason: _llm_offer(  # noqa: E731
            chat, profile, response_round, own_history, peer_terms, peer_text, reason
        )

    if attempt is not None:
        retry_reason: str | None = None
        for _ in range(config.LLM_RETRY_ATTEMPTS + 1):
            try:
                proposal = attempt(retry_reason)
            except llm.LLMError:
                proposal = None
                retry_reason = llm.REASON_LLM_ERROR
                continue
            ok, reason = negotiator.validate_proposal(
                proposal, profile, response_round, own_history
            )
            if ok and proposal is not None:
                terms = negotiator.canonical_terms(proposal["terms"])
                return Decision(
                    kind="offer",
                    terms=terms,
                    round=response_round,
                    own_u=profile.utility(terms),
                    source="llm",
                    rationale=str(proposal.get("rationale", "")),
                )
            retry_reason = reason

    terms = negotiator.fallback_offer(profile, response_round, own_history)
    return Decision(
        kind="offer",
        terms=terms,
        round=response_round,
        own_u=profile.utility(terms),
        source="fallback",
        rationale=negotiator.FALLBACK_RATIONALE,
    )


__all__ = [
    "ROLE",
    "load_designer_profile",
    "next_move",
    "opening_terms",
]
