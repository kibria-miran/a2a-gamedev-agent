"""Pure negotiation logic for the game spec: lattice fallback, validators.

No I/O. The LLM never decides; this module is the code that does (D-5/E-12).
The negotiable terms are the frozen spec in :mod:`src.protocol`: four numeric
issues and three categorical ones. The lattice is the v1.1 fallback space
(E-71):

    4 genres x 8 grid_w x 8 grid_h x 6 tick_ms x 8 enemy_count
    x 3 win_condition = 36,864 candidates

``reliability`` is held at the profile's preferred value while searching, so
it does not multiply the lattice; the plan fixes the count at 36,864 and
``tests/test_negotiator.py::test_lattice_exhaustive`` pins it.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from typing import Any

from src import config
from src.utility import ISSUES, Profile

# ---------------------------------------------------------------------------
# Stable validator reasons (E-29, E-25)
# ---------------------------------------------------------------------------

REASON_V1_SCHEMA = "V1 schema invalid"
REASON_V2_INFEASIBLE = "V2 infeasible terms"
REASON_V3_BAND = "V3 outside concession band"
REASON_V4_MONOTONE = "V4 non-monotone utility"
REASON_V5_LEAK = "V5 private value leak"
REASON_V6_INSTRUCTION = "V6 instruction phrase"
REASON_V7_MARKUP = "V7 markup token"

FALLBACK_RATIONALE = "Adjusted counter-offer within the agreed scope."

LEAK_PHRASES = (
    "my minimum",
    "my reservation",
    "my limit",
    "my bottom line",
    "lowest i can",
)

INSTRUCTION_PHRASES = (
    "ignore",
    "disregard",
    "override",
    "forget",
    "system prompt",
    "you are now",
    "new instructions",
    "you must accept",
    "reveal your",
    "as an ai",
)

MARKUP_TOKENS = ("</", "<peer", "```", "###")

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def normalize_text(text: str) -> str:
    """NFKC, strip zero-width chars, lowercase (E-14)."""

    return unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).lower()


def _strip_thousands(text: str) -> str:
    return re.sub(r"(?<=\d)[,\u066c\u00a0](?=\d)", "", text)


def contains_leak(text: str, profile: Profile) -> bool:
    """True if the text names a private reservation/limit phrase.

    Game specs are not secret values; the leak probe is phrase-based so a
    legitimate rationale may still mention "40x20".
    """

    normalized = _strip_thousands(normalize_text(text))
    return any(phrase in normalized for phrase in LEAK_PHRASES)


def contains_instruction(text: str) -> bool:
    normalized = normalize_text(text)
    return any(phrase in normalized for phrase in INSTRUCTION_PHRASES)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_canonical_terms(terms: Any) -> bool:
    """Cheap schema check for a terms object (V1's first half)."""

    if not isinstance(terms, dict):
        return False
    for name in ("grid_w", "grid_h", "tick_ms", "enemy_count"):
        if not _is_int(terms.get(name)):
            return False
    genre = terms.get("genre")
    win_condition = terms.get("win_condition")
    if not isinstance(genre, str) or genre not in config.GENRE_UTILITY:
        return False
    if not isinstance(win_condition, str) or win_condition not in config.WIN_CONDITION_UTILITY:
        return False
    reliability = terms.get("reliability", "deterministic")
    return isinstance(reliability, str) and reliability in config.RELIABILITY_UTILITY


def canonical_terms(terms: dict[str, Any]) -> dict[str, Any]:
    """Project negotiated terms onto the frozen schema (drop free text)."""

    return {
        "genre": str(terms["genre"]),
        "grid_w": int(terms["grid_w"]),
        "grid_h": int(terms["grid_h"]),
        "tick_ms": int(terms["tick_ms"]),
        "enemy_count": int(terms["enemy_count"]),
        "win_condition": str(terms["win_condition"]),
        "reliability": str(terms.get("reliability", "deterministic")),
    }


# ---------------------------------------------------------------------------
# Fallback lattice (E-71)
# ---------------------------------------------------------------------------


def iter_lattice() -> Any:
    """Yield every raw lattice combination, before feasibility filtering."""

    for genre, grid_w, grid_h, tick_ms, enemy_count, win_condition in itertools.product(
        config.GENRES,
        config.GRID_W_LATTICE,
        config.GRID_H_LATTICE,
        config.TICK_MS_LATTICE,
        config.ENEMY_COUNT_LATTICE,
        config.WIN_CONDITIONS,
    ):
        yield {
            "genre": genre,
            "grid_w": grid_w,
            "grid_h": grid_h,
            "tick_ms": tick_ms,
            "enemy_count": enemy_count,
            "win_condition": win_condition,
        }


def lattice_size() -> int:
    """The frozen E-71 count: 36,864."""

    return config.LATTICE_SIZE


# Cached feasible lattice as immutable tuples: (terms, utility).
_LATTICE_CACHE: dict[tuple[Any, ...], list[tuple[dict[str, Any], float]]] = {}


def _feasible_lattice(profile: Profile) -> list[tuple[dict[str, Any], float]]:
    key = profile.key()
    cached = _LATTICE_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_LATTICE_CACHE) > 16:
        _LATTICE_CACHE.clear()

    reliability = profile.best_categorical("reliability")
    candidates: list[tuple[dict[str, Any], float]] = []
    for raw in iter_lattice():
        terms = {**raw, "reliability": reliability}
        if not profile.feasible(terms):
            continue
        candidates.append((terms, profile.utility(terms)))
    if not candidates:
        raise ValueError("profile has no feasible candidates")
    _LATTICE_CACHE[key] = candidates
    return candidates


def _ideal_distance(terms: dict[str, Any], profile: Profile) -> float:
    distance = 0.0
    for name, issue in profile.issues.items():
        span = abs(issue.ideal - issue.reservation) or 1
        distance += abs(int(terms[name]) - issue.ideal) / span
    return distance


def _order_key(item: tuple[dict[str, Any], float]) -> tuple[Any, ...]:
    terms, _utility = item
    return (
        terms["genre"],
        terms["grid_w"],
        terms["grid_h"],
        terms["tick_ms"],
        terms["enemy_count"],
        terms["win_condition"],
    )


def fallback_offer(
    profile: Profile,
    round_number: int,
    own_history: list[float] | None = None,
) -> dict[str, Any]:
    """Deterministic offer closest to the concession target (E-1).

    With ``own_history == []`` the monotone filter is skipped (E-54).
    """

    history = list(own_history or [])
    target = profile.target_utility(round_number)
    feasible = _feasible_lattice(profile)

    monotone = feasible
    if history:
        ceiling = min(history) + config.EPS_MONOTONE
        monotone = [item for item in feasible if item[1] <= ceiling]
        if not monotone:
            # Nothing satisfies monotonicity: fall back to the lowest U.
            terms, _utility = min(feasible, key=lambda item: (item[1], _order_key(item)))
            return canonical_terms(terms)

    terms, _utility = min(
        monotone,
        key=lambda item: (
            abs(item[1] - target),
            _ideal_distance(item[0], profile),
            _order_key(item),
        ),
    )
    return canonical_terms(terms)


# ---------------------------------------------------------------------------
# Validators V1..V7 (first failure wins)
# ---------------------------------------------------------------------------


def _schema_ok(proposal: Any) -> bool:
    if not isinstance(proposal, dict):
        return False
    if not is_canonical_terms(proposal.get("terms")):
        return False
    rationale = proposal.get("rationale", "")
    return isinstance(rationale, str) and len(rationale) <= config.MAX_RATIONALE_CHARS


def validate_proposal(
    proposal: Any,
    profile: Profile,
    round_number: int,
    own_history: list[float] | None = None,
) -> tuple[bool, str | None]:
    if not _schema_ok(proposal):
        return False, REASON_V1_SCHEMA

    terms = canonical_terms(proposal["terms"])
    rationale = str(proposal.get("rationale", ""))

    if not profile.feasible(terms):
        return False, REASON_V2_INFEASIBLE

    target = profile.target_utility(round_number)
    utility = profile.utility(terms)
    if not (
        target - config.BAND_LOW - config.EPS_BAND
        <= utility
        <= target + config.BAND_HIGH + config.EPS_BAND
    ):
        return False, REASON_V3_BAND

    history = list(own_history or [])
    if history and utility > min(history) + config.EPS_MONOTONE:
        return False, REASON_V4_MONOTONE

    if contains_leak(rationale, profile):
        return False, REASON_V5_LEAK

    if contains_instruction(rationale):
        return False, REASON_V6_INSTRUCTION

    if any(token in rationale for token in MARKUP_TOKENS):
        return False, REASON_V7_MARKUP

    return True, None


# ---------------------------------------------------------------------------
# Code decides acceptance (D-5, E-12)
# ---------------------------------------------------------------------------


def decide_accept(peer_terms: dict[str, Any], profile: Profile, round_number: int) -> bool:
    try:
        if not profile.feasible(peer_terms):
            return False
        utility = profile.utility(peer_terms)
    except (ValueError, TypeError):
        return False
    return utility >= profile.target_utility(round_number) - config.EPS_ACCEPT


def target_utility(profile: Profile, round_number: int) -> float:
    return profile.target_utility(round_number)


__all__ = [
    "FALLBACK_RATIONALE",
    "INSTRUCTION_PHRASES",
    "ISSUES",
    "LEAK_PHRASES",
    "MARKUP_TOKENS",
    "REASON_V1_SCHEMA",
    "REASON_V2_INFEASIBLE",
    "REASON_V3_BAND",
    "REASON_V4_MONOTONE",
    "REASON_V5_LEAK",
    "REASON_V6_INSTRUCTION",
    "REASON_V7_MARKUP",
    "canonical_terms",
    "contains_instruction",
    "contains_leak",
    "decide_accept",
    "fallback_offer",
    "is_canonical_terms",
    "iter_lattice",
    "lattice_size",
    "normalize_text",
    "target_utility",
    "validate_proposal",
]
