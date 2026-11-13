"""Task lifecycle transitions (section 8.4, T1-T8; unchanged in v1.1).

E-66 adds message-layer decision rows (R17-R20), not state-layer ones; the
state machine itself is untouched by v1.1.
"""

from __future__ import annotations

from src import config


class StateError(Exception):
    """Raised for any transition not in the frozen table."""


_BASE_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # T1
        ("submitted", "working"),
        # T2
        ("submitted", "rejected"),
        ("submitted", "canceled"),
        ("submitted", "failed"),
        # T3
        ("working", "input-required"),
        # T4
        ("working", "completed"),
        # T5
        ("working", "rejected"),
        ("working", "canceled"),
        ("working", "failed"),
        # T6
        ("input-required", "working"),
        # T7
        ("input-required", "canceled"),
        ("input-required", "failed"),
    }
)


def is_terminal(state: str) -> bool:
    return state in config.TERMINAL_STATES


def can_transition(from_state: str, to_state: str, *, via_recovery: bool = False) -> bool:
    if from_state not in config.TASK_STATES or to_state not in config.TASK_STATES:
        return False
    if is_terminal(from_state):
        return False
    if (from_state, to_state) in _BASE_TRANSITIONS:
        return True
    # T8: submitted|working -> input-required only via startup recovery.
    return bool(
        via_recovery
        and to_state == "input-required"
        and from_state in {"submitted", "working"}
    )


def transition(from_state: str, to_state: str, *, via_recovery: bool = False) -> str:
    if not can_transition(from_state, to_state, via_recovery=via_recovery):
        raise StateError(f"illegal transition {from_state!r} -> {to_state!r}")
    return to_state
