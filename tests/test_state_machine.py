"""Tests for the frozen state machine (T1-T8; unchanged in v1.1)."""

import pytest

from src import state_machine

DEMO_EDGES = [
    # T1
    ("submitted", "working", True),
    # T2
    ("submitted", "rejected", True),
    ("submitted", "canceled", True),
    ("submitted", "failed", True),
    # T3
    ("working", "input-required", True),
    # T4
    ("working", "completed", True),
    # T5
    ("working", "rejected", True),
    ("working", "canceled", True),
    ("working", "failed", True),
    # T6
    ("input-required", "working", True),
    # T7
    ("input-required", "canceled", True),
    ("input-required", "failed", True),
    # T8 recovery only
    ("working", "input-required", True),
]

ILLEGAL = [
    ("submitted", "completed"),
    ("submitted", "input-required"),
    ("working", "submitted"),
    ("input-required", "completed"),
    ("input-required", "submitted"),
    ("completed", "working"),
    ("rejected", "working"),
    ("failed", "completed"),
    ("canceled", "working"),
    # T1 reverse
    ("working", "submitted"),
    ("rejected", "input-required"),
]


def test_T1_to_T7_base_edges():
    base = [
        ("submitted", "working"),
        ("submitted", "rejected"),
        ("submitted", "canceled"),
        ("submitted", "failed"),
        ("working", "input-required"),
        ("working", "completed"),
        ("working", "rejected"),
        ("working", "canceled"),
        ("working", "failed"),
        ("input-required", "working"),
        ("input-required", "canceled"),
        ("input-required", "failed"),
    ]
    for from_state, to_state in base:
        assert state_machine.can_transition(from_state, to_state), f"{from_state} -> {to_state}"
        assert state_machine.transition(from_state, to_state) == to_state


def test_T3_is_base_edge_but_submitted_to_input_required_needs_recovery():
    # T3: working -> input-required is a base edge (normal turn end)
    assert state_machine.can_transition("working", "input-required")
    # T8: submitted -> input-required only via startup recovery
    assert not state_machine.can_transition("submitted", "input-required")
    assert state_machine.can_transition("submitted", "input-required", via_recovery=True)
    assert state_machine.can_transition("working", "input-required", via_recovery=True)


@pytest.mark.parametrize(("from_state", "to_state"), ILLEGAL)
def test_illegal_transitions_raise(from_state, to_state):
    assert not state_machine.can_transition(from_state, to_state)
    with pytest.raises(state_machine.StateError):
        state_machine.transition(from_state, to_state)


def test_terminal_states_are_terminal():
    for state in ("completed", "canceled", "failed", "rejected"):
        assert state_machine.is_terminal(state)
        for to_state in ("submitted", "working", "input-required"):
            assert not state_machine.can_transition(state, to_state)
