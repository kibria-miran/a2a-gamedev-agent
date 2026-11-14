"""A2A pin sanity: maps inverse-consistent, reasons valid (E-29, R17-R20)."""

from src import config


def test_state_map_is_a_bijection():
    assert {
        wire: internal for internal, wire in config.A2A_STATE_BY_INTERNAL.items()
    } == config.INTERNAL_STATE_BY_A2A
    for internal in config.TASK_STATES:
        assert config.A2A_STATE_BY_INTERNAL[internal].startswith("TASK_STATE_")


def test_role_map_is_a_bijection():
    assert {
        wire: internal for internal, wire in config.A2A_ROLE_BY_INTERNAL.items()
    } == config.INTERNAL_ROLE_BY_A2A


def test_reasons_for_v11_rows_present_and_short():
    for key in ("R17", "R18", "R19", "R20"):
        assert key in config.REASONS
        assert len(config.REASONS[key]) <= 60
    assert config.REASONS["R20"] == "artifact not found"


def test_all_reasons_fit_e29():
    for key, value in config.REASONS.items():
        assert isinstance(value, str) and 0 < len(value) <= 60, key


def test_jsonrpc_codes():
    assert config.JSONRPC_INVALID_PARAMS == -32602
    assert config.A2A_TASK_NOT_FOUND == -32001
    assert config.A2A_TASK_NOT_CANCELABLE == -32002


def test_terminal_states_subset_of_task_states():
    assert set(config.TERMINAL_STATES) <= set(config.TASK_STATES)
