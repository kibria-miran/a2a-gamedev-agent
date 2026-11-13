"""Tests for the reference invaders (Phase 1 step 4, X2).

The reference game is the ground truth the LLM is benchmarked against:
hand-written, -Werror clean (E-73), determinism-gated (E-74), and it must
satisfy the harness contract (E-64). If this ever breaks, the benchmark is
broken — not the test.
"""

from pathlib import Path

import pytest

from src import sandbox

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "evals" / "reference_games" / "invaders" / "main.cpp"

SPEC_SURVIVE = '{"grid_w":40,"grid_h":20,"tick_ms":100,"enemy_count":8,"win_condition":"survive"}'
SPEC_SMALL = '{"grid_w":16,"grid_h":10,"tick_ms":50,"enemy_count":30,"win_condition":"score"}'
SPEC_CLEAR = '{"grid_w":24,"grid_h":16,"tick_ms":100,"enemy_count":4,"win_condition":"clear"}'


@pytest.fixture(scope="module")
def reference_source() -> str:
    assert REFERENCE.exists(), f"reference game missing: {REFERENCE}"
    return REFERENCE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def reference_build(reference_source):
    result = sandbox.build_game({"main.cpp": reference_source}, reference=True)
    assert result.ok, result.stderr
    return result


def test_builds_with_werror(reference_build):
    # E-73: reference games compile with -Werror and it must pass clean
    assert reference_build.violations == ()
    assert reference_build.nondeterministic_source is False


def test_builds_with_llm_flags(reference_source):
    result = sandbox.build_game({"main.cpp": reference_source}, reference=False)
    assert result.ok, result.stderr


def test_determinism_gate_identical_runs(reference_build):
    # E-74: reference games build byte-identically and replay identically
    gate = sandbox.determinism_gate(
        reference_build.binary_bytes,
        tape=b"tick\n" * 20,
        nondeterministic_source=False,
        extra_env={"HARNESS_SPEC_JSON": SPEC_SURVIVE},
    )
    assert gate.ok
    assert gate.identical
    assert gate.ran_twice
    assert gate.digests_first == gate.digests_second
    assert [n for n, _ in gate.digests_first] == list(range(20))


def test_plays_survive_mode_exits_zero(reference_build):
    run = sandbox.run_game(
        reference_build.binary_bytes,
        b"key k\nkey k\ntick\n" + b"tick\n" * 20,
        extra_env={"HARNESS_SPEC_JSON": SPEC_SURVIVE},
    )
    assert run.exit_code == 0
    assert len(run.digests) == 21
    assert run.digests[0][0] == 0 and run.digests[-1][0] == 20


def test_invaders_actually_advance(reference_build):
    run = sandbox.run_game(
        reference_build.binary_bytes,
        b"tick\n" * 30,
        extra_env={"HARNESS_SPEC_JSON": SPEC_SURVIVE},
    )
    digests = [h for _, h in run.digests]
    # a static formation would produce stable digests; invaders must move
    assert len(set(digests)) > 5


def test_small_grid_clamps_without_crashing(reference_build):
    # grid_w=16 enemies=30 is outside the lattice's comfort zone: the game must
    # clamp. On a 10-row field the invaders reach the trench quickly and the
    # game ends early (invasion -> harness_exit(0)); that is correct play, so
    # we assert a clean exit and sane digest output, not a fixed digest count.
    run = sandbox.run_game(
        reference_build.binary_bytes,
        b"tick\n" * 15,
        extra_env={"HARNESS_SPEC_JSON": SPEC_SMALL},
    )
    assert run.exit_code == 0
    assert 1 <= len(run.digests) <= 15
    assert run.digests[0][0] == 0


def test_clear_mode_small_enemy_count(reference_build):
    run = sandbox.run_game(
        reference_build.binary_bytes,
        b"tick\n" * 12,
        extra_env={"HARNESS_SPEC_JSON": SPEC_CLEAR},
    )
    assert run.exit_code == 0
    assert len(run.digests) == 12


def test_reference_source_line_count(reference_source):
    # the patch calls for ~400 lines of hand-written ground truth; keep it in
    # honest range (not a one-liner, not a monster)
    lines = len(reference_source.splitlines())
    assert 150 <= lines <= 600, f"reference invaders is {lines} lines"


def test_reference_source_meets_harness_contract(reference_source):
    violations = sandbox.check_harness_contract({"main.cpp": reference_source})
    assert violations == ()
    det = sandbox.scan_determinism({"main.cpp": reference_source})
    assert det.violations == ()
    assert det.nondeterministic_source is False


def test_reference_source_calls_harness_seed(reference_source):
    # E-74: RNG must flow from harness_seed(); humans get checked too
    assert "harness_seed()" in reference_source
