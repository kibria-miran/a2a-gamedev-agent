"""Borderline-infeasible profile pair: negotiation walks away (v1.1 roadmap
item 3). The two sides' feasible lattices are disjoint, so no agreement is
possible; the demo must reject gracefully — without a build, without
crashing, and with the evals-runner contract (agreement < 1.0) intact.
"""

from __future__ import annotations

import pytest

from src import history
from src.three_agent_demo import run_demo
from src.utility import load_profile


@pytest.fixture(scope="module")
def infeasible_designer():
    return load_profile("evals/infeasible_designer.json")


@pytest.fixture(scope="module")
def infeasible_coder():
    return load_profile("evals/infeasible_coder.json")


async def test_infeasible_pair_walks_away_gracefully(infeasible_designer, infeasible_coder) -> None:
    def forbidden_codegen(_spec):
        raise AssertionError("the build must never run for an infeasible pair")

    result = await run_demo(
        db_path=":memory:",
        brief_path=None,  # opening comes from the designer's private fallback
        designer_profile=infeasible_designer,
        coder_profile=infeasible_coder,
        codegen=forbidden_codegen,
        verbose=False,
    )

    task = result["task"]
    assert task["status"]["state"] == "TASK_STATE_REJECTED"
    assert result["decision"].kind == "rejected"
    assert result["artifact"] is None
    assert result["binary"] == b""

    types = [(history.data_of(m) or {}).get("type") for m in history.history(task)]
    # pure negotiation, then the terminal walk_away — and never a build
    assert "build" not in types
    assert types[-1] == "walk_away"
    assert set(types) <= {"offer", "counter", "walk_away"}
