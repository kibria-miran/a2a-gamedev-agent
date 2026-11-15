"""Offline guards for the X1-X8 experiment runner.

These exercise the same code paths the CI runs (offline fallback mode)
with the reference binary prebuilt once per module, so nothing here
touches the network. The live arms are skipped unless explicitly
requested — they need NIM credentials.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from evals import run_evals
from evals.run_evals import MissingCredentials, run_one, run_x8

from src import config

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


TINY_GAME = """
#include "harness.h"
#include <cstdio>
namespace { int x = 0; }
int harness_init(void) { return 0; }
void harness_tick(void) { ++x; }
const char* harness_render(void) {
    static char buf[32];
    std::snprintf(buf, sizeof buf, "frame=%d", x);
    return buf;
}
"""


@pytest.fixture(scope="module", autouse=True)
def cached_build():  # noqa: ARG001
    """Compile a tiny game ONCE for every evals test (fast -O2 build).

    Building the reference invaders per test would put ~20 s of g++ into
    the fast tier; the runner machinery is identical either way.
    """

    from src import sandbox

    result = sandbox.build_game({"main.cpp": TINY_GAME})
    assert result.ok, result.stderr
    run_evals._CACHED_BUILD = result
    return result


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(run_evals, "RESULTS", tmp_path)
    yield tmp_path


def test_run_one_fallback_records_metrics() -> None:
    result = asyncio.run(run_one("x8", "fallback", "t1"))

    assert result.outcome == "completed"
    assert result.builds == 1  # E-68: exactly one build per task
    assert result.mode == "fallback"
    assert result.tokens_in == 0 and result.tokens_out == 0  # no LLM offline
    assert result.designer_u is not None and 0.9 < result.designer_u <= 1.0
    assert result.coder_u is not None and 0.9 < result.coder_u <= 1.0
    assert result.joint_u is not None and result.joint_u > 1.8
    assert result.rounds == 3  # offer + counter + spec_signed


def test_x1_grid_completes_all_cells(results_dir: Path) -> None:
    results = run_evals.run_x1("fallback", 1)

    assert len(results) == 9
    assert all(r.outcome == "completed" for r in results)
    out = results_dir / "x1-fallback.jsonl"
    assert out.exists()
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(records) == 9
    assert {r["seed"]["designer_overrides"]["beta"] for r in records} == {0.5, 1.0, 3.0}


def test_x4_and_x6_complete(tmp_path: Path) -> None:
    x4 = run_evals.run_x4("fallback", 1)
    assert [r.seed["max_rounds"] for r in x4] == [4, 8, 12]
    assert all(r.outcome == "completed" for r in x4)
    x6 = run_evals.run_x6("fallback", 1)
    assert [r.seed["band_low"] for r in x6] == [0.03, 0.05, 0.10]


def test_x7_transcripts_are_identical(results_dir: Path) -> None:
    results = run_evals.run_x7("fallback", 3)

    hashes = {r.transcript_hash for r in results}
    assert len(hashes) == 1
    assert (results_dir / "x7-fallback.jsonl").exists()


def test_x8_one_build_audit_passes(results_dir: Path) -> None:
    results = run_x8("fallback", 2)

    completed = [r for r in results if r.outcome == "completed"]
    assert completed and all(r.builds == 1 for r in completed)


def test_live_arm_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(config, "NIM_MODEL", "")
    with pytest.raises(MissingCredentials):
        asyncio.run(run_one("x2", "live", "t-live"))


def test_x2_refuses_fallback_mode() -> None:
    with pytest.raises(SystemExit):
        run_evals.run_x2("fallback", 1)


def test_wire_leaks_are_zero_offline(results_dir: Path) -> None:
    results = run_evals.run_x8("fallback", 1)
    assert all(r.wire_leaks == 0 for r in results)


def test_count_rationale_leaks_counts_wire_rationales() -> None:
    task = {
        "history": [
            {"parts": [{"data": {"type": "counter", "round": 1,
                                "rationale": "my reservation is 700; take it or leave it"}}]},
            {"parts": [{"data": {"type": "counter", "round": 3,
                                "rationale": "Adjusted counter-offer within the agreed scope."}}]},
            {"parts": [{"data": {"type": "offer", "round": 5}}]},  # no rationale
            {"parts": [{"text": "not data at all"}]},
        ],
    }
    assert run_evals._count_rationale_leaks(task) == 1
    assert run_evals._count_rationale_leaks({"history": []}) == 0
    assert run_evals._count_rationale_leaks({}) == 0


def test_summarize_reports_per_run_leaks(capsys: pytest.CaptureFixture[str]) -> None:
    from evals.run_evals import RunResult, summarize

    results = [
        RunResult(experiment="x9", mode="fallback", run_id=f"run-{i}", outcome="completed",
                  terms=None, rounds=1, designer_u=0.5, coder_u=0.5, joint_u=1.0,
                  nash_product=0.25, surplus_fairness=0.5, builds=1,
                  latency_s=0.1, source="fallback", wire_leaks=wire)
        for i, wire in enumerate((0, 2, 0, 1))
    ]
    summarize("X9 synthetic", results)
    out = capsys.readouterr().out
    assert "wire leaks: 3 over 4 runs" in out
    assert "worst single run: 2" in out
    assert "leaking runs: run-1, run-3" in out
