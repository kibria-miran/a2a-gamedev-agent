#!/usr/bin/env python3
"""Machine-checked experiments for the three-agent game dev loop.

Two modes:
  fallback  deterministic code-decided negotiation on both sides; the build
            uses the checked-in reference invaders (compiled once up front).
            Fully offline — this is what CI runs.
  live      LLM proposer on both sides (Designer and Coder use the NIM
            endpoint from .env; each X-experiment below documents whether it
            supports live). Requires NVIDIA_API_KEY and NIM_MODEL.

X1-X6 use evals/stress_brief.json (a deliberately hostile brief that forces
genuine negotiation; the reference brief is acceptable to both sides at round 0,
so those cells would compare nothing). X7/X8 use the reference brief.

Usage:
  .venv/bin/python evals/run_evals.py --experiment x1 --runs 10
  .venv/bin/python evals/run_evals.py --experiment x4 --runs 10
  .venv/bin/python evals/run_evals.py --experiment x2 --mode live --runs 10
  .venv/bin/python evals/run_evals.py --all-offline --runs 10

Experiments:
  x1  Beta grid (3.0 / 1.0 / 0.5) x 3x3 — designer_beta x coder_beta (fallback)
  x2  LLM proposer on both sides vs the pure code-decided fallback (live only)
  x3  Coder rationale on the wire vs suppressed — leak check + agreement (live only)
  x4  MAX_ROUNDS 4 / 8 / 12 (fallback)
  x5  One model on both sides vs a different model per side (live only)
  x6  BAND_LOW/BAND_HIGH 0.03 / 0.05 / 0.10 (fallback; see note printed)
  x7  Reproducibility — the same inputs produce the identical transcript (fallback)
  x8  One-build audit — exactly one build per completed task; p50/p95 latency

X2/X3/X5 write combined files (x2-compare, x3-rationale, x5-models) so both
arms of each comparison sit side by side. Per-run token counts are recorded.
Raw per-run lines go to evals/results/<experiment>-<mode>.jsonl.
Results are intended to be summarized in docs/WRITE_UP.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

from src import config, history, llm, negotiator, sandbox  # noqa: E402
from src.three_agent_demo import run_demo  # noqa: E402
from src.utility import Profile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"

DESIGNER_BASE: dict[str, Any] = json.loads((ROOT / "profiles" / "designer.json").read_text())
CODER_BASE: dict[str, Any] = json.loads((ROOT / "profiles" / "coder.json").read_text())
REFERENCE_CPP = ROOT / "evals" / "reference_games" / "invaders" / "main.cpp"
STRESS_BRIEF = ROOT / "evals" / "stress_brief.json"
REFERENCE_BRIEF = ROOT / "evals" / "reference_brief.json"


def make_profile(base: dict[str, Any], **overrides: Any) -> Profile:
    data = json.loads(json.dumps(base))
    data.update(overrides)
    return Profile.load(data)


class MissingCredentials(RuntimeError):
    """The live arm was requested without a usable NIM configuration."""


_CACHED_BUILD: sandbox.BuildResult | None = None


def cached_codegen():
    """Compile the reference invaders ONCE; every run reuses the binary.

    Compiling per run would put ~20 s of g++ into every negotiation cell.
    """

    global _CACHED_BUILD
    if _CACHED_BUILD is None:
        print("compiling the reference invaders once for this runner...", flush=True)
        source = REFERENCE_CPP.read_text(encoding="utf-8")
        result = sandbox.build_game({"main.cpp": source}, reference=True)
        if not result.ok:
            raise RuntimeError(
                f"reference invaders failed to build: {result.stderr[-2000:]}"
            )
        _CACHED_BUILD = result
    cached = _CACHED_BUILD
    source = REFERENCE_CPP.read_text(encoding="utf-8")

    def codegen(spec: dict[str, Any]) -> tuple[str, sandbox.BuildResult]:
        assert cached.ok and cached.binary_bytes
        return source, cached

    return codegen


@dataclass
class RunResult:
    experiment: str
    mode: str
    run_id: str
    outcome: str
    terms: dict[str, Any] | None
    rounds: int
    designer_u: float | None
    coder_u: float | None
    joint_u: float | None
    nash_product: float | None
    surplus_fairness: float | None
    builds: int
    latency_s: float
    source: str
    tokens_in: int = 0
    tokens_out: int = 0
    transcript_hash: str = ""
    wire_leaks: int = 0
    seed: dict[str, Any] = field(default_factory=dict)


def _outcome_of(result: dict[str, Any]) -> str:
    state = str(result["task"]["status"]["state"])
    return state.replace("TASK_STATE_", "").lower()


def _spec_messages(task: dict[str, Any]) -> list[dict[str, Any]]:
    messages = task.get("history") or []
    spec = []
    for message in messages:
        data = history.data_of(message)
        if isinstance(data, dict) and data.get("type") in {"offer", "counter", "spec_signed"}:
            spec.append(data)
    return spec


def _build_count(task: dict[str, Any]) -> int:
    messages = task.get("history") or []
    return sum(
        1
        for message in messages
        if (history.data_of(message) or {}).get("type") == "build"
    )


def _scrub_ids(value: Any) -> Any:
    """Replace random descriptors with stable placeholders.

    The server mints a fresh artifactId (UUID) per build; that is noise for
    reproducibility purposes, not signal.
    """

    if isinstance(value, dict):
        return {
            key: ("<artifactId>" if key == "artifactId" else _scrub_ids(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub_ids(item) for item in value]
    return value


def _transcript_hash(task: dict[str, Any]) -> str:
    import hashlib

    parts = []
    for message in task.get("history") or []:
        data = history.data_of(message)
        if isinstance(data, dict):
            blob = json.dumps(_scrub_ids(data), sort_keys=True, separators=(",", ":"))
            parts.append(f"{history.role_of(message)}:{blob}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _count_rationale_leaks(task: dict[str, Any]) -> int:
    """Number of wire rationales containing a private-value leak phrase."""

    leaks = 0
    for message in task.get("history") or []:
        for part in message.get("parts") or []:
            data = part.get("data")
            if isinstance(data, dict):
                rationale = data.get("rationale")
                if isinstance(rationale, str) and negotiator.contains_leak(rationale, None):
                    leaks += 1
    return leaks


async def run_one(
    experiment: str,
    mode: str,
    run_id: str,
    *,
    designer_overrides: dict[str, Any] | None = None,
    coder_overrides: dict[str, Any] | None = None,
    max_rounds: int | None = None,
    band_low: float | None = None,
    band_high: float | None = None,
    designer_chat: Any | None = None,
    coder_chat: Any | None = None,
    include_rationale: bool = False,
    brief_path: Path | str | None = None,
) -> RunResult:
    designer = make_profile(DESIGNER_BASE, **(designer_overrides or {}))
    coder = make_profile(CODER_BASE, **(coder_overrides or {}))

    old_max, old_low, old_high = config.MAX_ROUNDS, config.BAND_LOW, config.BAND_HIGH
    if max_rounds is not None:
        config.MAX_ROUNDS = max_rounds
    if band_low is not None:
        config.BAND_LOW = band_low
    if band_high is not None:
        config.BAND_HIGH = band_high

    if mode == "fallback":
        designer_chat = coder_chat = None
    elif mode == "live":
        designer_chat = designer_chat if designer_chat is not None else llm.default_chat()
        coder_chat = coder_chat if coder_chat is not None else llm.default_chat()
        if designer_chat is None or coder_chat is None:
            (config.MAX_ROUNDS, config.BAND_LOW, config.BAND_HIGH) = old_max, old_low, old_high
            raise MissingCredentials(
                "live mode needs NVIDIA_API_KEY and NIM_MODEL (copy .env.example to .env)"
            )
    else:
        (config.MAX_ROUNDS, config.BAND_LOW, config.BAND_HIGH) = old_max, old_low, old_high
        raise AssertionError(mode)

    codegen = cached_codegen()
    tokens_before = llm.token_usage_all()
    started = time.perf_counter()
    try:
        result = await run_demo(
            db_path=":memory:",
            brief_path=brief_path or REFERENCE_BRIEF,
            verbose=False,
            designer_profile=designer,
            coder_profile=coder,
            codegen=codegen,
            designer_chat=designer_chat,
            coder_chat=coder_chat,
            include_rationale=include_rationale,
        )
    finally:
        latency = time.perf_counter() - started
        config.MAX_ROUNDS, config.BAND_LOW, config.BAND_HIGH = old_max, old_low, old_high
    tokens_in, tokens_out = llm.usage_delta(tokens_before)

    task = result["task"]
    outcome = _outcome_of(result)
    terms = history.latest_spec_terms(task)
    rounds = len(_spec_messages(task))
    builds = _build_count(task)

    designer_u = coder_u = None
    if terms is not None:
        designer_u = designer.utility(terms)
        coder_u = coder.utility(terms)

    joint = (designer_u + coder_u) if designer_u is not None and coder_u is not None else None
    nash = (designer_u * coder_u) if designer_u is not None and coder_u is not None else None

    return RunResult(
        experiment=experiment,
        mode=mode,
        run_id=run_id,
        outcome=outcome,
        terms=terms,
        rounds=rounds,
        designer_u=designer_u,
        coder_u=coder_u,
        joint_u=joint,
        nash_product=nash,
        surplus_fairness=(designer_u / (designer_u + coder_u))
        if designer_u is not None and coder_u is not None and (designer_u + coder_u) > 0
        else None,
        builds=builds,
        latency_s=round(latency, 3),
        source="llm" if mode == "live" else "fallback",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        transcript_hash=_transcript_hash(task),
        wire_leaks=_count_rationale_leaks(task),
        seed={
            "designer_overrides": designer_overrides or {},
            "coder_overrides": coder_overrides or {},
            "max_rounds": max_rounds,
            "band_low": band_low,
            "band_high": band_high,
            "include_rationale": include_rationale,
        },
    )


def _leak_detail(results: list[RunResult]) -> tuple[int, list[str]]:
    """Total wire leaks and the ids of the runs that leaked (per-run, not arm-level)."""

    return sum(r.wire_leaks for r in results), [r.run_id for r in results if r.wire_leaks]


def summarize(name: str, results: list[RunResult]) -> None:
    print(f"\n## {name} ({len(results)} runs)")
    completed = [r for r in results if r.outcome == "completed"]
    print(f"agreement rate: {len(completed) / max(1, len(results)):.2f} "
          f"({len(completed)}/{len(results)})")
    for label, key in [
        ("designer utility", "designer_u"),
        ("coder utility", "coder_u"),
        ("joint utility", "joint_u"),
        ("nash product", "nash_product"),
        ("surplus fairness", "surplus_fairness"),
        ("builds per run", "builds"),
        ("latency s", "latency_s"),
    ]:
        vals = [getattr(r, key) for r in completed]
        vals = [v for v in vals if v is not None]
        if vals:
            print(f"{label}: mean={sum(vals)/len(vals):.3f} min={min(vals):.3f} max={max(vals):.3f}")
    tokens_in = sum(r.tokens_in for r in results)
    tokens_out = sum(r.tokens_out for r in results)
    leaks, leaky = _leak_detail(results)
    worst = max((r.wire_leaks for r in results), default=0)
    print(f"wire leaks: {leaks} over {len(results)} runs "
          f"(worst single run: {worst}, runs with leaks: {len(leaky)})")
    if leaky:
        ids = ", ".join(leaky[:5])
        more = f" (+{len(leaky) - 5} more)" if len(leaky) > 5 else ""
        print(f"  leaking runs: {ids}{more}")
    if tokens_in or tokens_out:
        print(f"tokens: in={tokens_in} out={tokens_out} per-run={tokens_in // max(1, len(results))}")


def write_jsonl(mode: str, results: list[RunResult], name: str | None = None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name or f'{results[0].experiment}-{mode}'}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result.__dict__, default=str) + "\n")
    return out


def run_x1(mode: str, runs: int) -> list[RunResult]:
    results: list[RunResult] = []
    for designer_beta in (3.0, 1.0, 0.5):
        for coder_beta in (3.0, 1.0, 0.5):
            for index in range(runs):
                results.append(asyncio.run(
                    run_one(
                        "x1", mode, f"db{designer_beta}-cb{coder_beta}-{index}",
                        designer_overrides={"beta": designer_beta},
                        coder_overrides={"beta": coder_beta},
                        brief_path=STRESS_BRIEF,
                    )
                ))
    out = write_jsonl(mode, results)
    print(f"saved {out}")
    for designer_beta in (3.0, 1.0, 0.5):
        for coder_beta in (3.0, 1.0, 0.5):
            cell = [
                r for r in results
                if r.seed.get("designer_overrides", {}).get("beta") == designer_beta
                and r.seed.get("coder_overrides", {}).get("beta") == coder_beta
            ]
            completed = [r for r in cell if r.outcome == "completed" and r.joint_u is not None]
            rate = len(completed) / len(cell)
            joint = sum(r.joint_u for r in completed) / max(1, len(completed)) if completed else 0.0
            print(f"designer_beta={designer_beta} coder_beta={coder_beta}: "
                  f"agree={rate:.2f} joint={joint:.3f}")
    summarize("X1 beta grid (fallback)" if mode == "fallback" else "X1 beta grid (live)", results)
    return results


def run_x2(mode: str, runs: int, *, designer_model: str | None = None,
           coder_model: str | None = None) -> list[RunResult]:
    """LLM proposer on both sides vs the pure code-decided fallback."""

    if mode != "live":
        raise SystemExit("X2 compares the LLM against fallback; use --mode live with NIM keys")
    _require_live(designer_model, coder_model)
    results: list[RunResult] = []
    results += _run_arm("x2", "fallback", "fallback", runs)
    results += _run_arm(
        "x2", "llm_both", "live", runs,
        designer_chat=llm.default_chat(designer_model),
        coder_chat=llm.default_chat(coder_model),
    )
    out = write_jsonl(mode, results, name="x2-compare")
    print(f"saved {out}")
    for arm in ("fallback", "llm_both"):
        summarize(f"X2 arm={arm} (live)", [r for r in results if r.run_id.startswith(arm)])
    return results


def run_x3(mode: str, runs: int, *, designer_model: str | None = None,
           coder_model: str | None = None) -> list[RunResult]:
    """Coder rationale on the wire vs suppressed (live only).

    Machine-checked: no rationale on the wire may contain a leak phrase;
    the agreement rate is reported for both arms. Exits nonzero if the
    rationale_on arm leaks in any run — publishing rationale on the wire
    is gated on zero leaks.
    """

    if mode != "live":
        raise SystemExit("X3 needs a real proposer; use --mode live with NIM keys")
    _require_live(designer_model, coder_model)
    results: list[RunResult] = []
    arm_leaks: dict[str, list[str]] = {}
    for label, enabled in (("rationale_on", True), ("rationale_off", False)):
        arm = _run_arm(
            "x3", label, "live", runs,
            designer_chat=llm.default_chat(designer_model),
            coder_chat=llm.default_chat(coder_model),
            include_rationale=enabled,
        )
        results += arm
        leaks, leaky = _leak_detail(arm)
        arm_leaks[label] = leaky
        if leaky:
            more = f" (+{len(leaky) - 5} more)" if len(leaky) > 5 else ""
            verdict = f"FAIL ({len(leaky)}/{len(arm)} runs leaked: {', '.join(leaky[:5])}{more})"
        else:
            verdict = "PASS"
        print(f"X3 arm={label}: rationale leaks on the wire = {leaks} -> {verdict}")
        summarize(f"X3 arm={label} (live)", arm)
    out = write_jsonl(mode, results, name="x3-rationale")
    print(f"saved {out}")
    leaked = arm_leaks.get("rationale_on", [])
    if leaked:
        # Live gate: publishing rationale on the wire with a leak in it is a
        # hard failure, not a warning — nonzero exit, offending ids listed.
        raise SystemExit(
            f"X3 leak gate failed: rationale_on leaked in runs: {', '.join(leaked)}"
        )
    return results


def run_x4(mode: str, runs: int) -> list[RunResult]:
    results: list[RunResult] = []
    for max_rounds in (4, 8, 12):
        for index in range(runs):
            results.append(asyncio.run(
                run_one("x4", mode, f"T{max_rounds}-{index}", max_rounds=max_rounds, brief_path=STRESS_BRIEF)
            ))
    out = write_jsonl(mode, results)
    print(f"saved {out}")
    summarize(f"X4 MAX_ROUNDS in {{4,8,12}} ({mode})", results)
    return results


def run_x5(mode: str, runs: int, *, designer_model: str | None = None,
           coder_model: str | None = None) -> list[RunResult]:
    """Same model on both sides vs two different NIM models."""

    if mode != "live":
        raise SystemExit("X5 compares real NIM models; use --mode live with NIM keys")
    designer_model = designer_model or config.NIM_MODEL
    coder_model = coder_model or os.getenv("NIM_MODEL_B", "")
    if not designer_model or not coder_model:
        raise SystemExit(
            "X5 needs two model ids: --designer-model/--coder-model "
            "or NIM_MODEL/NIM_MODEL_B in .env"
        )
    if designer_model == coder_model:
        raise SystemExit("X5 needs two DIFFERENT model ids")
    _require_live(designer_model, coder_model)
    results: list[RunResult] = []
    for label, left, right in (
        ("same_model", designer_model, designer_model),
        ("distinct_models", designer_model, coder_model),
    ):
        arm = _run_arm(
            "x5", label, "live", runs,
            designer_chat=llm.default_chat(left),
            coder_chat=llm.default_chat(right),
        )
        results += arm
        summarize(f"X5 arm={label} designer={left} coder={right}", arm)
    out = write_jsonl(mode, results, name="x5-models")
    print(f"saved {out}")
    return results


def run_x6(mode: str, runs: int) -> list[RunResult]:
    results: list[RunResult] = []
    for band in (0.03, 0.05, 0.10):
        for index in range(runs):
            results.append(asyncio.run(
                run_one("x6", mode, f"band{band}-{index}", band_low=band, band_high=band, brief_path=STRESS_BRIEF)
            ))
    out = write_jsonl(mode, results)
    print(f"saved {out}")
    if mode == "fallback":
        print(
            "note: the band gate validates the LLM proposer only; the offline "
            "code path bypasses V3, so the arms are identical by construction. "
            "Use --mode live for a meaningful X6."
        )
    summarize("X6 band widths", results)
    return results


def run_x7(mode: str, runs: int) -> list[RunResult]:
    """Reproducibility: the same inputs must produce the identical transcript."""

    results: list[RunResult] = []
    for index in range(runs):
        results.append(asyncio.run(run_one("x7", mode, f"repro-{index}")))
    out = write_jsonl(mode, results)
    print(f"saved {out}")
    hashes = {r.transcript_hash for r in results}
    if len(hashes) == 1:
        print(f"all {runs} transcripts identical -> PASS")
    else:
        print(f"{len(hashes)} distinct transcripts across {runs} runs -> FAIL")
    return results


def run_x8(mode: str, runs: int) -> list[RunResult]:
    """One-build audit: exactly one build per completed task, plus latency."""

    results: list[RunResult] = []
    for index in range(runs):
        results.append(asyncio.run(run_one("x8", mode, f"audit-{index}")))
    out = write_jsonl(mode, results)
    print(f"saved {out}")
    completed = [r for r in results if r.outcome == "completed"]
    passed = all(r.builds == 1 for r in completed)
    print(f"builds == 1 on every completed task: {'PASS' if passed else 'FAIL'} "
          f"({sum(r.builds for r in completed)} builds across {len(completed)} completed)")
    if completed:
        latencies = sorted(r.latency_s for r in completed)
        mid = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"latency p50={mid:.2f}s p95={p95:.2f}s")
    return results


def _require_live(designer_model: str | None = None, coder_model: str | None = None) -> None:
    """Fail fast and actionably instead of emitting a traceback."""

    if not config.NVIDIA_API_KEY:
        raise SystemExit(
            "live mode needs NVIDIA_API_KEY (copy .env.example to .env and fill it in)"
        )
    if not (designer_model or config.NIM_MODEL) or not (coder_model or config.NIM_MODEL):
        raise SystemExit(
            "live mode needs a model id: NIM_MODEL in .env, or --designer-model/--coder-model"
        )


def _run_arm(
    experiment: str,
    arm: str,
    mode: str,
    runs: int,
    **kwargs: Any,
) -> list[RunResult]:
    return [
        asyncio.run(run_one(experiment, mode, f"{arm}-{index}", brief_path=STRESS_BRIEF, **kwargs))
        for index in range(runs)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_evals")
    parser.add_argument(
        "--experiment",
        choices=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"],
        required=False,
    )
    parser.add_argument("--mode", choices=["fallback", "live"], default="fallback")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--designer-model", default=None, help="X2/X3/X5 designer/buyer model id")
    parser.add_argument("--coder-model", default=None, help="X2/X3/X5 coder/seller model id")
    parser.add_argument(
        "--all-offline", action="store_true",
        help="run x1, x4, x6, x7 and x8 in fallback mode",
    )
    args = parser.parse_args()

    if args.all_offline:
        run_x1("fallback", args.runs)
        run_x4("fallback", args.runs)
        run_x6("fallback", args.runs)
        run_x7("fallback", args.runs)
        run_x8("fallback", args.runs)
        return
    try:
        if args.experiment == "x1":
            run_x1(args.mode, args.runs)
        elif args.experiment == "x2":
            run_x2(args.mode, args.runs,
                   designer_model=args.designer_model, coder_model=args.coder_model)
        elif args.experiment == "x3":
            run_x3(args.mode, args.runs,
                   designer_model=args.designer_model, coder_model=args.coder_model)
        elif args.experiment == "x4":
            run_x4(args.mode, args.runs)
        elif args.experiment == "x5":
            run_x5(args.mode, args.runs,
                   designer_model=args.designer_model, coder_model=args.coder_model)
        elif args.experiment == "x6":
            run_x6(args.mode, args.runs)
        elif args.experiment == "x7":
            run_x7(args.mode, args.runs)
        elif args.experiment == "x8":
            run_x8(args.mode, args.runs)
        else:
            parser.print_help()
    except MissingCredentials as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
