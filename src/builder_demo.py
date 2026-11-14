"""python -m src.builder_demo — the Phase 1 headline.

Builds the reference invaders from the reference brief with NO server, NO A2A,
NO DB. If NIM is configured (.env-style env vars), the coder generates the
game from the brief for real. Otherwise the demo uses the checked-in reference
source as the oracle and says so loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src import coder_agent, llm, sandbox

ROOT = Path(__file__).resolve().parents[1]
BRIEF_PATH = ROOT / "evals" / "reference_brief.json"
REFERENCE_SOURCE_PATH = ROOT / "evals" / "reference_games" / "invaders" / "main.cpp"
DEFAULT_TAPE = b"key a\nkey d\nkey k\ntick\ntick\n" + b"tick\n" * 25


def main() -> int:
    spec = json.loads(BRIEF_PATH.read_text(encoding="utf-8"))
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))

    print("=== A2A Game Dev Agent — builder demo (Phase 1) ===\n")
    print(f"brief  : {spec['brief']}")
    print(f"spec   : {canonical}\n")

    client = llm.default_chat()
    reference_source = REFERENCE_SOURCE_PATH.read_text(encoding="utf-8")
    offline_oracle = client is None
    if offline_oracle:
        print("llm    : NIM not configured — using the checked-in reference invaders")
        print("         as the codegen oracle (offline demo; export NIM_MODEL and")
        print("         NVIDIA_API_KEY for a real run)\n")

    if offline_oracle:
        result = sandbox.build_game({"main.cpp": reference_source})
        source = reference_source
    else:
        print("llm    : NIM configured — generating from the brief (real run)\n")
        source, result = coder_agent.generate_and_build(spec)

    if not result.ok:
        print("build  : FAILED")
        if result.sandbox_violation:
            print(f"         sandbox violation: {'; '.join(result.violations)}")
        else:
            print(f"         exit_code={result.exit_code}")
            print(result.stderr[-2000:] or "; ".join(result.violations))
        return 1

    print(f"build  : OK  sha256={result.binary_sha256}")
    assert result.binary_bytes is not None
    print(f"         binary {len(result.binary_bytes)} bytes\n")

    env = {"HARNESS_SPEC_JSON": canonical, "HARNESS_PLAY_TIMEOUT": "10"}
    run = sandbox.run_game(result.binary_bytes, DEFAULT_TAPE, extra_env=env)
    print(f"run    : exit_code={run.exit_code}  digests={len(run.digests)}")
    for tick, digest in run.digests:
        print(f"         tick {tick:<2} {digest}")

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "invaders"
    out_path.write_bytes(result.binary_bytes)
    out_path.chmod(0o755)
    print(f"\nplayable binary written to {out_path}")
    print("(the harness reads a tape on stdin — see README / §8.11 of the spec)")

    return 0 if run.exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
