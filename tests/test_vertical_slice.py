"""The Phase 1 done-when test (E-75): brief in, playable binary out.

No server, no A2A, no DB. FakeLLM stands in for the model and returns the
checked-in reference invaders source; the Critic-style tape then plays the
built binary. Everything else — codegen prompt, extraction, sandbox build,
digest printing — is the real path.
"""

import json
from pathlib import Path

from src import coder_agent, sandbox

from tests.conftest import FakeLLM

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_GAME = ROOT / "evals" / "reference_games" / "invaders" / "main.cpp"
REFERENCE_BRIEF = ROOT / "evals" / "reference_brief.json"

# critic-style tape: slide, fire, then hold the trigger for a while
CRITIC_TAPE = (
    b"key a\nkey d\nkey k\n"
    + b"tick\n" * 10
    + b"key k\ntick\n"
    + b"tick\n" * 15
)


def load_reference_brief() -> dict:
    return json.loads(REFERENCE_BRIEF.read_text(encoding="utf-8"))


def reference_source() -> str:
    return REFERENCE_GAME.read_text(encoding="utf-8")


def test_brief_to_playable_binary():
    spec = load_reference_brief()
    source = reference_source()
    fake = FakeLLM([source])

    built_source, result = coder_agent.generate_and_build(spec, chat=fake)
    assert result.ok, result.stderr
    assert built_source.strip()

    # the built binary plays: exit 0, every tick printed a digest line
    run = sandbox.run_game(
        result.binary_bytes,
        CRITIC_TAPE,
        extra_env={
            "HARNESS_SPEC_JSON": json.dumps(spec, sort_keys=True, separators=(",", ":")),
            "HARNESS_PLAY_TIMEOUT": "10",
        },
    )
    assert run.exit_code == 0
    assert len(run.digests) == CRITIC_TAPE.count(b"tick")
    assert run.digests[0][0] == 0
    assert run.digests[-1][0] == CRITIC_TAPE.count(b"tick") - 1
    assert all(len(h) == 8 for _, h in run.digests)

    # FakeLLM returned the reference source verbatim, so the binary must be
    # byte-identical to a direct reference build (same compiler, same flags)
    direct = sandbox.build_game({"main.cpp": source}, reference=False)
    assert direct.ok
    assert result.binary_sha256 == direct.binary_sha256


def test_critic_tape_deterministic_replay():
    spec = load_reference_brief()
    source = reference_source()
    direct = sandbox.build_game({"main.cpp": source}, reference=False)
    assert direct.ok
    env = {"HARNESS_SPEC_JSON": json.dumps(spec, sort_keys=True, separators=(",", ":"))}
    run1 = sandbox.run_game(direct.binary_bytes, CRITIC_TAPE, extra_env=env)
    run2 = sandbox.run_game(direct.binary_bytes, CRITIC_TAPE, extra_env=env)
    assert run1.stdout == run2.stdout


def test_reference_brief_is_a_canonical_spec():
    spec = load_reference_brief()
    assert spec["genre"] == "invaders"
    assert spec["grid_w"] == 40 and spec["grid_h"] == 20
    assert spec["win_condition"] == "survive"
    # sorted-keys compact separators are the harness_seed() source (E-74)
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    assert '"brief"' in canonical and '"grid_w":40' in canonical
