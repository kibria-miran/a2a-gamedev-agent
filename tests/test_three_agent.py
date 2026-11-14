"""S26: the headline three-agent run (Designer -> Coder -> Critic)."""

import hashlib
import json

from src import config, history, sandbox
from src.three_agent_demo import run_demo


async def test_s26_reference_brief_to_verified_playable_invaders():
    result = await run_demo(db_path=":memory:", verbose=False)
    task = result["task"]

    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert result["decision"].kind == "accept"

    types = [(history.data_of(message) or {}).get("type") for message in history.history(task)]
    assert types == ["offer", "counter", "spec_signed", "build", "accept"]
    assert types.count("build") == 1  # E-68: exactly one build

    rounds = [(history.data_of(message) or {}).get("round") for message in history.history(task)]
    # E-67: Designer even, Coder odd; spec_signed closes negotiation;
    # build and review do not advance the round.
    assert rounds == [0, 1, 2, 3, 3]

    spec = history.latest_spec_terms(task)
    assert spec["genre"] == "invaders"
    assert spec["reliability"] == "deterministic"

    artifact = result["artifact"]
    assert artifact is not None
    assert artifact["sha256"] == hashlib.sha256(result["binary"]).hexdigest()
    assert result["sha256"] == artifact["sha256"]

    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    played = sandbox.run_game(
        result["binary"],
        config.CRITIC_TAPE,
        extra_env={"HARNESS_SPEC_JSON": canonical, "HARNESS_PLAY_TIMEOUT": "10"},
    )
    assert played.exit_code == 0
    assert len(played.digests) == config.CRITIC_TAPE.count(b"tick")

    # The accept is the Critic's, and it carries the verified artifact.
    accept = history.data_of(history.history(task)[-1])
    assert accept["type"] == "accept"
    assert accept["build"]["artifactId"] == artifact["artifactId"]
    assert accept["build"]["sha256"] == artifact["sha256"]
