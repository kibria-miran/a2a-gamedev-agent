"""Agent-level tests: Designer signoff, Coder build-once, Critic verification."""

import hashlib

import pytest

from src import client as client_mod
from src import config, critic_agent, designer_agent, history, llm, sandbox
from src.coder_agent import respond as coder_respond
from src.transport import counter_parts, message_dict, offer_parts, spec_signed_parts
from src.utility import load_profile

MINI_GAME = """
#include "harness.h"
#include <cstdio>
namespace { int x = 0; }
int harness_init(void) { return 0; }
void harness_tick(void) { ++x; }
const char* harness_render(void) {
    static char buf[32];
    std::snprintf(buf, sizeof buf, "x=%d", x);
    return buf;
}
"""

BROKEN_GAME = """
#include "harness.h"
int harness_init(void) { return 0; }
void harness_tick(void) { nope_not_a_symbol(); }
const char* harness_render(void) { return "x"; }
"""

REFERENCE = {
    "genre": "invaders",
    "grid_w": 40,
    "grid_h": 20,
    "tick_ms": 100,
    "enemy_count": 8,
    "win_condition": "survive",
    "reliability": "deterministic",
}

UNATTRACTIVE = {
    "genre": "maze",
    "grid_w": 24,
    "grid_h": 12,
    "tick_ms": 250,
    "enemy_count": 2,
    "win_condition": "score",
    "reliability": "deterministic",
}


@pytest.fixture(scope="module")
def designer_profile():
    return load_profile("profiles/designer.json")


@pytest.fixture(scope="module")
def coder_profile():
    return load_profile("profiles/coder.json")


@pytest.fixture(scope="module")
def mini_build():
    result = sandbox.build_game({"main.cpp": MINI_GAME})
    assert result.ok, result.stderr
    return result


def view_with(messages, *, state="input-required", task_id="task-1"):
    return {
        "id": task_id,
        "contextId": "ctx",
        "status": {"state": config.A2A_STATE_BY_INTERNAL[state]},
        "history": messages,
        "artifacts": [],
    }


def offer(round_number, terms):
    return message_dict(role="user", parts=offer_parts(round_number, terms))


def counter(round_number, terms, text="counter"):
    return message_dict(
        role="agent", parts=counter_parts(round_number, terms),
        message_id=f"c-{round_number}",
    )


# ---------------------------------------------------------------------------
# Designer
# ---------------------------------------------------------------------------

def test_designer_opens_with_the_brief(designer_profile):
    decision = designer_agent.next_move(
        view_with([]), designer_profile, brief=REFERENCE
    )
    assert decision.kind == "offer"
    assert decision.round == 0
    assert decision.terms == REFERENCE


def test_designer_signs_an_acceptable_counter(designer_profile):
    view = view_with([offer(0, REFERENCE), counter(1, REFERENCE)])
    decision = designer_agent.next_move(view, designer_profile)
    assert decision.kind == "spec_signed"
    assert decision.round == 2
    assert decision.terms == REFERENCE


def test_designer_concedes_on_an_unacceptable_counter(designer_profile):
    view = view_with([offer(0, REFERENCE), counter(1, UNATTRACTIVE)])
    decision = designer_agent.next_move(view, designer_profile)
    assert decision.kind == "offer"
    assert decision.round == 2
    assert decision.terms != UNATTRACTIVE


def test_designer_walks_away_at_max_rounds(designer_profile):
    view = view_with([offer(0, REFERENCE), counter(config.MAX_ROUNDS, UNATTRACTIVE)])
    decision = designer_agent.next_move(view, designer_profile)
    assert decision.kind == "walk_away"
    assert decision.reason == "max_rounds"


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------

def test_coder_echoes_acceptable_terms_for_signoff(coder_profile):
    decision = coder_respond(
        view_with([]), {"type": "offer", "round": 0, "terms": REFERENCE},
        profile=coder_profile,
    )
    assert decision.kind == "counter"
    assert decision.round == 1
    assert decision.terms == REFERENCE


def test_coder_counters_with_its_own_terms(coder_profile):
    decision = coder_respond(
        view_with([]), {"type": "offer", "round": 0, "terms": UNATTRACTIVE},
        profile=coder_profile,
    )
    assert decision.kind == "counter"
    assert decision.round == 1
    assert coder_profile.feasible(decision.terms)
    assert decision.terms != UNATTRACTIVE


def test_coder_builds_once_after_spec_signed(coder_profile, mini_build):
    decision = coder_respond(
        view_with([]),
        {"type": "spec_signed", "round": 2, "terms": REFERENCE},
        profile=coder_profile,
        codegen=lambda terms: (MINI_GAME, mini_build),
    )
    assert decision.kind == "build"
    assert decision.build_result.ok
    assert decision.round == 3


def test_coder_walks_away_when_build_fails(coder_profile):
    failed = sandbox.BuildResult(ok=False, violations=("compile",), exit_code=1)
    decision = coder_respond(
        view_with([]),
        {"type": "spec_signed", "round": 2, "terms": REFERENCE},
        profile=coder_profile,
        codegen=lambda terms: (BROKEN_GAME, failed),
    )
    assert decision.kind == "walk_away"
    assert decision.reason == "build_failed"


def test_coder_walks_away_on_llm_error(coder_profile):
    def boom(terms):
        raise llm.LLMError("not configured")

    decision = coder_respond(
        view_with([]),
        {"type": "spec_signed", "round": 2, "terms": REFERENCE},
        profile=coder_profile,
        codegen=boom,
    )
    assert decision.kind == "walk_away"
    assert decision.reason == "llm_error"


def test_coder_rejects_invented_genre(coder_profile):
    decision = coder_respond(
        view_with([]),
        {"type": "offer", "round": 0, "terms": {**REFERENCE, "genre": "roguelike"}},
        profile=coder_profile,
    )
    assert decision.kind == "walk_away"
    assert decision.reason == "bad_terms"


# ---------------------------------------------------------------------------
# Critic (S25/S27)
# ---------------------------------------------------------------------------

class StubClient:
    def __init__(self, blob):
        self.blob = blob

    async def fetch_build(self, task_id, artifact_id):
        return self.blob, hashlib.sha256(self.blob).hexdigest(), len(self.blob)


class BadShaClient:
    async def fetch_build(self, task_id, artifact_id):
        raise client_mod.ShaMismatch("header does not match body")


def build_view(mini_build):
    artifact = {
        "artifactId": "artifact-1",
        "sha256": mini_build.binary_sha256,
        "size": len(mini_build.binary_bytes or b""),
    }
    return view_with(
        [
            offer(0, REFERENCE),
            counter(1, REFERENCE),
            message_dict(role="user", parts=spec_signed_parts(2, REFERENCE)),
            message_dict(
                role="agent",
                parts=[{"data": {"type": "build", "round": 3, "artifact": artifact}}],
            ),
        ]
    )


async def test_critic_accepts_a_verified_build(mini_build):
    decision = await critic_agent.review(
        build_view(mini_build), StubClient(mini_build.binary_bytes or b"")
    )
    assert decision.kind == "accept"
    assert decision.round == 3
    assert decision.message["artifact"]["artifactId"] == "artifact-1"
    assert decision.message["artifact"]["sha256"] == mini_build.binary_sha256


async def test_s27_sha_mismatch_walks_away(mini_build):
    decision = await critic_agent.review(build_view(mini_build), BadShaClient())
    assert decision.kind == "walk_away"
    assert decision.reason == "sha256_mismatch"


async def test_critic_walks_away_when_the_game_fails_to_play(mini_build):
    def failing_run(blob, tape, **kwargs):
        return sandbox.RunResult(exit_code=1, stdout=b"")

    decision = await critic_agent.review(
        build_view(mini_build),
        StubClient(mini_build.binary_bytes or b""),
        run_game=failing_run,
    )
    assert decision.kind == "walk_away"
    assert decision.reason == "play_failed"


async def test_critic_walks_away_without_a_build(mini_build):
    view = view_with([offer(0, REFERENCE), counter(1, REFERENCE)])
    decision = await critic_agent.review(view, StubClient(b"x"))
    assert decision.kind == "walk_away"
    assert decision.reason == "no_build"


ALLOW_LISTED_SOURCE = """
#include "harness.h"
#include <ctime>
#include <cstdio>
int harness_init(void) { return 0; }
void harness_tick(void) {}
const char* harness_render(void) {
    std::time_t t = std::time(nullptr); // allow-nondeterministic
    // reason: allow-listed escape hatch; digest may differ across boots
    static char buf[32];
    std::snprintf(buf, sizeof buf, "t=%lld", (long long)t);
    return buf;
}
"""


def _tar_bytes(sources):
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, text in sorted(sources.items()):
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class AllowListedClient(StubClient):
    """StubClient plus the R20 source route, backed by a fixed tar."""

    def __init__(self, blob, tar_bytes):
        super().__init__(blob)
        self._tar = tar_bytes

    async def fetch_source(self, task_id, artifact_id):
        return self._tar, hashlib.sha256(self._tar).hexdigest(), len(self._tar)


async def test_critic_threads_the_source_allow_list_into_the_gate(mini_build):
    """E-74: a source allow-listed via ``// allow-nondeterministic`` must reach
    the determinism gate as ``nondeterministic_source=True`` — the Coder's
    wire claims are not trusted; the fetched source tar is the evidence."""
    tar = _tar_bytes({"main.cpp": ALLOW_LISTED_SOURCE})
    captured = {}

    def recording_gate(blob, *, tape, nondeterministic_source, extra_env):
        captured["nondeterministic_source"] = nondeterministic_source
        return sandbox.DeterminismGate(
            ran_twice=True, ok=True, nondeterministic_source=nondeterministic_source
        )

    decision = await critic_agent.review(
        build_view(mini_build),
        AllowListedClient(mini_build.binary_bytes or b"", tar),
        gate=recording_gate,
    )
    assert captured["nondeterministic_source"] is True
    assert decision.kind == "accept"


async def test_critic_falls_back_to_the_strict_gate_without_a_source_route(mini_build):
    """A client without fetch_source (or a failing fetch) fails closed:
    the gate runs as if the source were not allow-listed."""
    captured = {}

    def recording_gate(blob, *, tape, nondeterministic_source, extra_env):
        captured["nondeterministic_source"] = nondeterministic_source
        return sandbox.DeterminismGate(
            ran_twice=True, ok=True, nondeterministic_source=nondeterministic_source
        )

    decision = await critic_agent.review(
        build_view(mini_build),
        StubClient(mini_build.binary_bytes or b""),
        gate=recording_gate,
    )
    assert captured["nondeterministic_source"] is False
    assert decision.kind == "accept"


def test_sample_digests_keeps_the_final_frame():
    digests = tuple((i, f"{i:08x}") for i in range(200))
    sampled = history.sample_digests(digests, limit=10)
    assert len(sampled) == 10
    assert sampled[-1] == digests[-1]
