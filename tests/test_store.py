"""Tests for the persistence foundation (Phase 2 part 1).

E-66 spec_signed flow: the Designer's spec_signed arrives as a normal user
message via begin_turn; the store sets spec_signed=1 in that transaction so
R18/R19 are plain column reads. Round accounting stays E-47: counter +2,
everything else +0.
"""

import hashlib

import pytest

from src import config, state_machine
from src import store as store_mod


@pytest.fixture
def store(tmp_path):
    s = store_mod.Store(str(tmp_path / "coder.db"))
    yield s
    s.close()


def agent_parts(kind: str, **extra):
    data = {"type": kind}
    data.update(extra)
    return [{"data": data}]


def user_counter(round_number: int, terms=None):
    return [{"data": {"type": "counter", "round": round_number, "terms": terms or {}}}]


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_schema_has_spec_signed_and_artifact_blobs(store):
    task_cols = {row["name"] for row in store.conn.execute("PRAGMA table_info(tasks)")}
    assert "spec_signed" in task_cols
    artifact_cols = {row["name"] for row in store.conn.execute("PRAGMA table_info(artifacts)")}
    assert {"binary", "source_tar"} <= artifact_cols


def test_create_task_defaults(store):
    task = store.create_task()
    assert task["state"] == "submitted"
    assert task["round"] == 0
    assert task["spec_signed"] == 0


# ---------------------------------------------------------------------------
# spec_signed flag (E-66)
# ---------------------------------------------------------------------------

def test_begin_turn_spec_signed_sets_flag(store):
    task = store.create_task()
    updated = store.begin_turn(
        task["id"], "msg-sign-1", [{"data": {"type": "spec_signed", "round": 0}}]
    )
    assert updated["state"] == "working"
    assert store.spec_signed(task["id"]) is True


def test_begin_turn_non_spec_signed_keeps_flag_unset(store):
    task = store.create_task()
    store.begin_turn(task["id"], "msg-offer-1", user_counter(0))
    assert store.spec_signed(task["id"]) is False


def test_spec_signed_flag_is_sticky(store):
    task = store.create_task()
    store.begin_turn(task["id"], "msg-1", [{"data": {"type": "spec_signed", "round": 0}}])
    store.cas_commit(
        task["id"],
        "input-required",
        "build",
        agent_message_id="build-1",
        agent_parts=agent_parts("build", round=0),
    )
    task2 = store.get_task(task["id"])
    assert task2["spec_signed"] == 1


# ---------------------------------------------------------------------------
# round accounting (E-47) and the build kind
# ---------------------------------------------------------------------------

def test_begin_turn_commits_spec_message_plus_one(store):
    task = store.create_task()
    updated = store.begin_turn(task["id"], "msg-1", user_counter(0))
    assert updated["round"] == 1
    assert updated["round_at_turn_start"] == 0


def test_parity_designer_even_coder_odd(store):
    task = store.create_task()
    # designer offer 1: arrives at round 0, stored, round -> 1
    store.begin_turn(task["id"], "d-1", user_counter(0))
    # coder counter: spec message, round -> 2
    store.cas_commit(
        task["id"], "input-required", "counter", agent_parts=agent_parts("counter", round=1)
    )
    assert store.get_task(task["id"])["round"] == 2
    # designer offer 2: arrives at round 2, stored, round -> 3
    store.begin_turn(task["id"], "d-2", user_counter(2))
    assert store.get_task(task["id"])["round"] == 3
    # coder counter 2: round -> 4
    store.cas_commit(
        task["id"], "input-required", "counter", agent_parts=agent_parts("counter", round=3)
    )
    assert store.get_task(task["id"])["round"] == 4


def test_cas_commit_counter_advances_round_by_one(store):
    task = store.create_task()
    store.begin_turn(task["id"], "msg-1", user_counter(0))
    ok = store.cas_commit(
        task["id"], "input-required", "counter", agent_parts=agent_parts("counter", round=0)
    )
    assert ok
    # begin_turn committed +1 (spec message), counter commits +1
    assert store.get_task(task["id"])["round"] == 2


def test_cas_commit_build_keeps_round(store):
    task = store.create_task()
    store.begin_turn(task["id"], "msg-1", [{"data": {"type": "spec_signed", "round": 0}}])
    ok = store.cas_commit(
        task["id"], "input-required", "build", agent_parts=agent_parts("build", round=0)
    )
    assert ok
    updated = store.get_task(task["id"])
    assert updated["state"] == "input-required"
    # designer's spec_signed committed +1 (spec message), the build commits 0
    assert updated["round"] == 1
    assert updated["spec_signed"] == 1


def test_begin_turn_rejects_invalid_transition(store):
    task = store.create_task()
    store.begin_turn(task["id"], "msg-1", user_counter(0))
    # still working (set above) -> working is not a legal transition from working
    with pytest.raises(state_machine.StateError):
        store.begin_turn(task["id"], "msg-2", user_counter(2))


def test_agent_parts_reject_designer_only_types(store):
    task = store.create_task()
    store.begin_turn(task["id"], "msg-1", user_counter(0))
    # the Coder agent may not emit spec_signed (designer's move, E-66)
    with pytest.raises(store_mod.StoreError, match="spec_signed"):
        store.cas_commit(
            task["id"],
            "input-required",
            "counter",
            agent_parts=[{"data": {"type": "spec_signed", "round": 0}}],
        )


# ---------------------------------------------------------------------------
# binary artifacts (E-70)
# ---------------------------------------------------------------------------

def test_store_and_fetch_binary_artifact(store):
    task = store.create_task()
    binary = b"\x7fELF" + bytes(range(256)) * 40  # 10,244 bytes
    source_tar = b"fake-tar-bytes"
    descriptor = store.store_binary_artifact(
        task["id"], binary=binary, source_tar=source_tar, name="game"
    )
    assert descriptor["sha256"] == hashlib.sha256(binary).hexdigest()
    assert descriptor["size"] == len(binary)

    fetched = store.get_binary_artifact(task["id"], descriptor["artifactId"])
    assert fetched is not None
    assert fetched["binary"] == binary
    assert fetched["sha256"] == descriptor["sha256"]
    assert store.get_source_tar_artifact(task["id"], descriptor["artifactId"]) == source_tar


def test_store_binary_artifact_enforces_size_cap(store):
    task = store.create_task()
    with pytest.raises(store_mod.StoreError, match="MAX_ARTIFACT_BYTES"):
        store.store_binary_artifact(
            task["id"], binary=b"x" * (config.MAX_ARTIFACT_BYTES + 1)
        )


def test_store_source_tar_enforces_size_cap(store):
    task = store.create_task()
    with pytest.raises(store_mod.StoreError, match="MAX_ARTIFACT_BYTES"):
        store.store_binary_artifact(
            task["id"],
            binary=b"tiny",
            source_tar=b"x" * (config.MAX_ARTIFACT_BYTES + 1),
        )


def test_artifact_listing_omits_blob_payloads(store):
    task = store.create_task()
    store.store_binary_artifact(task["id"], binary=b"a" * 4096, name="game")
    listing = store.artifacts_for_task(task["id"])
    assert len(listing) == 1
    assert listing[0]["binaryPresent"] is True
    assert "binary" not in listing[0]
    assert "source_tar" not in listing[0]


def test_get_binary_artifact_unknown_ids(store):
    task = store.create_task()
    assert store.get_binary_artifact(task["id"], "nope") is None
    assert store.get_binary_artifact("other-task", "nope") is None


# ---------------------------------------------------------------------------
# lifecycle plumbing shared with the demo
# ---------------------------------------------------------------------------

def test_startup_recovery_recovers_submitted_to_input_required(store):
    task = store.create_task()
    assert store.startup_recovery() == 1
    recovered = store.get_task(task["id"])
    assert recovered["state"] == "input-required"


def test_complete_with_accept_critic_path_no_terms(store):
    task = store.create_task()
    store.begin_turn(task["id"], "msg-1", [{"data": {"type": "spec_signed", "round": 0}}])
    store.cas_commit(
        task["id"], "input-required", "build", agent_parts=agent_parts("build", round=0)
    )
    done = store.complete_with_accept(
        task["id"], None, message_id="critic-accept-1",
        parts=[{"data": {"type": "accept", "round": 0}}],
    )
    assert done["state"] == "completed"
    # critic accept must not write a fake agreement artifact
    assert store.artifacts_for_task(task["id"]) == []


def test_cancel_task_terminal_states_return_not_cancelable(store):
    task = store.create_task()
    store.begin_turn(task["id"], "msg-1", [{"data": {"type": "spec_signed", "round": 0}}])
    store.cas_commit(
        task["id"], "input-required", "build", agent_parts=agent_parts("build", round=0)
    )
    done = store.complete_with_accept(task["id"])
    canceled, code = store.cancel_task(done["id"])
    assert code == config.A2A_TASK_NOT_CANCELABLE
    gone, code = store.cancel_task("does-not-exist")
    assert gone is None and code == config.A2A_TASK_NOT_FOUND


def test_cas_commit_rejects_unknown_sent_kind(store):
    task = store.create_task()
    store.begin_turn(task["id"], "msg-1", user_counter(0))
    with pytest.raises(store_mod.StoreError, match="unknown sent_kind"):
        store.cas_commit(task["id"], "input-required", "teleport")
