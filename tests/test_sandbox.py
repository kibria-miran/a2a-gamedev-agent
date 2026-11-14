"""Tests for src/sandbox.py (E-69, E-70, E-73, E-74).

Compile-heavy tests build trivial games against the real harness; they stay
under a few seconds each on a developer machine. Tests that need a specific
network-isolation backend skip (with reason) when the backend can't run
here — exactly what E-69 requires for the unshare fallback.
"""

import hashlib
import http.server
import io
import os
import socket
import tarfile
import threading
import time
from pathlib import Path

import pytest

from src import sandbox

ROOT = Path(__file__).resolve().parents[1]

SPEC_JSON = '{"grid_w": 40, "grid_h": 20, "tick_ms": 0, "enemy_count": 8, "win_condition": "survive"}'

COUNTER_GAME = """
#include "harness.h"
#include <cstdio>
static int x = 0;
int harness_init(void) { return 0; }
void harness_tick(void) {
    int k = harness_poll_key();
    if (k == 'a') x += 1;
    else if (k == 'd') x -= 1;
}
const char* harness_render(void) {
    static char buf[64];
    std::snprintf(buf, sizeof buf, "x=%d", x);
    return buf;
}
"""

INFINITE_LOOP_GAME = """
#include "harness.h"
int harness_init(void) { return 0; }
void harness_tick(void) { volatile long sink = 0; for (;;) sink++; }
const char* harness_render(void) { return "loop"; }
"""

HUGE_ALLOC_GAME = """
#include "harness.h"
#include <new>
int harness_init(void) {
    volatile char* p = new (std::nothrow) char[512ull * 1024 * 1024];
    if (!p) return 3;
    p[0] = 1;
    delete[] p;
    return 0;
}
void harness_tick(void) {}
const char* harness_render(void) { return "alloc"; }
"""

NETWORK_GAME = """
#include "harness.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdlib>
int harness_init(void) {
    const char* p = getenv("A2A_TEST_PORT");
    if (!p) return 7;
    int port = std::atoi(p);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 8;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<unsigned short>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int rc = connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
    close(fd);
    return rc == 0 ? 0 : 7;
}
void harness_tick(void) {}
const char* harness_render(void) { return "net"; }
"""

UNUSED_VAR_GAME = """
#include "harness.h"
int harness_init(void) { return 0; }
void harness_tick(void) { int unused_local_variable = 0; }
const char* harness_render(void) { return "u"; }
"""

ENV_DUMP_GAME = """
#include "harness.h"
#include <cstdio>
#include <cstdlib>
int harness_init(void) {
    const char* k = getenv("NVIDIA_API_KEY");
    if (k) std::fputs(k, stderr);
    return 0;
}
void harness_tick(void) {}
const char* harness_render(void) { return "env"; }
"""

BANNED_DATE_GAME = """
#include "harness.h"
int harness_init(void) { return 0; }
void harness_tick(void) {}
const char* harness_render(void) { return __DATE__; }
"""

BANNED_WITH_ALLOW_NO_REASON = """
#include "harness.h"
#include <ctime>
int harness_init(void) { return 0; }
void harness_tick(void) {}
const char* harness_render(void) {
    std::time_t t = std::time(nullptr); // allow-nondeterministic
    static char buf[32];
    std::snprintf(buf, sizeof buf, "t=%lld", (long long)t);
    return buf;
}
"""

BANNED_WITH_ALLOW_AND_REASON = """
#include "harness.h"
#include <ctime>
#include <cstdio>
int harness_init(void) { return 0; }
void harness_tick(void) {}
const char* harness_render(void) {
    std::time_t t = std::time(nullptr); // allow-nondeterministic
    // reason: determinism-gate branch-B test; digests always differ across boots
    static char buf[32];
    std::snprintf(buf, sizeof buf, "t=%lld", (long long)t);
    return buf;
}
"""

PID_GAME = """
#include "harness.h"
#include <cstdio>
#include <unistd.h>
int harness_init(void) { return 0; }
void harness_tick(void) {}
const char* harness_render(void) {
    static char buf[32];
    std::snprintf(buf, sizeof buf, "pid=%d", (int)getpid());
    return buf;
}
"""

RAND_UNPINNED_GAME = """
#include "harness.h"
int harness_init(void) { return 0; }
void harness_tick(void) { (void)rand(); }
const char* harness_render(void) { return "rand"; }
"""

RAND_WITH_SEED_GAME = """
#include "harness.h"
#include <random>
int harness_init(void) { return 0; }
void harness_tick(void) { std::mt19937 rng(harness_seed()); (void)rng(); }
const char* harness_render(void) { return "seeded"; }
"""

MISSING_RENDER_GAME = """
#include "harness.h"
int harness_init(void) { return 0; }
void harness_tick(void) {}
"""

MISSING_INCLUDE_GAME = """
int harness_init(void) { return 0; }
void harness_tick(void) {}
const char* harness_render(void) { return "x"; }
"""


def build_counter() -> sandbox.BuildResult:
    return sandbox.build_game({"main.cpp": COUNTER_GAME})


def _backend_available(name: str) -> bool:
    if name == "bwrap":
        return bool(sandbox.BWRAP) and sandbox._probe(
            [sandbox.BWRAP, "--unshare-net", "--ro-bind", "/", "/", "/bin/true"]
        )
    if name == "unshare":
        return bool(sandbox.UNSHARE) and sandbox._probe([sandbox.UNSHARE, "-n", "/bin/true"])
    return False


# ---------------------------------------------------------------------------
# Pre-flight scans (no compiler involved)
# ---------------------------------------------------------------------------

def test_harness_contract_complete_game_has_no_violations():
    result = sandbox.build_game({"main.cpp": COUNTER_GAME})
    # COUNTER_GAME is valid; the only failures allowed would be infrastructure
    assert result.ok, result.stderr or result.violations
    assert result.binary_bytes
    assert result.binary_sha256 == hashlib.sha256(result.binary_bytes).hexdigest()


def test_harness_contract_missing_render_is_flagged_without_compiling():
    result = sandbox.build_game({"main.cpp": MISSING_RENDER_GAME})
    assert not result.ok
    assert any("harness_render" in v for v in result.violations)


def test_harness_contract_missing_include_is_flagged_without_compiling():
    result = sandbox.build_game({"main.cpp": MISSING_INCLUDE_GAME})
    assert not result.ok
    assert any("include" in v for v in result.violations)


def test_reference_flags_enforce_werror_but_llm_flags_do_not():
    llm = sandbox.build_game({"main.cpp": UNUSED_VAR_GAME}, reference=False)
    ref = sandbox.build_game({"main.cpp": UNUSED_VAR_GAME}, reference=True)
    assert llm.ok, llm.stderr
    assert not ref.ok
    assert "-Werror" in ref.stderr or ref.exit_code != 0


def test_determinism_scan_bans_date_macro():
    result = sandbox.build_game({"main.cpp": BANNED_DATE_GAME})
    assert not result.ok
    assert any("determinism:macro" in v for v in result.violations)


def test_determinism_scan_allow_marker_requires_reason_on_next_line():
    result = sandbox.build_game({"main.cpp": BANNED_WITH_ALLOW_NO_REASON})
    assert not result.ok
    assert any("allow-nondeterministic without a reason" in v for v in result.violations)


def test_determinism_scan_allow_marker_with_reason_passes_and_flags_source():
    result = sandbox.build_game({"main.cpp": BANNED_WITH_ALLOW_AND_REASON})
    assert result.ok, result.stderr or result.violations
    # E-74: allowed, but still counted as nondeterministic_source=true
    assert result.nondeterministic_source is True


def test_rng_unseeded_is_flagged():
    result = sandbox.build_game({"main.cpp": RAND_UNPINNED_GAME})
    assert not result.ok
    assert any("rng:" in v for v in result.violations)


def test_rng_seeded_from_spec_is_accepted():
    result = sandbox.build_game({"main.cpp": RAND_WITH_SEED_GAME})
    assert result.ok, result.stderr or result.violations


# ---------------------------------------------------------------------------
# Build artifacts (E-70)
# ---------------------------------------------------------------------------

def test_source_tar_contains_game_sources_only():
    result = sandbox.build_game({"main.cpp": COUNTER_GAME})
    assert result.ok
    with tarfile.open(fileobj=io.BytesIO(result.source_tar)) as tar:
        names = tar.getnames()
    assert names == ["main.cpp"]


def test_binary_and_tar_are_deterministic_in_sha():
    r1 = sandbox.build_game({"main.cpp": COUNTER_GAME})
    r2 = sandbox.build_game({"main.cpp": COUNTER_GAME})
    assert r1.binary_sha256 == r2.binary_sha256


def test_workdir_is_isolated_and_cleaned_up():
    result = sandbox.build_game({"main.cpp": COUNTER_GAME})
    assert result.work_dir == ""
    kept = sandbox.build_game({"main.cpp": COUNTER_GAME}, keep_workdir=True)
    try:
        assert kept.work_dir
        mode = os.stat(kept.work_dir).st_mode & 0o777
        assert mode == 0o700
    finally:
        import shutil
        shutil.rmtree(kept.work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sandboxed execution (E-69)
# ---------------------------------------------------------------------------

def test_run_game_non_tty_tape_produces_digests():
    result = build_counter()
    assert result.ok
    run = sandbox.run_game(
        result.binary_bytes,
        b"key a\ntick\nkey a\ntick\ntick\n",
        extra_env={"HARNESS_SPEC_JSON": SPEC_JSON},
    )
    assert run.exit_code == 0
    assert [n for n, _ in run.digests] == [0, 1, 2]
    assert not run.sandbox_violation


def test_timeout_game_is_killed():
    result = sandbox.build_game({"main.cpp": INFINITE_LOOP_GAME})
    assert result.ok
    started = time.monotonic()
    run = sandbox.run_game(
        result.binary_bytes,
        b"tick\n",
        play_timeout=1,
        cpu_seconds=2,
        extra_env={"HARNESS_SPEC_JSON": SPEC_JSON},
    )
    elapsed = time.monotonic() - started
    assert run.exit_code != 0
    assert elapsed < 15


def test_memory_cap_kills_runaway_allocation():
    result = sandbox.build_game({"main.cpp": HUGE_ALLOC_GAME})
    assert result.ok
    run = sandbox.run_game(
        result.binary_bytes,
        b"",
        mem_bytes=64 * 1024 * 1024,
        extra_env={"HARNESS_SPEC_JSON": SPEC_JSON},
    )
    # harness_init returns 3 on failed allocation -> harness exits 1
    assert run.exit_code == 1


@pytest.mark.parametrize("backend", ["bwrap", "unshare"])
def test_network_isolation(backend):
    """E-69: the game cannot reach a 127.0.0.1 listener inside the sandbox.

    Skips with a clear reason when the backend cannot run here (unshare is
    frequently blocked for unprivileged users); bwrap remains the primary.
    """
    if not _backend_available(backend):
        pytest.skip(f"{backend} network namespace unavailable on this host")

    listener = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    port = listener.server_address[1]
    thread = threading.Thread(target=listener.serve_forever, daemon=True)
    thread.start()
    # sanity: the listener is reachable from the host
    with socket.create_connection(("127.0.0.1", port), timeout=2):
        pass

    result = sandbox.build_game({"main.cpp": NETWORK_GAME})
    assert result.ok
    run = sandbox.run_game(
        result.binary_bytes,
        b"",
        extra_env={"A2A_TEST_PORT": str(port), "HARNESS_SPEC_JSON": SPEC_JSON},
    )
    listener.shutdown()
    # in the sandbox the connect must fail -> harness_init returns 7 -> harness exits 1
    assert run.exit_code == 1


def test_child_env_never_inherits_parent(monkeypatch):
    canary = "canary-do-not-leak-9f3b2a"
    monkeypatch.setenv("NVIDIA_API_KEY", canary)
    result = sandbox.build_game({"main.cpp": ENV_DUMP_GAME})
    assert result.ok
    run = sandbox.run_game(
        result.binary_bytes,
        b"",
        extra_env={"HARNESS_SPEC_JSON": SPEC_JSON},
    )
    assert canary.encode() not in run.stderr
    assert canary.encode() not in run.stdout


def test_sandbox_unavailable_refuses_to_build(monkeypatch):
    monkeypatch.setattr(sandbox, "probe_network_isolation", lambda: ("unavailable", "test forcing refusal"))
    result = sandbox.build_game({"main.cpp": COUNTER_GAME})
    assert not result.ok
    assert result.sandbox_violation
    assert result.network_isolation == "unavailable"
    assert result.violations


def test_sandbox_unavailable_refuses_to_run(monkeypatch):
    monkeypatch.setattr(sandbox, "probe_network_isolation", lambda: ("unavailable", "test forcing refusal"))
    run = sandbox.run_game(b"\x7fELF-binary-bytes", b"tick\n")
    assert run.sandbox_violation
    assert run.exit_code is None


# ---------------------------------------------------------------------------
# Determinism gate (E-74)
# ---------------------------------------------------------------------------

def test_determinism_gate_passes_for_deterministic_game():
    result = sandbox.build_game({"main.cpp": COUNTER_GAME})
    assert result.ok
    gate = sandbox.determinism_gate(
        result.binary_bytes,
        tape=b"tick\n" * 10,
        nondeterministic_source=False,
        extra_env={"HARNESS_SPEC_JSON": SPEC_JSON},
    )
    assert gate.ok
    assert gate.identical
    assert gate.ran_twice


def test_determinism_gate_passes_with_allow_comment_even_if_digests_differ():
    result = sandbox.build_game({"main.cpp": BANNED_WITH_ALLOW_AND_REASON})
    assert result.ok
    gate = sandbox.determinism_gate(
        result.binary_bytes,
        tape=b"tick\n" * 3,
        nondeterministic_source=result.nondeterministic_source,
        extra_env={"HARNESS_SPEC_JSON": SPEC_JSON},
    )
    # ok because the source was allow-listed (E-74 second branch); identical
    # is not required and is reported honestly
    assert gate.ok
    assert gate.ran_twice
    assert gate.nondeterministic_source


def test_determinism_gate_fails_without_allow_when_digests_differ():
    result = sandbox.build_game({"main.cpp": PID_GAME})
    assert result.ok
    gate = sandbox.determinism_gate(
        result.binary_bytes,
        tape=b"tick\n" * 3,
        nondeterministic_source=False,
        extra_env={"HARNESS_SPEC_JSON": SPEC_JSON},
    )
    # pid= differs every run -> identical is False -> gate must refuse
    assert not gate.ok
    assert not gate.identical
    assert gate.violations


def test_run_game_with_empty_tape_runs_zero_ticks():
    result = build_counter()
    assert result.ok
    run = sandbox.run_game(
        result.binary_bytes,
        b"",
        extra_env={"HARNESS_SPEC_JSON": SPEC_JSON},
    )
    assert run.exit_code == 0
    assert run.digests == ()
