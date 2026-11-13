"""Tests for the game harness contract (§8.11, E-64/E-65/E-74).

The harness is the linchpin of the Critic: it defines how input is read
(tty vs non-tty tape), what is printed per tick ("tick N <fnv1a32>"), and
when the game exits. These tests pin that contract so the Critic can rely
on it.

Both stdio modes print the same digest format. In tty mode the terminal
line discipline translates '\\n' -> '\\r\\n' (ONLCR), so digest lines are
parsed with splitlines() rather than compared byte-wise.
"""

import os
import pty
import re
import select
import shutil
import subprocess
import termios
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = ROOT / "src" / "harness"
HARNESS_C = HARNESS_DIR / "harness.c"
HARNESS_H = HARNESS_DIR / "harness.h"

CC = shutil.which("cc")
CXX = shutil.which("g++")
pytestmark = pytest.mark.skipif(not CC or not CXX, reason="cc/g++ toolchain required")

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def fnv1a32(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


TICK_RE = re.compile(r"^tick (\d+) ([0-9a-f]{8})$")

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
    else if (k == 'k') x += 10;
}
const char* harness_render(void) {
    static char buf[64];
    std::snprintf(buf, sizeof buf, "x=%d", x);
    return buf;
}
"""

EXIT_GAME = """
#include "harness.h"
#include <cstdio>
static int n = 0;
int harness_init(void) { return 0; }
void harness_tick(void) {
    ++n;
    if (n == 3) harness_exit(42);
}
const char* harness_render(void) {
    static char buf[64];
    std::snprintf(buf, sizeof buf, "n=%d", n);
    return buf;
}
"""

FAILING_INIT_GAME = """
#include "harness.h"
int harness_init(void) { return 1; }
void harness_tick(void) {}
const char* harness_render(void) { return "never"; }
"""

CONFIG_GAME = """
#include "harness.h"
#include <cstdio>
int harness_init(void) { return 0; }
void harness_tick(void) {}
const char* harness_render(void) {
    static char buf[64];
    const harness_config_t* cfg = harness_config();
    std::snprintf(buf, sizeof buf, "w=%d win=%d max=%d", cfg->grid_w, cfg->win_condition, cfg->max_ticks);
    return buf;
}
"""


def build_game(tmp_path: Path, source: str, name: str = "game.cpp") -> Path:
    assert HARNESS_C.exists() and HARNESS_H.exists()
    src = tmp_path / name
    src.write_text(source)
    harness_o = tmp_path / "harness.o"
    game_o = tmp_path / (name + ".o")
    exe = tmp_path / "game"
    steps = [
        [CC, "-std=c11", "-O1", "-Wall", "-Wextra",
         "-I", str(HARNESS_DIR), "-c", str(HARNESS_C), "-o", str(harness_o)],
        [CXX, "-std=c++17", "-O2", "-Wall", "-Wextra",
         "-I", str(HARNESS_DIR), "-c", str(src), "-o", str(game_o)],
        [CXX, str(game_o), str(harness_o), "-o", str(exe)],
    ]
    for cmd in steps:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.fail(f"build step failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return exe


def clean_env(**extra: str) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("HARNESS_")
    }
    env["HARNESS_SPEC_JSON"] = SPEC_JSON
    env.update(extra)
    return env


def run_game(exe: Path, tape: bytes | None, env_extra: dict | None = None, timeout: float = 10.0) -> subprocess.CompletedProcess:
    env = clean_env(**(env_extra or {}))
    return subprocess.run(
        [str(exe)],
        input=tape,
        capture_output=True,
        timeout=timeout,
        env=env,
    )


def digest_lines(stdout: bytes) -> list[tuple[int, str]]:
    out = []
    for line in stdout.decode(errors="replace").splitlines():
        m = TICK_RE.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def test_ten_line_tape_exactly_ten_digest_lines(tmp_path):
    exe = build_game(tmp_path, COUNTER_GAME)
    proc = run_game(exe, b"tick\n" * 10)
    assert proc.returncode == 0
    digests = digest_lines(proc.stdout)
    assert len(digests) == 10
    assert [n for n, _ in digests] == list(range(10))
    assert all(re.fullmatch(r"[0-9a-f]{8}", h) for _, h in digests)
    # stdout carries ONLY digest lines
    for line in proc.stdout.decode().splitlines():
        assert TICK_RE.match(line), f"unexpected stdout line: {line!r}"


def test_digests_match_fnv1a_of_rendered_frame(tmp_path):
    exe = build_game(tmp_path, COUNTER_GAME)
    tape = b"key a\ntick\nkey a\ntick\nkey d\ntick\nkey k\ntick\ntick\n"
    proc = run_game(exe, tape)
    assert proc.returncode == 0
    digests = digest_lines(proc.stdout)
    xs = [1, 2, 1, 11, 11]
    expected = [(i, f"{fnv1a32(f'x={v}'.encode()):08x}") for i, v in enumerate(xs)]
    assert digests == expected


def test_non_tty_output_is_deterministic(tmp_path):
    exe = build_game(tmp_path, COUNTER_GAME)
    tape = b"key a\ntick\ntick\nkey d\ntick\ntick\n"
    p1 = run_game(exe, tape)
    p2 = run_game(exe, tape)
    assert p1.returncode == p2.returncode == 0
    assert p1.stdout == p2.stdout


def test_quit_directive_still_prints_final_frame(tmp_path):
    exe = build_game(tmp_path, COUNTER_GAME)
    proc = run_game(exe, b"key a\ntick\nkey a\ntick\nquit\n")
    assert proc.returncode == 0
    digests = digest_lines(proc.stdout)
    assert [n for n, _ in digests] == [0, 1, 2]
    assert digests[-1][1] == f"{fnv1a32(b'x=2'):08x}"


def test_unknown_directive_is_rejected(tmp_path):
    exe = build_game(tmp_path, COUNTER_GAME)
    proc = run_game(exe, b"frobnicate\n")
    assert proc.returncode == 3
    assert digest_lines(proc.stdout) == []
    assert b"unknown directive" in proc.stderr


def test_key_directive_without_argument_is_rejected(tmp_path):
    exe = build_game(tmp_path, COUNTER_GAME)
    proc = run_game(exe, b"key\n")
    assert proc.returncode == 3


def test_harness_exit_ends_loop_with_game_code(tmp_path):
    exe = build_game(tmp_path, EXIT_GAME)
    proc = run_game(exe, b"tick\ntick\ntick\ntick\ntick\n")
    assert proc.returncode == 42
    digests = digest_lines(proc.stdout)
    assert [n for n, _ in digests] == [0, 1, 2]
    assert digests[-1][1] == f"{fnv1a32(b'n=3'):08x}"


def test_harness_init_failure_aborts_before_any_tick(tmp_path):
    exe = build_game(tmp_path, FAILING_INIT_GAME)
    proc = run_game(exe, b"tick\ntick\n")
    assert proc.returncode == 1
    assert digest_lines(proc.stdout) == []
    assert b"harness_init" in proc.stderr


def test_config_comes_from_spec_json(tmp_path):
    exe = build_game(tmp_path, CONFIG_GAME)
    tape = b"tick\n"
    a = run_game(exe, tape, {"HARNESS_SPEC_JSON": '{"grid_w": 40, "win_condition": "survive"}'})
    b = run_game(exe, tape, {"HARNESS_SPEC_JSON": '{"grid_w": 33, "win_condition": "clear"}'})
    assert a.returncode == b.returncode == 0
    assert digest_lines(a.stdout)[0][1] != digest_lines(b.stdout)[0][1]


def test_win_condition_string_maps_to_codes(tmp_path):
    exe = build_game(tmp_path, CONFIG_GAME)
    tape = b"tick\n"
    expected = {"survive": 0, "clear": 1, "score": 2}
    digests = {}
    for name in expected:
        proc = run_game(exe, tape, {"HARNESS_SPEC_JSON": f'{{"win_condition": "{name}"}}'})
        assert proc.returncode == 0
        digests[name] = digest_lines(proc.stdout)[0][1]
    # survive/clear/score must produce three distinct digests, in a stable order
    assert len(set(digests.values())) == 3


def test_pty_mode_produces_same_digests_as_non_tty(tmp_path):
    """Under a pty the harness reads raw keypresses; with a deterministic
    burst (all keys queued before the start-delay expires) the digest
    sequence must equal the equivalent non-tty tape (E-65)."""
    exe = build_game(tmp_path, COUNTER_GAME)

    ref = run_game(exe, b"key a\ntick\nkey a\ntick\nkey d\ntick\nkey d\ntick\ntick\ntick\n")
    assert ref.returncode == 0
    ref_digests = digest_lines(ref.stdout)
    assert len(ref_digests) == 6

    env = clean_env(
        HARNESS_MAX_TICKS="6",
        HARNESS_TTY_START_DELAY_MS="300",
        HARNESS_PLAY_TIMEOUT="5",
    )
    pid, master = pty.fork()
    if pid == 0:  # child
        try:
            os.execvpe(str(exe), [str(exe)], env)
        except Exception:
            os._exit(127)

    # the pty pair still has ECHO/ICANON from before fork; disable both on the
    # master BEFORE writing, or the burst is echoed and glues onto the first
    # digest line. termios lives on the pty, not the process, and the child
    # re-applies the same raw config after exec (idempotent).
    attrs = termios.tcgetattr(master)
    attrs[0] &= ~termios.ICRNL      # c_iflag
    attrs[3] &= ~(termios.ECHO | termios.ICANON)  # c_lflag
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(master, termios.TCSANOW, attrs)

    os.write(master, b"aadd")
    chunks: list[bytes] = []
    deadline = time.monotonic() + 8.0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail("pty child did not exit within 8 s")
            ready, _, _ = select.select([master], [], [], remaining)
            if not ready:
                pytest.fail("pty child produced no output / hung")
            try:
                chunk = os.read(master, 4096)
            except OSError:  # EIO: slave closed, child exited
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(master)

    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0

    pty_digests = digest_lines(b"".join(chunks))
    assert [n for n, _ in pty_digests] == [0, 1, 2, 3, 4, 5]
    assert pty_digests == ref_digests


def test_pty_mode_ignores_tape_eof_and_exits_at_budget(tmp_path):
    """A pty child with no input must stop at HARNESS_MAX_TICKS and exit 0."""
    exe = build_game(tmp_path, COUNTER_GAME)
    env = clean_env(HARNESS_MAX_TICKS="3", HARNESS_PLAY_TIMEOUT="5", HARNESS_TTY_START_DELAY_MS="50")
    pid, master = pty.fork()
    if pid == 0:
        try:
            os.execvpe(str(exe), [str(exe)], env)
        except Exception:
            os._exit(127)

    chunks: list[bytes] = []
    deadline = time.monotonic() + 8.0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail("pty child did not exit within 8 s")
            ready, _, _ = select.select([master], [], [], remaining)
            if not ready:
                pytest.fail("pty child hung with no input")
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(master)

    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    digests = digest_lines(b"".join(chunks))
    assert len(digests) == 3
