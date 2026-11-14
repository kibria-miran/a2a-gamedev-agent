"""Frozen configuration for the A2A game dev agent.

Everything in this module is a constant. Private profile values never live
here; see ``profiles/*.json``. No I/O happens at import time beyond reading
environment variables (never writes, never touches the network).

This file carries the E-71/E-72/E-73 constants from v1.1 and the Phase 0
config additions. The A2A wire vocabulary and server constants land with
the server phase (revised §12, Phase 2); the harness owns its own runtime
knobs (``HARNESS_*``) in ``src/harness/harness.c``.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# A2A protocol pin (see docs/A2A_PIN.md; same architecture as the demo)
# ---------------------------------------------------------------------------

A2A_SPEC_VERSION = "1.0.0"
A2A_PROTOCOL_VERSION = "1.0"
A2A_VERSION_HEADER = "A2A-Version"

AGENT_CARD_PATH = "/.well-known/agent-card.json"
AGENT_CARD_ALIAS_PATH = "/.well-known/agent.json"
A2A_RPC_PATH = "/"

A2A_METHOD_SEND = "SendMessage"
A2A_METHOD_GET_TASK = "GetTask"
A2A_METHOD_CANCEL_TASK = "CancelTask"
A2A_METHODS = frozenset({A2A_METHOD_SEND, A2A_METHOD_GET_TASK, A2A_METHOD_CANCEL_TASK})

# Standard JSON-RPC 2.0 error codes.
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

# A2A-specific error codes (spec section 5.4; R20 shares TASK_NOT_FOUND's -32001).
A2A_TASK_NOT_FOUND = -32001
A2A_TASK_NOT_CANCELABLE = -32002
A2A_PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
A2A_UNSUPPORTED_OPERATION = -32004
A2A_CONTENT_TYPE_NOT_SUPPORTED = -32005
A2A_INVALID_AGENT_RESPONSE = -32006
A2A_EXTENDED_AGENT_CARD_NOT_CONFIGURED = -32007
A2A_EXTENSION_SUPPORT_REQUIRED = -32008
A2A_VERSION_NOT_SUPPORTED = -32009

# Internal (database / state machine) names. Do not rename these without a
# migration; the wire mapping below is the only place A2A spellings appear.
TASK_STATES = (
    "submitted",
    "working",
    "input-required",
    "completed",
    "canceled",
    "failed",
    "rejected",
)
TERMINAL_STATES = frozenset({"completed", "canceled", "failed", "rejected"})

ROLES = ("user", "agent")

A2A_STATE_BY_INTERNAL = {
    "submitted": "TASK_STATE_SUBMITTED",
    "working": "TASK_STATE_WORKING",
    "input-required": "TASK_STATE_INPUT_REQUIRED",
    "completed": "TASK_STATE_COMPLETED",
    "canceled": "TASK_STATE_CANCELED",
    "failed": "TASK_STATE_FAILED",
    "rejected": "TASK_STATE_REJECTED",
}
INTERNAL_STATE_BY_A2A = {wire: internal for internal, wire in A2A_STATE_BY_INTERNAL.items()}

A2A_ROLE_BY_INTERNAL = {"user": "ROLE_USER", "agent": "ROLE_AGENT"}
INTERNAL_ROLE_BY_A2A = {wire: internal for internal, wire in A2A_ROLE_BY_INTERNAL.items()}

# ---------------------------------------------------------------------------
# Negotiation curve and tolerance (v1.1 Phase 4; same shape as the demo)
# ---------------------------------------------------------------------------
# Boulware concession: target utility falls from START_U to END_U over
# MAX_ROUNDS. The code decides acceptance (D-5/E-12); the LLM may only
# propose. EPS_* are tolerances, never loosen them to hide a bug.
MAX_ROUNDS = 8
START_U = 0.90
END_U = 0.10
BETA_DEFAULT = 2.0
BAND_LOW = 0.05
BAND_HIGH = 0.10
EPS_ACCEPT = 0.03
EPS_MONOTONE = 0.02
EPS_BAND = 0.03

# Private profiles live outside config; each agent reads only its own file.
PROFILES_DIR = os.getenv("A2A_PROFILES_DIR", "profiles")
DESIGNER_PROFILE_PATH = os.getenv("A2A_DESIGNER_PROFILE", "profiles/designer.json")
CODER_PROFILE_PATH = os.getenv("A2A_CODER_PROFILE", "profiles/coder.json")

# The Critic plays a fixed, deterministic tape over every build. Keeping it
# in config makes the review reproducible across processes.
CRITIC_TAPE = (
    b"key a\nkey d\nkey k\n"
    + b"tick\n" * 10
    + b"key k\ntick\n"
    + b"tick\n" * 15
)

# ---------------------------------------------------------------------------
# Caps, limits, tolerance, storage
# ---------------------------------------------------------------------------
MAX_MESSAGE_BYTES = 16_384
MAX_OPEN_TASKS = 20
MAX_HISTORY_RESPONSE = 20
MAX_RATIONALE_CHARS = 400

SQLITE_BUSY_TIMEOUT_MS = 5_000
TRACE_U_DECIMALS = 2
TRACE_PATH = os.getenv("A2A_TRACE_PATH", "logs/trace.jsonl")
CODER_DB_PATH = os.getenv("A2A_CODER_DB", "data/coder.db")

# ---------------------------------------------------------------------------
# Stable reason strings (tests reference these keys, never literals).
# All <= 60 chars (E-29). R17-R20 arrive with v1.1 (E-66/E-70).
# ---------------------------------------------------------------------------
REASONS = {
    "R4": "server busy",
    "R6A": "messageId already used in a different task",
    "R6D": "turn in progress for this message",
    "R7": "task is terminal",
    "R8": "task is busy",
    "R9": "round mismatch",
    "R10": "accept mismatch",
    "RESUME_DRIFT": "resume round drift",
    "R17": "spec_signed not from designer",
    "R18": "build before spec_signed",
    "R19": "accept not from critic or before signoff",
    "R20": "artifact not found",
    "R21": "spec message not from designer",
    "ALREADY_SIGNED": "spec already signed",
}

# ---------------------------------------------------------------------------
# Agent roles and their bearer tokens (E-66/R17/R19: the server must know who
# is who — messages are otherwise indistinguishable). Tokens come from the
# environment; empty means "not configured" and the server must fail closed.
# ---------------------------------------------------------------------------
A2A_TOKEN_DESIGNER = os.getenv("A2A_TOKEN_DESIGNER", "")
A2A_TOKEN_CRITIC = os.getenv("A2A_TOKEN_CRITIC", "")

# ---------------------------------------------------------------------------
# Compile toolchain (E-73)
# ---------------------------------------------------------------------------
# LLM-authored code never gets -Werror; reference games in
# evals/reference_games/ compile with -Werror enforced.
CPP_FLAGS_LLM = ["-std=c++17", "-O2", "-Wall", "-Wextra"]
CPP_FLAGS_REFERENCE = CPP_FLAGS_LLM + ["-Werror"]

# One build per task, after spec_signed (E-68). COMPILE_RETRIES counts g++
# invocations *inside* the single build turn (fix-and-retry loop). The
# worst case stays ~12k tokens against the 15k budget:
#   8 spec rounds (~1 LLM call each, ~4k tokens)
# + 1 build (~5k) + fix retries (~2k) + critic review (~1k)  ≈ 12k tokens < 15k.
BUILD_ONCE = True
COMPILE_RETRIES = 3
TOKEN_BUDGET_PER_TASK = 15_000

# ---------------------------------------------------------------------------
# Artifact caps (E-70)
# ---------------------------------------------------------------------------
# A 600-line C++ game with -O2 is ~50 KB; 4 MiB is generous.
MAX_ARTIFACT_BYTES = 4_194_304  # 4 MiB

# Hard cap on bytes captured from any sandboxed child's stdout/stderr.
# Idle/malicious games that spam output would otherwise grow the parent's
# memory without bound (E-69 spirit). 4 MiB is far beyond any tape run.
MAX_OUTPUT_BYTES = 4_194_304  # 4 MiB

# The Critic's prompt receives at most this many evenly sampled frame-digest
# lines, plus the final line and the exit code (§8.11).
MAX_FRAME_DIGESTS_TO_LLM = 64

# ---------------------------------------------------------------------------
# Fallback lattice (E-71)
# ---------------------------------------------------------------------------
# Enumerated by the deterministic fallback when the LLM cannot produce a
# spec. 4 genres x 8 grid_w x 8 grid_h x 6 tick_ms x 8 enemy_count
# x 3 win_condition = 36,864 iterations per fallback call (~5 ms in Python).
GENRES = ("invaders", "breakout", "snake", "maze")
GRID_W_LATTICE = (16, 20, 24, 28, 32, 40, 48, 64)      # 8
GRID_H_LATTICE = (12, 16, 20, 24, 28, 32, 40, 48)      # 8
TICK_MS_LATTICE = (50, 75, 100, 125, 150, 200)         # 6
ENEMY_COUNT_LATTICE = (2, 4, 6, 8, 10, 12, 14, 16)     # 8
WIN_CONDITIONS = ("survive", "clear", "score")          # 3
LATTICE_SIZE = (
    len(GENRES)
    * len(GRID_W_LATTICE)
    * len(GRID_H_LATTICE)
    * len(TICK_MS_LATTICE)
    * len(ENEMY_COUNT_LATTICE)
    * len(WIN_CONDITIONS)
)  # 36,864 (E-71); asserted by tests/test_config.py

# win_condition codes exposed to games via harness_config_t.win_condition;
# must match the string mapping in src/harness/harness.c (§8.11).
WIN_CONDITION_CODE = {"survive": 0, "clear": 1, "score": 2}

# ---------------------------------------------------------------------------
# Categorical utility lookup tables (E-72)
# ---------------------------------------------------------------------------
# Categorical issues have no numeric reservation/ideal scale, so they cannot
# use the numeric normalizer; the utility is a pure lookup. Missing values
# resolve to 0.0 with a warning — never a crash (the LLM may invent a genre;
# the validators reject it before utility is computed).
GENRE_UTILITY = {
    "invaders": 1.0,
    "breakout": 0.7,
    "snake": 0.5,
    "maze": 0.2,
}
WIN_CONDITION_UTILITY = {
    "survive": 1.0,
    "clear": 0.7,
    "score": 0.3,
}
RELIABILITY_UTILITY = {
    "deterministic": 1.0,
    "declared_nondeterministic": 0.5,
    "undeclared_nondeterministic": 0.0,
}
CATEGORICAL_TABLES = {
    "genre": GENRE_UTILITY,
    "win_condition": WIN_CONDITION_UTILITY,
    "reliability": RELIABILITY_UTILITY,
}

# ---------------------------------------------------------------------------
# Address of the Coder service (the card URL and clients derive from this)
# ---------------------------------------------------------------------------
CODER_HOST = os.getenv("A2A_CODER_HOST", "127.0.0.1")
CODER_PORT = int(os.getenv("A2A_CODER_PORT", "8003"))
CODER_URL = f"http://{CODER_HOST}:{CODER_PORT}"

# ---------------------------------------------------------------------------
# Concurrency and polling (seconds unless stated otherwise)
# ---------------------------------------------------------------------------
BLOCKING_TIMEOUT = float(os.getenv("A2A_BLOCKING_TIMEOUT", "45.0"))
HTTP_TIMEOUT = float(os.getenv("A2A_HTTP_TIMEOUT", "60.0"))
POLL_MAX = float(os.getenv("A2A_POLL_MAX", "60.0"))
POLL_INTERVAL = 2.0
MAX_RETRY_TOTAL = 15.0

# ---------------------------------------------------------------------------
# LLM / NIM (values come from the environment; never logged)
# ---------------------------------------------------------------------------
# Stdlib-only client (src/llm.py). NVIDIA NIM speaks OpenAI-compatible chat
# completions; a real run needs NIM_MODEL and NVIDIA_API_KEY set.
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = os.getenv("NIM_MODEL", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

LLM_TIMEOUT = float(os.getenv("A2A_LLM_TIMEOUT", "60.0"))
LLM_TEMPERATURE = 0.3
LLM_RETRY_ATTEMPTS = 1  # one retry after the first failure (D-13)
LLM_MAX_TOKENS = 8_192  # a whole .cpp file comfortably fits; do not starve it

