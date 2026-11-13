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
# LLM / NIM (values come from the environment; never logged)
# ---------------------------------------------------------------------------
# Stdlib-only client (src/llm.py). NVIDIA NIM speaks OpenAI-compatible chat
# completions; a real run needs NIM_MODEL and NVIDIA_API_KEY set.
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = os.getenv("NIM_MODEL", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

LLM_TIMEOUT = float(os.getenv("A2A_LLM_TIMEOUT", "60.0"))
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 8_192  # a whole .cpp file comfortably fits; do not starve it

