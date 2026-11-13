"""Tests for src/config.py (E-68, E-70, E-71, E-72, E-73)."""

from src import config


def test_cpp_flags_llm_has_no_werror():
    assert "-Werror" not in config.CPP_FLAGS_LLM
    assert config.CPP_FLAGS_LLM == ["-std=c++17", "-O2", "-Wall", "-Wextra"]


def test_cpp_flags_reference_extends_llm_with_werror():
    assert config.CPP_FLAGS_REFERENCE == config.CPP_FLAGS_LLM + ["-Werror"]


def test_build_once_asserted():
    # E-68: exactly one build per task, after spec_signed.
    assert config.BUILD_ONCE is True
    assert config.COMPILE_RETRIES == 3
    assert config.TOKEN_BUDGET_PER_TASK == 15_000


def test_max_artifact_bytes_is_4_mib():
    assert config.MAX_ARTIFACT_BYTES == 4 * 1024 * 1024 == 4_194_304


def test_max_frame_digests_to_llm():
    assert config.MAX_FRAME_DIGESTS_TO_LLM == 64


def test_lattice_size_is_36864():
    # E-71: 4 x 8 x 8 x 6 x 8 x 3
    assert len(config.GENRES) == 4
    assert len(config.GRID_W_LATTICE) == 8
    assert len(config.GRID_H_LATTICE) == 8
    assert len(config.TICK_MS_LATTICE) == 6
    assert len(config.ENEMY_COUNT_LATTICE) == 8
    assert len(config.WIN_CONDITIONS) == 3
    assert config.LATTICE_SIZE == 36_864


def test_win_condition_codes_match_harness_contract():
    # harness.c maps survive/clear/score -> 0/1/2 (§8.11); keep them in sync.
    assert config.WIN_CONDITION_CODE == {"survive": 0, "clear": 1, "score": 2}
    assert set(config.WIN_CONDITION_CODE) == set(config.WIN_CONDITIONS)


def test_categorical_tables_exact_contents():
    assert config.GENRE_UTILITY == {"invaders": 1.0, "breakout": 0.7, "snake": 0.5, "maze": 0.2}
    assert config.WIN_CONDITION_UTILITY == {"survive": 1.0, "clear": 0.7, "score": 0.3}
    assert config.RELIABILITY_UTILITY == {
        "deterministic": 1.0,
        "declared_nondeterministic": 0.5,
        "undeclared_nondeterministic": 0.0,
    }
    for table in config.CATEGORICAL_TABLES.values():
        assert all(0.0 <= v <= 1.0 for v in table.values())


def test_categorical_tables_registry_matches_constants():
    assert config.CATEGORICAL_TABLES["genre"] is config.GENRE_UTILITY
    assert config.CATEGORICAL_TABLES["win_condition"] is config.WIN_CONDITION_UTILITY
    assert config.CATEGORICAL_TABLES["reliability"] is config.RELIABILITY_UTILITY
