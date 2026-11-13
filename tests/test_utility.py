"""Tests for src/utility.py — categorical utility (E-72)."""

import logging

import pytest

from src import config, utility

ALL_TABLES = [
    ("genre", config.GENRE_UTILITY, "invaders"),
    ("win_condition", config.WIN_CONDITION_UTILITY, "survive"),
    ("reliability", config.RELIABILITY_UTILITY, "deterministic"),
]


@pytest.mark.parametrize(("table_name", "table", "example"), ALL_TABLES)
def test_categorical_lookup_known_keys(table_name, table, example):
    for key, value in table.items():
        assert utility.norm_categorical(key, table, table_name=table_name) == value


@pytest.mark.parametrize(("table_name", "table", "example"), ALL_TABLES)
def test_categorical_lookup_unknown_value_warns_and_defaults(table_name, table, example, caplog):
    with caplog.at_level(logging.WARNING, logger="src.utility"):
        assert utility.norm_categorical("nonsense-invented-by-llm", table, table_name=table_name) == 0.0
    assert "unknown value" in caplog.text
    assert table_name in caplog.text


def test_categorical_lookup_custom_default():
    assert utility.norm_categorical("bogus", config.GENRE_UTILITY, default=0.42) == 0.42


def test_categorical_lookup_non_string_value_warns_and_defaults(caplog):
    with caplog.at_level(logging.WARNING, logger="src.utility"):
        assert utility.norm_categorical(None, config.WIN_CONDITION_UTILITY, table_name="win_condition") == 0.0
        assert utility.norm_categorical(7, config.RELIABILITY_UTILITY, table_name="reliability") == 0.0
    assert caplog.text.count("unknown value") == 2


def test_categorical_lookup_never_crashes():
    for weird in ("", " ", "INVADERS", "maze ", "maze\n"):
        assert utility.norm_categorical(weird, config.GENRE_UTILITY, default=0.0) in (0.0, 0.2)


@pytest.mark.parametrize(("table_name", "table", "example"), ALL_TABLES)
def test_validate_categorical_accepts_table_keys(table_name, table, example):
    for key in table:
        utility.validate_categorical(table_name, key)  # must not raise


@pytest.mark.parametrize(("table_name", "table", "example"), ALL_TABLES)
def test_validate_categorical_rejects_unknown(table_name, table, example):
    with pytest.raises(ValueError, match=table_name):
        utility.validate_categorical(table_name, "definitely-not-a-key")


def test_validate_categorical_rejects_unknown_table_name():
    with pytest.raises(ValueError, match="unknown categorical table"):
        utility.validate_categorical("genre2", "invaders")
