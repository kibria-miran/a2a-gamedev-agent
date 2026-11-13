"""Private profiles and utility arithmetic.

Utility for numeric issues uses a normalized [0, 1] scale. Utility for
categorical issues (genre, win_condition, reliability) is a pure lookup
into the frozen tables in ``config`` (E-72); a missing value yields
``default`` with a warning — never a crash and never an invented genre
silently scoring 0.0 in the trace.

E-72 (Profile.load clause): the real Profile loader lands with the
negotiator and must validate that every categorical preference maps to a
key of the corresponding table; ``validate_categorical`` below is the hook
it will call. The v1.0 profile schema is not in this tree yet, so nothing
here invents one.
"""

from __future__ import annotations

import logging
from typing import Any

from src import config

logger = logging.getLogger(__name__)


def norm_categorical(
    value: Any,
    table: dict[str, float],
    default: float = 0.0,
    *,
    table_name: str = "categorical",
) -> float:
    """Return the lookup-table utility of ``value`` (E-72).

    Unknown values resolve to ``default`` and log a warning; they never
    raise. V1/V-series validators reject invented genre strings before the
    utility is computed, so reaching the default branch means the profile
    or spec is inconsistent and the trace should say so.
    """
    if isinstance(value, str) and value in table:
        return float(table[value])
    logger.warning("%s: unknown value %r; using default %.2f", table_name, value, default)
    return float(default)


def validate_categorical(table_name: str, value: Any) -> None:
    """Raise ``ValueError`` unless ``value`` is a key of the named table.

    Tables are the source of truth (E-72). ``table_name`` is one of
    ``genre``, ``win_condition``, ``reliability``.
    """
    table = config.CATEGORICAL_TABLES.get(table_name)
    if table is None:
        raise ValueError(f"unknown categorical table: {table_name!r}")
    if not isinstance(value, str) or value not in table:
        raise ValueError(
            f"{table_name}: {value!r} is not one of {sorted(table)}"
        )
