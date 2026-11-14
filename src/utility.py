"""Private profiles and utility arithmetic.

The negotiable game spec has two kinds of issues:

* numeric — ``grid_w``, ``grid_h``, ``tick_ms``, ``enemy_count`` — with an
  ideal, a reservation, a direction (``min``/``max``) and a weight;
* categorical — ``genre``, ``win_condition``, ``reliability`` — with a weight
  and an explicit lookup table of per-value utilities.

Categorical utility is a pure lookup into the frozen tables in ``config``
(E-72). A value missing from the table resolves to ``default`` with a warning;
the validators reject invented values before negotiation ever starts, so
reaching the default branch means something is inconsistent. Never a crash.

Profiles are private: each agent loads only its own JSON file. ``Profile.load``
validates every categorical key against the corresponding frozen table, and
that the weights sum to 1. No I/O happens here except in ``load_profile``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import config

logger = logging.getLogger(__name__)

NUMERIC_ISSUES = ("grid_w", "grid_h", "tick_ms", "enemy_count")
CATEGORICAL_ISSUES = ("genre", "win_condition", "reliability")
ISSUES = NUMERIC_ISSUES + CATEGORICAL_ISSUES

_DIRECTIONS = ("min", "max")
_WEIGHT_TOLERANCE = 1e-6
_DEFAULT_RELIABILITY = "deterministic"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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


@dataclass(frozen=True)
class Issue:
    """A numeric spec issue (section 8.2)."""

    direction: str
    ideal: int
    reservation: int
    weight: float

    def norm(self, value: int) -> float:
        raw = (value - self.reservation) / (self.ideal - self.reservation)
        return max(0.0, min(1.0, raw))

    def acceptable(self, value: int) -> bool:
        if self.direction == "min":
            return value <= self.reservation
        return value >= self.reservation


@dataclass(frozen=True)
class CategoricalIssue:
    """A categorical spec issue (E-72): a weight plus a value->utility map."""

    weight: float
    values: dict[str, float]

    def utility(self, value: str) -> float:
        return norm_categorical(value, self.values, table_name="categorical")

    def acceptable(self, value: str) -> bool:
        return isinstance(value, str) and value in self.values


@dataclass(frozen=True)
class Profile:
    """One agent's private preferences over the negotiated game spec."""

    name: str
    issues: dict[str, Issue]
    categorical: dict[str, CategoricalIssue]
    beta: float

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, data: dict[str, Any]) -> Profile:
        if not isinstance(data, dict):
            raise ValueError("profile must be an object")

        raw_issues = data.get("issues")
        if not isinstance(raw_issues, dict):
            raise ValueError("profile.issues must be an object")

        issues: dict[str, Issue] = {}
        total_weight = 0.0
        for name, raw in raw_issues.items():
            if name not in NUMERIC_ISSUES:
                raise ValueError(f"unknown numeric issue {name!r}")
            if not isinstance(raw, dict):
                raise ValueError(f"issue {name} must be an object")
            direction = raw.get("direction")
            if direction not in _DIRECTIONS:
                raise ValueError(f"issue {name}: direction must be min or max")
            ideal = raw.get("ideal")
            reservation = raw.get("reservation")
            weight = raw.get("weight")
            if (
                not isinstance(ideal, int) or isinstance(ideal, bool)
                or not isinstance(reservation, int) or isinstance(reservation, bool)
            ):
                raise ValueError(f"issue {name}: ideal and reservation must be ints")
            if not isinstance(weight, int | float) or isinstance(weight, bool):
                raise ValueError(f"issue {name}: weight must be a number")
            if ideal == reservation:
                raise ValueError(f"issue {name}: ideal must differ from reservation")
            if direction == "min" and ideal > reservation:
                raise ValueError(f"issue {name}: min direction needs ideal <= reservation")
            if direction == "max" and ideal < reservation:
                raise ValueError(f"issue {name}: max direction needs ideal >= reservation")
            weight_f = float(weight)
            if weight_f < 0:
                raise ValueError(f"issue {name}: weight must be >= 0")
            total_weight += weight_f
            issues[name] = Issue(direction, int(ideal), int(reservation), weight_f)

        missing_numeric = set(NUMERIC_ISSUES) - set(issues)
        if missing_numeric:
            raise ValueError(f"profile.issues missing {sorted(missing_numeric)}")

        raw_categorical = data.get("categorical") or {}
        if not isinstance(raw_categorical, dict):
            raise ValueError("profile.categorical must be an object")
        categorical: dict[str, CategoricalIssue] = {}
        for name, raw in raw_categorical.items():
            if name not in CATEGORICAL_ISSUES:
                raise ValueError(f"unknown categorical issue {name!r}")
            if not isinstance(raw, dict):
                raise ValueError(f"categorical.{name} must be an object")
            weight = raw.get("weight")
            values = raw.get("values")
            if not isinstance(weight, int | float) or isinstance(weight, bool):
                raise ValueError(f"categorical.{name}.weight must be a number")
            if not isinstance(values, dict):
                raise ValueError(f"categorical.{name}.values must be an object")
            weight_f = float(weight)
            if weight_f < 0:
                raise ValueError(f"categorical.{name}.weight must be >= 0")
            table = config.CATEGORICAL_TABLES[name]
            checked: dict[str, float] = {}
            for key, utility in values.items():
                # E-72: every categorical weight maps to a key of the frozen
                # table. An invented genre is a profile bug, not a runtime
                # surprise (the LLM never gets to add one).
                if key not in table:
                    raise ValueError(
                        f"categorical.{name}: {key!r} is not one of {sorted(table)}"
                    )
                if (
                    not isinstance(utility, int | float)
                    or isinstance(utility, bool)
                    or not 0.0 <= float(utility) <= 1.0
                ):
                    raise ValueError(
                        f"categorical.{name}.values[{key!r}] must be in [0, 1]"
                    )
                checked[key] = float(utility)
            total_weight += weight_f
            categorical[name] = CategoricalIssue(weight_f, checked)

        if abs(total_weight - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(f"weights must sum to 1 (got {total_weight})")

        beta = data.get("beta", config.BETA_DEFAULT)
        if not isinstance(beta, int | float) or isinstance(beta, bool) or beta <= 0:
            raise ValueError("beta must be a positive number")

        return cls(
            name=str(data.get("name", "profile")),
            issues=issues,
            categorical=categorical,
            beta=float(beta),
        )

    # -- terms --------------------------------------------------------------

    def _check_terms(self, terms: dict[str, Any]) -> None:
        if not isinstance(terms, dict):
            raise ValueError("terms must be an object")
        for name in NUMERIC_ISSUES:
            if name not in terms:
                raise ValueError(f"terms missing {name}")
            if not _is_int(terms[name]):
                raise ValueError(f"terms.{name} must be a strict int")
        for name in CATEGORICAL_ISSUES:
            value = terms.get(name, _DEFAULT_RELIABILITY if name == "reliability" else None)
            if not isinstance(value, str):
                raise ValueError(f"terms.{name} must be a string")

    def utility(self, terms: dict[str, Any]) -> float:
        self._check_terms(terms)
        total = 0.0
        for name in NUMERIC_ISSUES:
            issue = self.issues[name]
            total += issue.weight * issue.norm(int(terms[name]))
        for name, cat_issue in self.categorical.items():
            value = terms.get(name)
            if value is None and name == "reliability":
                value = _DEFAULT_RELIABILITY
            total += cat_issue.weight * cat_issue.utility(str(value))
        return total

    def issue_utilities(self, terms: dict[str, Any]) -> dict[str, float]:
        self._check_terms(terms)
        out: dict[str, float] = {}
        for name in NUMERIC_ISSUES:
            issue = self.issues[name]
            out[name] = issue.weight * issue.norm(int(terms[name]))
        for name, cat_issue in self.categorical.items():
            value = terms.get(name)
            if value is None and name == "reliability":
                value = _DEFAULT_RELIABILITY
            out[name] = cat_issue.weight * cat_issue.utility(str(value))
        return out

    def feasible(self, terms: dict[str, Any]) -> bool:
        """True when terms are inside every numeric reservation and the
        categorical values exist in their frozen tables."""

        try:
            self._check_terms(terms)
        except ValueError:
            return False
        for name in NUMERIC_ISSUES:
            if not self.issues[name].acceptable(int(terms[name])):
                return False
        for name, issue in self.categorical.items():
            value = terms.get(name)
            if value is None and name == "reliability":
                value = _DEFAULT_RELIABILITY
            if not issue.acceptable(str(value)):
                return False
        return True

    def target_utility(self, round_number: int) -> float:
        """Boulware concession curve: START_U down to END_U at round T."""

        t = max(0, min(int(round_number), config.MAX_ROUNDS))
        fraction = t / config.MAX_ROUNDS
        return config.START_U - (config.START_U - config.END_U) * (fraction**self.beta)

    def best_categorical(self, name: str) -> str:
        issue = self.categorical.get(name)
        if issue is None:
            if name == "reliability":
                return _DEFAULT_RELIABILITY
            raise ValueError(f"profile {self.name!r} has no categorical issue {name!r}")
        return max(sorted(issue.values), key=lambda key: issue.values[key])

    # -- cache key ----------------------------------------------------------

    def key(self) -> tuple[Any, ...]:
        return (
            tuple(
                (name, self.issues[name].direction, self.issues[name].ideal,
                 self.issues[name].reservation, self.issues[name].weight)
                for name in NUMERIC_ISSUES
            ),
            tuple(
                (name, issue.weight, tuple(sorted(issue.values.items())))
                for name, issue in sorted(self.categorical.items())
            ),
            self.beta,
        )


def load_profile(path: str | Path) -> Profile:
    return Profile.load(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "CATEGORICAL_ISSUES",
    "CategoricalIssue",
    "ISSUES",
    "Issue",
    "NUMERIC_ISSUES",
    "Profile",
    "load_profile",
    "norm_categorical",
    "validate_categorical",
]
