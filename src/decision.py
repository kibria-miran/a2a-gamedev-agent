"""Small decision object shared by the coder agent and the server commit path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Decision:
    kind: str  # counter | spec_signed | build | accept | walk_away | cancel | resend | wait
    terms: dict[str, Any] | None = None
    round: int = 0
    rationale: str = ""
    own_u: float | None = None
    source: str = "code"  # llm | fallback | code
    message: dict[str, Any] | None = field(default=None)
    reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    build_result: Any | None = None  # sandbox.BuildResult for kind == "build"
