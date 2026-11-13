"""Standalone Coder agent (Phase 1): spec in, binary out. No server, no A2A.

``generate_and_build`` is the whole loop: one LLM codegen call, then up to
``config.COMPILE_RETRIES`` compile attempts inside the single build turn
(E-68: exactly one build turn, never a renegotiated spec). A failed attempt
feeds the stable build log — never the LLM's rejection of it — back into a
fix prompt (E-13).

Returns ``(source, BuildResult)``; callers (Phase 4) walk away with
``build_failed`` if the result is not ok.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from src import config, llm, prompts, sandbox

logger = logging.getLogger(__name__)

Chat = Callable[..., str]

ROLE = "coder"
BUILD_LOG_CHARS = 6000


def _build_log(result: sandbox.BuildResult) -> str:
    """Stable, model-facing failure summary (E-13)."""
    parts: list[str] = []
    if result.violations:
        parts.append("validator violations:\n" + "\n".join(f"- {v}" for v in result.violations))
    if result.stderr.strip():
        parts.append(result.stderr.strip()[-BUILD_LOG_CHARS:])
    if result.exit_code is not None:
        parts.append(f"exit_code={result.exit_code}")
    return "\n".join(parts) if parts else "build produced no diagnostic output"


def _canonicalize(spec: dict[str, Any]) -> dict[str, Any]:
    """Canonical spec JSON: sorted keys, compact separators (the seed source)."""
    return json.loads(json.dumps(spec, sort_keys=True, separators=(",", ":")))


def generate_and_build(
    spec: dict[str, Any],
    chat: Chat | None = None,
) -> tuple[str, sandbox.BuildResult]:
    """Generate main.cpp from ``spec`` and build it in the sandbox.

    ``chat`` injects the LLM call (tests pass FakeLLM); when omitted, the
    real NIM client is used via ``llm.default_chat()`` and an unconfigured
    endpoint raises ``LLMError`` (E-50).
    """
    canonical = _canonicalize(spec)
    client: Chat | None = chat or llm.default_chat()
    if client is None:
        raise llm.LLMError("NIM_MODEL/NVIDIA_API_KEY not configured; cannot codegen offline")

    source = ""
    messages = prompts.build_coder_messages(canonical)
    last_result: sandbox.BuildResult | None = None

    for attempt in range(1, config.COMPILE_RETRIES + 1):
        raw = client(messages, role=ROLE, temperature=config.LLM_TEMPERATURE)
        extracted = prompts.extract_source(raw)
        if extracted is None:
            build_log = "the reply did not contain extractable C++ source"
            source = raw
            last_result = sandbox.BuildResult(ok=False, violations=("extract:no source in reply",))
            logger.info("coder attempt %d: unparseable reply", attempt)
            messages = prompts.build_fix_messages(canonical, source, build_log)
            continue
        source = extracted
        result = sandbox.build_game({"main.cpp": source})
        last_result = result
        if result.ok:
            logger.info("coder attempt %d: build ok", attempt)
            return source, result
        logger.info(
            "coder attempt %d: build failed (%s)",
            attempt,
            "; ".join(result.violations) or f"exit_code={result.exit_code}",
        )
        build_log = _build_log(result)
        messages = prompts.build_fix_messages(canonical, source, build_log)

    assert last_result is not None
    return source, last_result
