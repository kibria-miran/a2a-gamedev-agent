"""Thin NVIDIA NIM client (OpenAI-compatible), stdlib-only, with token accounting.

``LLMError`` is a normal negotiation/build failure (E-50): the caller retries
or falls back; it is not a ``BaseException``. No network call happens unless a
model and an API key are configured — see ``default_chat``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src import config

REASON_LLM_ERROR = "LLM error"


class LLMError(Exception):
    """Network/timeout/non-2xx/malformed-envelope failure (E-50)."""


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0


_USAGE: dict[str, TokenUsage] = {}


def reset_token_usage() -> None:
    _USAGE.clear()


def token_usage(role: str) -> TokenUsage:
    return _USAGE.setdefault(role, TokenUsage())


def token_usage_all() -> dict[str, TokenUsage]:
    return {name: TokenUsage(value.input, value.output) for name, value in _USAGE.items()}


def usage_delta(before: dict[str, TokenUsage]) -> tuple[int, int]:
    """Return (tokens_in, tokens_out) accumulated since ``before`` was taken."""

    after = token_usage_all()
    roles = set(before) | set(after)
    tokens_in = sum(
        after.get(role, TokenUsage()).input - before.get(role, TokenUsage()).input
        for role in roles
    )
    tokens_out = sum(
        after.get(role, TokenUsage()).output - before.get(role, TokenUsage()).output
        for role in roles
    )
    return tokens_in, tokens_out


def chat(
    messages: list[dict[str, Any]],
    role: str = "coder",
    temperature: float = config.LLM_TEMPERATURE,
    model: str | None = None,
) -> str:
    """Call the chat completions endpoint and return the raw content string."""

    selected_model = model or config.NIM_MODEL
    if not selected_model:
        raise LLMError("NIM_MODEL is not configured")
    if not config.NVIDIA_API_KEY:
        raise LLMError("NVIDIA_API_KEY is not configured")

    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": config.LLM_MAX_TOKENS,
    }
    request = urllib.request.Request(
        config.NIM_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise LLMError(f"LLM request failed: HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc

    try:
        envelope = json.loads(body)
        content = envelope["choices"][0]["message"]["content"]
        usage = envelope.get("usage") or {}
        entry = _USAGE.setdefault(role, TokenUsage())
        entry.input += int(usage.get("prompt_tokens", 0) or 0)
        entry.output += int(usage.get("completion_tokens", 0) or 0)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMError("malformed LLM envelope") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMError("malformed LLM envelope")
    return content


def default_chat(model: str | None = None) -> Any | None:
    """The real client only when a model and a key are configured.

    ``model`` overrides ``config.NIM_MODEL`` for this client instance (X5).
    """

    selected = model or config.NIM_MODEL
    if not selected or not config.NVIDIA_API_KEY:
        return None

    def _chat(
        messages: list[dict[str, Any]],
        role: str = "coder",
        temperature: float = config.LLM_TEMPERATURE,
    ) -> str:
        return chat(messages, role=role, temperature=temperature, model=selected)

    return _chat
