"""Shared test fixtures: FakeLLM and the offline network guard."""

from __future__ import annotations

import copy
import socket
from typing import Any

import pytest

from src import config


@pytest.fixture(autouse=True)
def _no_real_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may open a socket to a non-local host.

    Loopback (127.0.0.1 / localhost) is allowed so sandbox/network tests can
    reach local listeners; anything else fails loudly.
    """
    real_connect = socket.socket.connect

    def guard(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(f"network access to {host!r} is forbidden (fast tier must be offline)")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guard)


class FakeLLM:
    """Scripted stand-in for ``src.llm.chat``."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        role: str = "coder",
        temperature: float = 0.3,
    ) -> str:
        self.calls.append(
            {
                "role": role,
                "messages": copy.deepcopy(messages),
                "temperature": temperature,
            }
        )
        from src import llm

        usage = llm.token_usage(role)
        usage.input += 900
        usage.output += 700
        if self._index >= len(self._responses):
            pytest.fail(
                "FakeLLM exhausted: the test staged fewer responses than the "
                "agent requested (stage one response per expected attempt)"
            )
        response = self._responses[self._index]
        self._index += 1
        return response

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.chat(*args, **kwargs)

    @property
    def count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self._index = 0
        self.calls.clear()
