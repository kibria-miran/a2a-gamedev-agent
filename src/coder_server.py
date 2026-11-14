"""Run the Coder service: ``python -m src.coder_server``.

The same service talks to both peers, distinguished only by role-scoped
bearer tokens (E-38/E-66). Set ``A2A_TOKEN_DESIGNER`` and ``A2A_TOKEN_CRITIC``
before starting; without them every non-card route fails closed.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from src import config, trace  # noqa: E402
from src.coder_agent import load_coder_profile  # noqa: E402
from src.server import create_app  # noqa: E402
from src.store import Store  # noqa: E402


def build_app():
    store = Store(config.CODER_DB_PATH)
    trace.configure(config.TRACE_PATH)
    return create_app(store, profile=load_coder_profile(), card_url=config.CODER_URL)


app = build_app()


def main() -> int:
    import uvicorn

    if not (config.A2A_TOKEN_DESIGNER and config.A2A_TOKEN_CRITIC):
        print(
            "warning: A2A_TOKEN_DESIGNER / A2A_TOKEN_CRITIC are unset; "
            "all non-card routes will return 401 (E-38 fail-closed)."
        )
    print(f"Coder agent card: {config.CODER_URL}{config.AGENT_CARD_PATH}")
    uvicorn.run(app, host=config.CODER_HOST, port=config.CODER_PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
