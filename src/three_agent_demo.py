"""The headline three-agent demo (S26), in one process.

``python -m src.three_agent_demo``

Designer → Coder (over the real A2A routes, in-process ASGI) → Critic.
Without ``NIM_MODEL``/``NVIDIA_API_KEY`` the Coder uses the checked-in
reference invaders as the offline codegen oracle, exactly like the Phase 1
builder demo. With them, the same flow generates from the brief for real.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

from src import config, critic_agent, designer_agent, history, sandbox, trace  # noqa: E402
from src.client import A2AClient  # noqa: E402
from src.coder_agent import load_coder_profile, make_responder  # noqa: E402
from src.designer_cli import load_brief, run_designer  # noqa: E402
from src.server import create_app  # noqa: E402
from src.store import Store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO_DESIGNER_TOKEN = "demo-designer-token"
DEMO_CRITIC_TOKEN = "demo-critic-token"


async def run_demo(
    *,
    db_path: str = ":memory:",
    brief_path: str | Path = "evals/reference_brief.json",
    designer_token: str | None = None,
    critic_token: str | None = None,
    trace_path: str | None = None,
    verbose: bool = True,
    designer_profile: Any | None = None,
    coder_profile: Any | None = None,
    codegen: Any | None = None,
    designer_chat: Any | None = None,
    coder_chat: Any | None = None,
    include_rationale: bool = False,
) -> dict[str, Any]:
    """Run the full lifecycle and return the final task plus binary bytes.

    ``designer_profile`` / ``coder_profile`` / ``codegen`` are injected by the
    evals runner; production callers leave them unset and get the checked-in
    profiles and the standard offline/LLM codegen path.
    """

    designer_token = designer_token or config.A2A_TOKEN_DESIGNER or DEMO_DESIGNER_TOKEN
    critic_token = critic_token or config.A2A_TOKEN_CRITIC or DEMO_CRITIC_TOKEN
    if trace_path:
        trace.configure(trace_path)

    store = Store(db_path)
    coder_profile = coder_profile or load_coder_profile()
    app = create_app(
        store,
        responder=make_responder(profile=coder_profile, chat=coder_chat, codegen=codegen),
        designer_token=designer_token,
        critic_token=critic_token,
        trace_agent="coder",
        include_rationale=include_rationale,
    )
    brief = load_brief(brief_path)
    designer_profile = designer_profile or designer_agent.load_designer_profile()
    binary, sha256 = b"", ""

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=config.HTTP_TIMEOUT
        ) as designer_http:
            designer = A2AClient(
                "http://testserver", token=designer_token, http_client=designer_http
            )
            task_id, designer_task = await run_designer(
                designer, profile=designer_profile, brief=brief, verbose=verbose,
                chat=designer_chat,
            )
            signed_view = await designer.get_task(task_id)

        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=config.HTTP_TIMEOUT
        ) as critic_http:
            critic = A2AClient(
                "http://testserver", token=critic_token, http_client=critic_http
            )
            decision = await critic_agent.review(signed_view, critic)
            if verbose:
                print(
                    f"critic  : {decision.kind} round {decision.round}"
                    + (f" ({decision.reason})" if decision.reason else "")
                )
                if decision.rationale:
                    print(f"critic  : {decision.rationale}")

            from src.transport import accept_parts, message_dict, walk_away_parts

            if decision.kind in {"complete", "rejected", "canceled", "failed"}:
                # The task is already terminal on the server (e.g. the build
                # walk-away); there is nothing to send.
                if verbose:
                    print(f"critic  : task already {decision.kind}; nothing to send")
                final = signed_view
            else:
                if decision.kind == "accept":
                    artifact = (decision.message or {}).get("artifact")
                    parts = accept_parts(decision.round, build=artifact)
                else:
                    parts = walk_away_parts(decision.round, decision.reason)
                final = await critic.send_message(
                    message_dict(
                        role="user", parts=parts, task_id=task_id,
                        context_id=signed_view.get("contextId"),
                    )
                )

            artifact = history.build_artifact(final)
            if artifact is not None and decision.kind == "accept":
                binary, sha256, _size = await critic.fetch_build(
                    task_id, artifact["artifactId"]
                )
    finally:
        store.close()

    return {
        "task_id": task_id,
        "designer_task": designer_task,
        "task": final,
        "decision": decision,
        "binary": binary,
        "sha256": sha256,
        "artifact": history.build_artifact(final),
    }


def _main(args: argparse.Namespace) -> int:
    result = asyncio.run(
        run_demo(
            db_path=args.db,
            brief_path=args.brief,
            trace_path=args.trace,
            verbose=True,
        )
    )
    task = result["task"]
    state = task["status"]["state"]
    print(f"\nstate   : {state}")
    print(f"task    : {result['task_id']}")
    spec = history.latest_spec_terms(task)
    if spec:
        print(f"spec    : {json.dumps(spec, sort_keys=True, separators=(',', ':'))}")
    artifact = result["artifact"]
    if artifact:
        print(f"artifact: {artifact['artifactId']} sha256={artifact['sha256']}")
    builds = [
        message for message in history.history(task)
        if (history.data_of(message) or {}).get("type") == "build"
    ]
    print(f"builds  : {len(builds)} (E-68: exactly one)")

    if state != "TASK_STATE_COMPLETED" or not result["binary"]:
        print("demo FAILED")
        return 1

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "invaders-a2a"
    out_path.write_bytes(result["binary"])
    out_path.chmod(0o755)
    print(f"binary  : {out_path} ({len(result['binary'])} bytes)")

    play = sandbox.run_game(
        result["binary"], config.CRITIC_TAPE,
        extra_env={
            "HARNESS_SPEC_JSON": json.dumps(spec, sort_keys=True, separators=(",", ":")),
            "HARNESS_PLAY_TIMEOUT": "10",
        },
    )
    print(f"play    : exit_code={play.exit_code} digests={len(play.digests)}")
    print("demo OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=config.CODER_DB_PATH)
    parser.add_argument("--brief", default="evals/reference_brief.json")
    parser.add_argument("--trace", default=config.TRACE_PATH)
    args = parser.parse_args()
    trace.configure(args.trace)
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
