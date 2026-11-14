"""Designer CLI: negotiate the spec and sign it off.

``python -m src.designer_cli --brief evals/reference_brief.json \\
    --url http://127.0.0.1:8003 --token "$A2A_TOKEN_DESIGNER"``

Prints the task id; run ``python -m src.critic_cli --task ...`` afterwards.
The Designer never accepts (E-66): its terminal action is ``spec_signed``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src import config, designer_agent  # noqa: E402
from src.client import A2AClient  # noqa: E402
from src.transport import (  # noqa: E402
    message_dict,
    offer_parts,
    spec_signed_parts,
    walk_away_parts,
)

MAX_STEPS = 20


def load_brief(path: str | Path | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parts_for(decision) -> list[dict]:
    if decision.kind == "spec_signed":
        return spec_signed_parts(decision.round, decision.terms or {})
    if decision.kind == "counter":
        from src.transport import counter_parts

        return counter_parts(decision.round, decision.terms or {})
    if decision.kind == "walk_away":
        return walk_away_parts(decision.round, decision.reason)
    return offer_parts(decision.round, decision.terms or {})


async def run_designer(
    client: A2AClient,
    *,
    profile=None,
    brief: dict | None = None,
    chat=None,
    verbose: bool = True,
) -> tuple[str, dict]:
    """Drive the Designer loop until signoff/terminal. Returns (task_id, view)."""

    profile = profile or designer_agent.load_designer_profile()
    opening = designer_agent.opening_terms(profile, brief)
    message = message_dict(role="user", parts=offer_parts(0, opening))
    task = await client.send_message(message)
    task_id = task["id"]
    if verbose:
        print(f"designer: offer round 0 -> {task['status']['state']}")

    for _ in range(MAX_STEPS):
        decision = designer_agent.next_move(task, profile, brief=brief, chat=chat)
        if verbose:
            print(
                f"designer: {decision.kind} round {decision.round}"
                + (f" ({decision.reason})" if decision.reason else "")
            )
        if decision.kind in {"complete", "rejected", "canceled", "failed"}:
            return task_id, task
        if decision.kind == "wait":
            return task_id, task
        if decision.kind == "resend" and decision.message is not None:
            task = await client.send_message(decision.message)
            continue

        message = message_dict(
            role="user", parts=_parts_for(decision),
            task_id=task_id, context_id=task.get("contextId"),
        )
        task = await client.send_message(message)
        if decision.kind in {"spec_signed", "walk_away"}:
            return task_id, task

    raise RuntimeError("designer exceeded MAX_STEPS without signoff")


async def _main(args: argparse.Namespace) -> int:
    token = args.token or config.A2A_TOKEN_DESIGNER
    if not token:
        print("refusing to run without a Designer token (A2A_TOKEN_DESIGNER)")
        return 2
    profile = designer_agent.load_designer_profile(args.profile)
    brief = load_brief(args.brief)
    async with A2AClient(args.url, token=token) as client:
        card = await client.fetch_card()
        print(f"designer: talking to {card['name']}")
        task_id, task = await run_designer(client, profile=profile, brief=brief)
    print(f"designer: task {task_id} is {task['status']['state']}")
    print(f"next    : python -m src.critic_cli --task {task_id} --url {args.url}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=config.CODER_URL)
    parser.add_argument("--token", default=None)
    parser.add_argument("--profile", default=config.DESIGNER_PROFILE_PATH)
    parser.add_argument("--brief", default="evals/reference_brief.json")
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
