"""Critic CLI: fetch, verify and play the build; accept or walk away.

``python -m src.critic_cli --task <task-id> --url http://127.0.0.1:8003 \\
    --token "$A2A_TOKEN_CRITIC"``

Only the Critic can accept (E-66); the accept carries the verified artifact
descriptor from the R20 fetch.
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

from src import config, critic_agent  # noqa: E402
from src.client import A2AClient  # noqa: E402
from src.transport import accept_parts, message_dict, walk_away_parts  # noqa: E402


async def run_critic(client: A2AClient, task_id: str, *, verbose: bool = True) -> dict:
    view = await client.get_task(task_id)
    decision = await critic_agent.review(view, client)
    if verbose:
        print(
            f"critic: {decision.kind} round {decision.round}"
            + (f" ({decision.reason})" if decision.reason else "")
        )
        if decision.rationale:
            print(f"critic: {decision.rationale}")
    if decision.kind == "accept":
        artifact = (decision.message or {}).get("artifact")
        parts = accept_parts(decision.round, build=artifact)
    elif decision.kind == "walk_away":
        parts = walk_away_parts(decision.round, decision.reason)
    else:
        return view

    message = message_dict(role="user", parts=parts, task_id=task_id,
                           context_id=view.get("contextId"))
    return await client.send_message(message)


async def _main(args: argparse.Namespace) -> int:
    token = args.token or config.A2A_TOKEN_CRITIC
    if not token:
        print("refusing to run without a Critic token (A2A_TOKEN_CRITIC)")
        return 2
    async with A2AClient(args.url, token=token) as client:
        card = await client.fetch_card()
        print(f"critic  : talking to {card['name']}")
        task = await run_critic(client, args.task)
    print(f"critic  : task {args.task} is {task['status']['state']}")
    return 0 if task["status"]["state"] == "TASK_STATE_COMPLETED" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--url", default=config.CODER_URL)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
