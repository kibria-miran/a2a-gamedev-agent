# A2A Game Dev Agent

Three independent agents, built separately, negotiate and produce a playable
2D game over a minimal subset of the A2A protocol — same architecture as the
A2A negotiation demo (`pr=a2a-demo===done`), applied to generative game dev.

- **Designer** — owns the game brief; negotiates the spec with the Coder.
- **Coder** — builds the game once the spec is signed; compile-and-fix loop.
- **Critic** — reviews the built binary; the only role that can accept.

Status: **v1.1 — Phases 1–4 complete.** The full three-agent lifecycle runs
end to end: `offer → counter → spec_signed → build (once) → R20 fetch+verify →
critic accept → completed`. The service (agent card, JSON-RPC
SendMessage/GetTask/CancelTask, R17–R21 guards, R20 binary/source routes)
enforces one build per task via its own `spec_signed` state; the Designer and
Coder negotiate from private utility profiles over a finite proposal lattice;
only the Critic can accept, after fetching the binary and replaying a scripted
tape. Without a model configured, the Coder falls back to the checked-in
reference source as an offline codegen oracle.

Remote: https://github.com/kibria-miran/a2a-gamedev-agent

## Layout

```
src/harness/harness.h   # the game contract — written by us, shipped to games
src/harness/harness.c   # main(); tape/tty input; digest printing; watchdog
src/config.py           # frozen constants: CPP flags, artifact caps, lattice,
                        # BUILD_ONCE, categorical utility tables, A2A pin
src/utility.py          # Profile, numeric/categorical utility (E-72)
src/negotiator.py       # feasible lattice, fallback offers, accept/reject,
                        # proposal validation (E-66)
src/designer_agent.py   # open/concede/sign/walk-away strategy
src/coder_agent.py      # responder: counter or single build after signoff
src/critic_agent.py     # R20 fetch, sha verify, play, determinism gate, accept
src/history.py          # task-view introspection for the agents
src/prompts.py          # LLM prompts; proposal parsing; fence stripping
src/llm.py              # stdlib NIM client, LLMError, token accounting
src/sandbox.py          # prlimit caps + bwrap/unshare net isolation; E-69/70/74
src/protocol.py         # A2A wire models (strict spec terms, parts, card)
src/decision.py         # shared Decision object
src/trace.py            # append-only JSONL trace
src/clock.py            # single timestamp source (D-11)
src/state_machine.py    # T1-T8 task lifecycle
src/store.py            # SQLite + CAS; §7.0 BLOBs; spec_signed flag
src/server.py           # FastAPI: card + SendMessage/GetTask/CancelTask + R20
src/client.py           # A2A client; fetch_build/fetch_source sha-verified
src/transport.py        # wire message builders
src/coder_server.py     # run the Coder service (uvicorn)
src/designer_cli.py     # Designer peer: negotiate and sign off
src/critic_cli.py       # Critic peer: review a task and accept/walk away
src/three_agent_demo.py # S26: all three agents, one process, real routes
profiles/               # designer.json / coder.json — private utility profiles
docs/A2A_PIN.md         # pinned wire vocabulary + R-table + lifecycle
tests/                  # unit + integration; ASGI in-memory client/server
```

## Test

```bash
make check          # fast tests + ruff + mypy
make fast           # pytest -m "not slow"
make all            # everything, including slow compiles
```

Requires `cc` and `g++` (the harness is compiled into every game). Uses a
venv at `.venv` by default (`PYTHON=...` to override); install
`requirements.txt` plus `pytest`, `ruff`, and `mypy`.

## The 3-minute demo

All three agents, one process, real A2A routes:

```bash
python3 -m src.three_agent_demo
```

Reads `evals/reference_brief.json`, negotiates a spec, builds a playable
invaders binary, has the Critic fetch it over R20, verify sha256, play a
scripted tape, check determinism, and accept. Writes the binary under
`data/invaders-a2a/` and the trace to `logs/trace.jsonl`. Without
`NIM_MODEL`/`NVIDIA_API_KEY` the Coder uses the checked-in reference source as
the codegen oracle; with them set, it generates from the brief for real.

To run the agents as separate processes:

```bash
python3 -m src.coder_server
python3 -m src.designer_cli --brief evals/reference_brief.json \
    --url http://127.0.0.1:8003 --token "$A2A_TOKEN_DESIGNER"
python3 -m src.critic_cli --task <task-id> \
    --url http://127.0.0.1:8003 --token "$A2A_TOKEN_CRITIC"
```

The Designer CLI prints the task id; pass it to the Critic. Copy
`.env.example` to `.env` for tokens and model settings.

## The harness in one paragraph

`main()` is provided by the harness; a game supplies `harness_init`,
`harness_tick`, `harness_render`. Input arrives as a line-oriented tape on
stdin (non-tty: `tick` / `key <c>` / `quit`) or as raw keypresses (tty). Every
tick prints one digest line `tick N <fnv1a32-of-rendered-frame>`; the Critic
parses exactly those lines and nothing else. The game exits when the tick
budget (tape length) is exhausted, on `quit`, on `harness_exit(code)`, or when
the `HARNESS_PLAY_TIMEOUT` watchdog fires (exit 124). RNG is seeded from the
canonical spec JSON (FNV-1a) before `harness_init()`.
