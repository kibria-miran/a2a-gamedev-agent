# A2A Game Dev Agent

Three independent agents, built separately, negotiate and produce a playable
2D game over a minimal subset of the A2A protocol — same architecture as the
A2A negotiation demo (`pr=a2a-demo===done`), applied to generative game dev.

- **Designer** — owns the game brief; negotiates the spec with the Coder.
- **Coder** — builds the game once the spec is signed; compile-and-fix loop.
- **Critic** — reviews the built binary; the only role that can accept.

Status: **v1.1 freeze candidate — Phase 2 (A2A thin wrapper) built.**
Server (agent card + JSON-RPC SendMessage/GetTask/CancelTask + R20
build/source fetch routes), client (role-scoped tokens, sha256-verified
artifact fetch), store (§7.0 + `spec_signed`, binary/source_tar artifacts),
E-67 round accounting, REASONS R17–R20. The full lifecycle runs in tests
(offer → counter → spec_signed → one build → R20 fetch+verify → critic
accept → completed). The real negotiator/agents land in Phases 3–4.

Remote: https://github.com/kibria-miran/a2a-gamedev-agent

## Layout

```
src/harness/harness.h   # the game contract — written by us, shipped to games
src/harness/harness.c   # main(); tape/tty input; digest printing; watchdog
src/config.py           # frozen constants: CPP flags, artifact caps, lattice,
                        # BUILD_ONCE, categorical utility tables, A2A pin
src/utility.py          # norm_categorical / validate_categorical (E-72)
src/sandbox.py          # prlimit caps + bwrap/unshare net isolation; E-69/70/74
src/llm.py              # stdlib NIM client, LLMError, token accounting
src/prompts.py          # coder prompts; fence stripping; E-13 fix feedback
src/coder_agent.py      # generate_and_build(spec) -> (source, BuildResult)
src/protocol.py         # A2A wire models (strict spec terms, parts, card)
src/decision.py         # shared Decision object
src/trace.py            # append-only JSONL trace
src/clock.py            # single timestamp source (D-11)
src/state_machine.py    # T1-T8 task lifecycle (unchanged in v1.1)
src/store.py            # SQLite + CAS; §7.0 BLOBs; spec_signed flag
src/server.py           # FastAPI: card + SendMessage/GetTask/CancelTask + R20
src/client.py           # A2A client; fetch_build/fetch_source sha-verified
src/transport.py        # wire message builders
docs/A2A_PIN.md         # pinned wire vocabulary + R-table + lifecycle
tests/                  # unit + integration; ASGI in-memory client/server
```

## Test

```bash
python3 -m pytest tests/
```

Requires `cc` and `g++` (the harness is compiled into every game).
Uses only pytest; no venv needed yet.

## The 3-minute demo

```bash
python3 -m src.builder_demo
```

Reads `evals/reference_brief.json`, builds a playable invaders binary, plays
it through the harness on a scripted tape, and writes the binary to
`data/invaders`. Without `NIM_MODEL`/`NVIDIA_API_KEY` set it uses the
checked-in reference source as the codegen oracle; with them set, the Coder
generates the game from the brief for real (still no server, no A2A, no DB).

## The harness in one paragraph

`main()` is provided by the harness; a game supplies `harness_init`,
`harness_tick`, `harness_render`. Input arrives as a line-oriented tape on
stdin (non-tty: `tick` / `key <c>` / `quit`) or as raw keypresses (tty). Every
tick prints one digest line `tick N <fnv1a32-of-rendered-frame>`; the Critic
parses exactly those lines and nothing else. The game exits when the tick
budget (tape length) is exhausted, on `quit`, on `harness_exit(code)`, or when
the `HARNESS_PLAY_TIMEOUT` watchdog fires (exit 124). RNG is seeded from the
canonical spec JSON (FNV-1a) before `harness_init()`.
