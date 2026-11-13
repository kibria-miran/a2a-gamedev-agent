# A2A Game Dev Agent

Three independent agents, built separately, negotiate and produce a playable
2D game over a minimal subset of the A2A protocol — same architecture as the
A2A negotiation demo (`pr=a2a-demo===done`), applied to generative game dev.

- **Designer** — owns the game brief; negotiates the spec with the Coder.
- **Coder** — builds the game once the spec is signed; compile-and-fix loop.
- **Critic** — reviews the built binary; the only role that can accept.

Current status: **v1.1 freeze candidate — Phase 1 (VERTICAL SLICE) COMPLETE.**
No server, no A2A, no DB: `python -m src.builder_demo` builds the reference
invaders from the reference brief and plays it through the harness. The A2A
layer (server, client, state machine, decision table) lands in Phases 2–4 as
a thin wrapper around this builder (revised §12).

Remote: https://github.com/kibria-miran/a2a-gamedev-agent

## Layout

```
src/harness/harness.h   # the game contract — written by us, shipped to games
src/harness/harness.c   # main(); tape/tty input; digest printing; watchdog
src/config.py           # frozen constants: CPP flags, artifact caps, lattice,
                        # BUILD_ONCE, categorical utility tables
src/utility.py          # norm_categorical / validate_categorical (E-72)
src/sandbox.py          # prlimit caps + bwrap/unshare net isolation; E-69/70/74
src/llm.py              # stdlib NIM client, LLMError, token accounting
src/prompts.py          # coder prompts; fence stripping; E-13 fix feedback
src/coder_agent.py      # generate_and_build(spec) -> (source, BuildResult)
tests/test_harness.py   # pins the harness contract
tests/test_config.py    # E-68/E-70/E-71/E-72/E-73 assertions
tests/test_utility.py   # E-72 categorical lookup, all three tables
tests/test_sandbox.py   # compile/run sandbox, determinism gate, isolation
tests/test_coder_standalone.py  # S21 broken->fixed, retry exhaustion, prompts
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
