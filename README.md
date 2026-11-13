# A2A Game Dev Agent

Three independent agents, built separately, negotiate and produce a playable
2D game over a minimal subset of the A2A protocol — same architecture as the
A2A negotiation demo (`pr=a2a-demo===done`), applied to generative game dev.

- **Designer** — owns the game brief; negotiates the spec with the Coder.
- **Coder** — builds the game once the spec is signed; compile-and-fix loop.
- **Critic** — reviews the built binary; the only role that can accept.

Current status: **v1.1 freeze candidate, Phase 1 (vertical slice) in progress.**
This tree contains the harness contract (§8.11, E-64/E-65/E-74) only; the A2A
layer, sandbox, config, and evals land in later phases (revised §12).

## Layout

```
src/harness/harness.h   # the game contract — written by us, shipped to games
src/harness/harness.c   # main(); tape/tty input; digest printing; watchdog
tests/test_harness.py   # pins the contract (10-line tape, pty, FNV digests)
```

## Test

```bash
python3 -m pytest tests/test_harness.py
```

Requires `cc` and `g++` (the harness is compiled into every game).
Uses only pytest; no venv needed yet.

## The harness in one paragraph

`main()` is provided by the harness; a game supplies `harness_init`,
`harness_tick`, `harness_render`. Input arrives as a line-oriented tape on
stdin (non-tty: `tick` / `key <c>` / `quit`) or as raw keypresses (tty). Every
tick prints one digest line `tick N <fnv1a32-of-rendered-frame>`; the Critic
parses exactly those lines and nothing else. The game exits when the tick
budget (tape length) is exhausted, on `quit`, on `harness_exit(code)`, or when
the `HARNESS_PLAY_TIMEOUT` watchdog fires (exit 124). RNG is seeded from the
canonical spec JSON (FNV-1a) before `harness_init()`.
