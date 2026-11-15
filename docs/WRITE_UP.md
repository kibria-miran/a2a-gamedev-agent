# A2A Game Dev Agent — Write-up (v1.1)

Three independent A2A agents — **Designer**, **Coder**, **Critic** — negotiate
a game spec, build a playable 2D game, and verify the artifact before anyone
declares victory. No agent can do another's job: only the Designer signs
(`spec_signed`), the Coder builds exactly once (E-68), only the Critic
accepts. This document summarizes what the system does, what was measured on
this machine, and where it is honest to stop trusting it.

## Architecture (as built)

- **Coder service** (`src/coder_server.py` → `src/server.py`): one public
  agent card, role-scoped bearer tokens (`A2A_TOKEN_DESIGNER` /
  `A2A_TOKEN_CRITIC`), JSON-RPC SendMessage/GetTask/CancelTask, decision
  rows R0–R21 (+ `ALREADY_SIGNED` for the one-build invariant). R20 routes
  serve the built binary and the source tar; sha256 verified at both ends.
- **Designer** (`src/designer_agent.py`): opens from the brief, concedes
  along a beta concession curve, signs when the peer's terms clear its
  target. Never accepts.
- **Coder** (`src/coder_agent.py`): counters (echoing acceptable peer terms
  for signoff, E-66), builds once after `spec_signed`, walks away on
  failure. Never accepts.
- **Critic** (`src/critic_agent.py`): fetches the binary over R20, verifies
  sha256/size, re-derives the source allow-list by re-scanning the attested
  source tar, replays a scripted tape twice (determinism gate, E-74), and
  only then accepts. The allow-listed branch still requires both runs to
  agree on the final frame digest — noise tolerated, divergence rejected.
  Its own terminal action is `accept` or `walk_away`.
- **Negotiator** (`src/negotiator.py`): frozen 36,864-candidate lattice
  fallback; LLM proposals validated V1–V7 (schema, feasible, band, monotone,
  leak, instruction, markup); *code decides acceptance* (D-5). Utilities are
  per-role private profiles (`profiles/designer.json`,
  `profiles/coder.json`); categorical issues use lookup tables (E-72).
- **Sandbox** (`src/sandbox.py`): prlimit caps (`--as`, `--cpu`),
  `bwrap --unshare-net` primary with `unshare -n` fallback, no-net or
  nothing (E-69), read-only root inside bwrap, minimal child env, fresh
  `mkdtemp` workdir, no `preexec_fn` anywhere.
- **Harness** (`src/harness/harness.h[c]`): the game contract; `main()` is
  harness-provided; every frame prints `tick N <fnv1a-of-render>`; RNG is
  seeded from the canonical spec JSON (E-74).

## Measured results (this machine, 2026-11-14)

Offline evals (`--all-offline`, `runs=10`; results in
`evals/results/*-fallback.jsonl`; single g++ build of the reference invaders
reused across runs):

| Experiment | Finding |
|---|---|
| X1 beta grid (3×3) | Agreement 1.00 in every cell — the code-decided protocol always converges on this profile pair. Joint utility differentiates: 0.58 → 1.69 with beta (designer-leaning cells reach 1.69; stubborn-Coder cells land at 0.58). |
| X4 MAX_ROUNDS 4/8/12 | Agreement 1.00; the round budget is absorbed, not binding. |
| X6 band 0.03/0.05/0.10 | Arms identical by construction — the band validates the LLM proposer only; the offline code path bypasses V3. Use `--mode live` for a meaningful X6. |
| X7 reproducibility | All 10 offline transcripts byte-identical (artifactIds scrubbed). PASS. |
| X8 one-build audit | 10/10 completed tasks contain exactly one `build` message (E-68). p50 = 1.11 s, p95 = 2.07 s per task. PASS. |
| X1–X6 | 0 wire leaks recorded (rationale never enters data parts by default; `include_rationale=True` is the experimental X3 arm). |

Security suite (offline, `tests/test_security.py`, 15 tests):

- injection corpus (9 payloads × designer proposer, coder proposer, build
  brief, fix prompt) — payloads are dropped, the raw text never reaches the
  model, `injection_attempt` lands in the trace, agent output stays feasible;
- leak probes (7 probes) — no private values or leak phrases on the wire;
  retry prompts carry only the V-reason, never the rejected text;
- **S22** — escape-hatch calls (`system`, `popen`, `exec*`, `fork`, `socket`,
  `connect`, `ptrace`, …) rejected *before* the compiler runs; allow-comment
  escape hatch (`// allow-banned-call` + reason) mirrors E-74;
- **S24** — adversarial binary output: forged well-formed digest lines parse
  as ordinary ticks (they are untrusted but harmless), protocol-shaped and
  malformed junk is dropped, 1 MB of junk parses in <1 s;
- output cap: a game printing ~10 MB is killed and the play fails; parent
  memory is bounded (`config.MAX_OUTPUT_BYTES`).

Slow tier (`tests/test_slow.py`, real subprocesses, `make slow`):

- **S9** — kill -9 the Coder mid-build; restart; resend the same messageId;
  R6c resume runs the build again; final history holds exactly one `build`;
  the fetched binary matches the descriptor sha256.
- **S10a** — two concurrent `SendMessage` with the same messageId produce
  one task, one stored offer, one agent reply.
- **S10b** — an `accept` posted during the build turn gets -32602 (R8).
- **S14** — fake NIM sleeping past `LLM_TIMEOUT` yields a clean
  `walk_away(llm_error)` rejection with zero artifacts.

Live smoke (account with exactly one usable NIM model — most listed models
return 404 "Function not found for account"; `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
is what the key can call today):

- end-to-end (`python -m src.three_agent_demo`): `spec_signed → build →
  critic accept → TASK_STATE_COMPLETED`; one build; 26 deterministic frames;
  final tick digest `988abd12` (differs from the offline oracle's
  `dcf77ff3` — genuinely the model-generated game);
- **X2 (n=2)** — LLM proposers on both sides reach joint utility 1.933 on
  the stress brief; the code-decided fallback reaches 1.075 (converges near
  reservations). The LLM is not better because of magic; it is better
  because it proposes instead of conceding lattice steps;
- **X3 (n=2)** — 0 rationales with leak phrases on the wire in either arm;
- **X3 (n=10, 2026-11-15)** — both arms 10/10 agreement; 0 rationale leaks
  on the wire; the `rationale_on` arm passes the new hard gate (X3 exits
  nonzero on any leaked rationale). Latency mean 46 s/37 s per arm, max
  288 s; tokens per arm ≈9.7k in / 38.8k out (≈970 in per run). Rationale
  on the wire did not change agreement (1.00 both arms).
- **X5** — cannot run here: the account enables a single usable model id.
  Use `--designer-model/--coder-model` with two enabled models.
- **X6 (live)** — not run here; see X6 note above.

## Honest limitations

- **Agreement is engineered, not emergent.** The code-decided fallback
  protocol agrees 1.00 everywhere on this profile pair; the interesting
  signal is joint utility, not completion. A profile pair engineered near
  infeasibility would behave differently — that is a v1.1+ experiment, not a
  bug here.
- **Offline X6 is vacuous** (V3 guards the LLM path only); likewise the
  fallback never produces rejected proposals, so `fallback` totals are
  optimal-but-rigid.
- **Live sample sizes are small** (X2 n=2; X3 n=10). X3 shows 206–288 s
  latency outliers (reasoning model) — treat the numbers as directional,
  not significant.
- **Model availability is account-gated.** `step-5-preview`,
  `minimax-m3.1-flash`, `deepseek-v4.1-flash` all 200 on smoke tests but
  hang or return zero tokens on large codegen prompts; the usable model is
  a small reasoning model with visible latency variance.
- **`unshare -n` is unavailable on this host** (test skipped); `bwrap`
  works. On a host with neither, the sandbox *refuses to build* (E-69) —
  it does not fall back to an unisolated compile.
- **Reproducibility** covers spec transcripts (artifactIds scrubbed); the
  two offline builds of the reference invaders are byte-identical, but the
  sandbox makes no promise about builds on other toolchains/hardening flags.
- **S22 is defense in depth**, not a sandbox boundary; the real boundary is
  bwrap + prlimit + no-net + minimal env. The escape hatch (`// allow-banned-call`)
  exists so the runtime isolation can be tested; it is visible in source,
  like `// allow-nondeterministic`.

## Token budget (E-68)

Offline: 0. Live reference brief: ~450 tokens/run. Live stress brief:
~2.3k tokens/run (nemotron-3-nano). Worst observed live run ≈ 4.7k in /
16.7k out across 2 runs — comfortably under the 15k/task budget for the
reference-brief path; the LLM proposer path with retries is what would
consume it first, and `LLM_RETRY_ATTEMPTS = 1` keeps that bounded.

## What I would do next

1. A second enabled NIM model → run X5 for real; a non-reasoning model
   would also cut the X3 latency outlier.
2. Digest-forge detection research: identical-line spam filtering inside
   the Critic (currently indistinguishable from honest ticks by design —
   that is why the critic does not consume binary stdout).

Done since v1.1: borderline-infeasibility profile pair + stress brief —
`evals/infeasible_{designer,coder}.json`, demonstrated in
`tests/test_infeasible.py` (agreement is 0.0 by construction: the lattices
are disjoint, the demo rejects without a build, `make check` and CI guard
it). The slow tier (kill-9 resume, concurrency, LLM-timeout) now also
runs in CI (`.github/workflows/ci.yml`). Per-run rationale leak scanning
is recorded (`wire_leaks`, `seed`) in `evals/results/*.jsonl` and
`summarize()` prints worst-run/leaking-ids; X3 prints a per-run verdict.

## Reproduce

```bash
make check                     # fast tier + ruff + mypy
make slow                      # S9/S10/S14, real subprocesses
python3 -m src.three_agent_demo   # offline-or-live end-to-end
.venv/bin/python evals/run_evals.py --all-offline --runs 10
.venv/bin/python evals/run_evals.py --experiment x2 --mode live --runs 10
```
