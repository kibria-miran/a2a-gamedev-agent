# A2A Wire Pin (v1.0 + v1.1 deltas)

Pinned wire vocabulary for the three-agent game dev loop. Implementation:
`src/server.py` (Coder service), `src/client.py` (Designer/Critic), `src/protocol.py` (models).

The peers themselves live in `src/designer_agent.py`, `src/coder_agent.py`, and
`src/critic_agent.py`; `src/negotiator.py` supplies the proposal lattice and
acceptance rule. `src/coder_server.py`, `src/designer_cli.py`,
`src/critic_cli.py`, and `src/three_agent_demo.py` are the runnable entry
points.

## Methods

| JSON-RPC method | Semantics |
|---|---|
| `SendMessage` | Send one message into a task (creates a task if taskless). |
| `GetTask` | Fetch the task view. `historyLength` clamped server-side. |
| `CancelTask` | Cancel a non-terminal task. Idempotent if already canceled. |

JSON-RPC 2.0 envelope. Errors: `-32700` parse, `-32600` invalid request,
`-32601` method not found, `-32602` invalid params, `-32603` internal,
`-32001` task/artifact not found, `-32002` task not cancelable.

## Roles and auth (E-38, E-66)

Three agents, two clients of the Coder service. The **only public
(unauthenticated) route is the agent card GET**. Everything else is bearer:

- `A2A_TOKEN_DESIGNER` — Designer token: `offer`, `counter`, `spec_signed`, `walk_away`
- `A2A_TOKEN_CRITIC` — Critic token: `accept`, `walk_away`

The token *is* the role identity; there is no role field on the wire that
the server trusts. Unset token ⇒ the service fails closed.

| Route | Auth | Response |
|---|---|---|
| `GET /.well-known/agent-card.json` (alias `/agent-card.json`) | none | 200 card JSON |
| `POST /` (JSON-RPC) | `Authorization: Bearer <token>` | 401 or JSON-RPC result/error |
| `GET /tasks/{id}/build/{artifact_id}` | bearer (either role) | 200 binary bytes + `X-Build-Sha256` + `Content-Length`; 404 `{"error":{"code":-32001,"message":"artifact not found"}}` |
| `GET /tasks/{id}/source/{artifact_id}` | bearer (either role) | 200 tar bytes + same headers |

The Critic fetches the binary via the build route, **recomputes sha256 and
rejects on mismatch** (E-70; the client does this for you in
`client.A2AClient.fetch_build`).

## Parts

Exactly one content field per part (`text` or `data`). The structured
payload is a single `data` part:

```json
{"type": "offer|counter|spec_signed", "round": 3, "terms": {…}}
{"type": "build",    "round": 3, "artifact": {"artifactId","name","sha256","size"}}
{"type": "accept",   "round": 3, "build": {"artifactId","sha256"}}
{"type": "walk_away","round": 3, "reason": "build_failed"}
```

Spec `terms` are strictly typed (E-7): ints are never coerced from
floats/bools; `genre`, `win_condition`, `reliability` must be members of the
frozen tables (`config.GENRE_UTILITY`, `WIN_CONDITION_UTILITY`,
`RELIABILITY_UTILITY`) — unknown values are `-32602`, never silently
defaulted (V1 validators run before utility is computed).

## Round accounting (E-47 superseded by E-67; v1.1 ruling)

`round` is the **index of the spec message inside the task** and commits +1
for every spec message (`offer`, `counter`, `spec_signed`, `walk_away`) by
either role; `build`, `accept`, `failed` deliveries commit +0.

- Designer sends even rounds, the Coder odd.
- `R9` (incoming round == `tasks.round`) applies only to spec messages.
- The critic's `accept` carries the round of the task (`round` =
  `tasks.round`, no increment) and is not an R9-checked message.
- `spec_signed` carries `round` = its own index; it **is** a spec message.

State names on the wire: `TASK_STATE_*` (e.g. `TASK_STATE_INPUT_REQUIRED`).
Roles: `ROLE_USER`, `ROLE_AGENT`.

## Decision rows (R-table)

R0 size → R1 JSON → R2 envelope/method → R3 schema → R4 busy → R5 unknown
task → R6a/b/c/d replay/turn-in-progress → R7 terminal → R8 busy turn → R9
round mismatch → R10 accept mismatch → **R17 spec_signed not from designer
→ R18 build before spec_signed → R19 accept not from critic or before
signoff → R20 artifact not found** → R21 spec message not from designer.

`ALREADY_SIGNED` (`-32602`, "spec already signed") is the E-68 guard layered on
top of those rows: once a task has `spec_signed`, another `spec_signed` is
rejected rather than opening a second build turn. Combined with R18, this
makes "one build per task" a server-side invariant, not a client convention.

G1–G3 (`GetTask` param check / clamp / not found) and C1–C5 (`CancelTask`
param check / idempotent / not cancelable) map onto the server's
`_get_task`/`_cancel_task`.

## Task lifecycle (E-66)

```
submitted -> working -> input-required (spec agreed, awaiting build)
  -> working (build turn) -> input-required (build ready, awaiting review)
  -> working (review turn) -> completed | rejected
```

The Designer signs the spec with `spec_signed` (never by the critic; never by
the coder's service). The Coder builds **once** (E-68) and replies with a
`build` message (`state=input-required`). The Critic reviews by fetching the
binary over R20 (sha256 verified) and plays a scripted tape; its terminal
actions are `accept` (completed) or `walk_away` (rejected). A failed build
after `COMPILE_RETRIES` yields `walk_away` with `reason=build_failed`.

Because the server remembers `spec_signed`, a re-signed spec is refused with
`ALREADY_SIGNED` and no second build turn can start. The Coder never accepts:
when incoming terms clear its own acceptance threshold it counters with the
same terms so the Designer can sign off (E-66).
