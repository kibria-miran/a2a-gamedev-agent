"""FastAPI Coder: agent card + one JSON-RPC route + R20 artifact fetch routes.

Rows R0-R4 run before any lock. Rows R4b-R14 run inside the per-task lock
(or the creation lock for taskless requests), with one transaction per state
change. The turn is a background asyncio task; the HTTP request waits for it
up to BLOCKING_TIMEOUT and then returns the committed snapshot.

v1.1 rows (E-66/E-70): R17-R20. Role identity comes from role-scoped bearer
tokens (E-38/E-66): the Designer and the Critic talk to the same service; the
token is the only way to tell them apart. Only the card GET is public.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from src import coder_agent, config, decision, protocol, state_machine, trace
from src.store import Store
from src.utility import Profile


class RpcFault(Exception):
    def __init__(self, code: int, message: str, *, kind: str | None = None, severity: str = "error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.kind = kind
        self.severity = severity


@dataclass(frozen=True)
class StartedTurn:
    task_id: str
    turn: asyncio.Task[None]


def _fault(code: int, message: str, *, kind: str | None = None, severity: str = "error") -> RpcFault:
    return RpcFault(code, message, kind=kind, severity=severity)


# which role may send which spec-message type (critic cannot offer/counter/sign)
_ROLE_TYPES = {
    "designer": {"offer", "counter", "spec_signed", "walk_away"},
    "critic": {"accept", "walk_away"},
}


def create_app(
    store: Store,
    *,
    responder: Any | None = None,
    profile: Profile | None = None,
    codegen: coder_agent.Codegen | None = None,
    chat: Any | None = None,
    designer_token: str | None = None,
    critic_token: str | None = None,
    card_url: str | None = None,
    startup_recovery: bool = True,
    trace_agent: str = "coder",
    include_rationale: bool = False,
) -> FastAPI:
    # include_rationale is an experimental X3 arm: when True, agent counter
    # parts carry their (leak-validated) rationale on the wire. The default
    # keeps the pinned part shape unchanged.
    designer_token = designer_token if designer_token is not None else config.A2A_TOKEN_DESIGNER
    critic_token = critic_token if critic_token is not None else config.A2A_TOKEN_CRITIC

    app = FastAPI(title="A2A Game Dev Coder")
    app.state.store = store
    app.state.designer_token = designer_token
    app.state.critic_token = critic_token
    app.state.chat = chat
    app.state.include_rationale = include_rationale
    app.state.card_url = card_url or config.CODER_URL
    app.state.turns = {}
    app.state.turn_tasks = {}
    app.state.trace_agent = trace_agent

    if responder is None:
        # The real Coder negotiator and one-build turn (Phase 4). It loads the
        # private profile lazily; tests and embedded demos can inject another
        # responder without touching this path.
        app.state.responder = coder_agent.make_responder(
            profile=profile, chat=chat, codegen=codegen
        )
    else:
        app.state.responder = responder

    if startup_recovery:
        store.startup_recovery()

    # -- auth ---------------------------------------------------------------

    def role_for(header: str) -> str | None:
        if app.state.designer_token and header == f"Bearer {app.state.designer_token}":
            return "designer"
        if app.state.critic_token and header == f"Bearer {app.state.critic_token}":
            return "critic"
        return None

    def authorized(request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        return role_for(header)

    # -- card ---------------------------------------------------------------

    def card() -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": "A2A Game Dev Coder",
            "description": (
                "Builds a tiny harness-contract C++ game from a negotiated spec; "
                "spec negotiation with the Designer, binary handoff to the Critic."
            ),
            "version": "0.1.0",
            "supportedInterfaces": [
                {
                    "url": app.state.card_url,
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": config.A2A_PROTOCOL_VERSION,
                }
            ],
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": ["application/json"],
            "defaultOutputModes": ["application/json"],
            "skills": [
                {
                    "id": "build_game",
                    "name": "Build the game",
                    "description": "Negotiate the spec, build it once, hand the binary to the Critic.",
                    "tags": ["cpp", "harness", "game", "build"],
                }
            ],
        }
        if app.state.designer_token or app.state.critic_token:
            body["securitySchemes"] = {
                "bearerAuth": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}
            }
        return body

    @app.get(config.AGENT_CARD_PATH)
    async def agent_card() -> dict[str, Any]:
        return card()  # E-38: only the card GET is public

    @app.get(config.AGENT_CARD_ALIAS_PATH)
    async def agent_card_alias() -> dict[str, Any]:
        return card()

    # -- R20 artifact fetch routes (E-70) -----------------------------------

    @app.get("/tasks/{task_id}/build/{artifact_id}")
    async def fetch_build(task_id: str, artifact_id: str, request: Request) -> Response:
        if not authorized(request):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        artifact = store.get_binary_artifact(task_id, artifact_id)
        if artifact is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": config.A2A_TASK_NOT_FOUND, "message": config.REASONS["R20"]}},
            )
        return Response(
            content=artifact["binary"],
            media_type="application/octet-stream",
            headers={
                "X-Build-Sha256": artifact["sha256"],
                "Content-Length": str(artifact["size"]),
            },
        )

    @app.get("/tasks/{task_id}/source/{artifact_id}")
    async def fetch_source(task_id: str, artifact_id: str, request: Request) -> Response:
        if not authorized(request):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        tar_bytes = store.get_source_tar_artifact(task_id, artifact_id)
        if tar_bytes is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": config.A2A_TASK_NOT_FOUND, "message": config.REASONS["R20"]}},
            )

        return Response(
            content=tar_bytes,
            media_type="application/x-tar",
            headers={
                "X-Build-Sha256": hashlib.sha256(tar_bytes).hexdigest(),
                "Content-Length": str(len(tar_bytes)),
            },
        )

    # -- JSON-RPC entry -----------------------------------------------------

    @app.post(config.A2A_RPC_PATH)
    async def rpc(request: Request) -> JSONResponse:
        role = authorized(request)
        if role is None:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})

        request_id: str | int | None = None
        raw = await request.body()

        # R0: body size.
        if len(raw) > config.MAX_MESSAGE_BYTES:
            return JSONResponse(
                protocol.jsonrpc_error(request_id, config.JSONRPC_INVALID_REQUEST, "request too large")
            )

        # R1: valid JSON.
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(
                protocol.jsonrpc_error(request_id, config.JSONRPC_PARSE_ERROR, "invalid JSON")
            )

        # R2: envelope and method.
        if not isinstance(payload, dict):
            return JSONResponse(
                protocol.jsonrpc_error(request_id, config.JSONRPC_INVALID_REQUEST, "invalid request")
            )
        request_id = payload.get("id")
        try:
            envelope = protocol.JSONRPCRequest.model_validate(payload)
        except ValidationError:
            return JSONResponse(
                protocol.jsonrpc_error(request_id, config.JSONRPC_INVALID_REQUEST, "invalid request")
            )
        if envelope.method not in config.A2A_METHODS:
            return JSONResponse(
                protocol.jsonrpc_error(request_id, config.JSONRPC_METHOD_NOT_FOUND, "method not found")
            )

        try:
            if envelope.method == config.A2A_METHOD_SEND:
                result = {"task": await _send_message(envelope.params, role)}
            elif envelope.method == config.A2A_METHOD_GET_TASK:
                result = _get_task(envelope.params)
            else:
                result = _cancel_task(envelope.params)
        except RpcFault as fault:
            return JSONResponse(protocol.jsonrpc_error(request_id, fault.code, fault.message))
        except Exception:
            return JSONResponse(
                protocol.jsonrpc_error(request_id, config.JSONRPC_INTERNAL_ERROR, "internal error")
            )

        return JSONResponse(protocol.jsonrpc_result(request_id, result))

    # -- message/send -------------------------------------------------------

    async def _send_message(params: dict[str, Any] | None, role: str) -> dict[str, Any]:
        try:
            message, offer, blocking = _validate_send_params(params, role)
        except RpcFault:
            trace.trace("offer", direction="in", agent=app.state.trace_agent,
                        severity="error", error_kind="bad_params")
            raise
        task_id = message.taskId

        if not task_id:
            stored = store.find_message(message.messageId)
            if stored is not None:
                async with store.lock_for(stored["task_id"]):
                    outcome = _handle_existing(stored["task_id"], message, offer, role)
                return await _finish(outcome, blocking)

            # taskless walk_away/accept cannot open a task (R12b)
            if offer["type"] in {"accept", "walk_away"}:
                raise _fault(config.JSONRPC_INVALID_PARAMS, f"{offer['type']} requires an existing task", kind="bad_params")

            # R4: fast pre-check, then the authoritative re-check inside the creation lock.
            if store.open_task_count() >= config.MAX_OPEN_TASKS:
                trace.trace("server_busy", direction="in", agent=app.state.trace_agent, severity="info")
                raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R4"], kind="server_busy", severity="info")
            async with store.creation_lock:
                if store.open_task_count() >= config.MAX_OPEN_TASKS:
                    trace.trace("server_busy", direction="in", agent=app.state.trace_agent, severity="info")
                    raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R4"], kind="server_busy", severity="info")
                task = store.create_task(message.contextId)
                outcome = _begin_turn(task["id"], message, offer, role)
            return await _finish(outcome, blocking)

        if store.get_task(task_id) is None:
            raise _fault(config.A2A_TASK_NOT_FOUND, "task not found")

        async with store.lock_for(task_id):
            outcome = _handle_existing(task_id, message, offer, role)
        return await _finish(outcome, blocking)

    async def _finish(
        outcome: dict[str, Any] | StartedTurn, blocking: bool
    ) -> dict[str, Any]:
        if isinstance(outcome, StartedTurn):
            # A2A SendMessageConfiguration.returnImmediately: hand back the
            # committed snapshot (usually TASK_STATE_WORKING) instead of
            # waiting for the background turn. The client then polls GetTask.
            if blocking:
                return await _await_turn(outcome)
            view = store.task_view(outcome.task_id)
            assert view is not None
            return view
        return outcome

    def _handle_existing(
        task_id: str, message: protocol.Message, offer: dict[str, Any], role: str
    ) -> dict[str, Any] | StartedTurn:
        stored = store.find_message(message.messageId)

        if stored is not None:
            # R6a: the message belongs to another task.
            if stored["task_id"] != task_id or stored["role"] != "user":
                raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R6A"], kind="bad_params")

            task = store.get_task(stored["task_id"])
            assert task is not None
            agent_after = any(
                row["role"] == "agent" and row["seq"] > stored["seq"]
                for row in store.messages_for_task(stored["task_id"])
            )

            # R6b: turn already completed -> current task view.
            if state_machine.is_terminal(task["state"]) or agent_after:
                trace.trace("replay", direction="in", agent=app.state.trace_agent,
                            task_id=stored["task_id"], round=int(offer["round"]))
                view = store.task_view(task["id"])
                assert view is not None
                return view

            # R6d: the turn is still running.
            if task["state"] == "working":
                raise _fault(
                    config.JSONRPC_INVALID_PARAMS, config.REASONS["R6D"], kind="bad_params"
                )

            # R6c: resume an incomplete turn.
            if task["round_at_turn_start"] is None:
                raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["RESUME_DRIFT"], kind="bad_params")
            stored_round = store.message_round(stored)
            if stored_round != task["round_at_turn_start"]:
                raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["RESUME_DRIFT"], kind="bad_params")
            store.resume_turn(task["id"])
            trace.trace("resume", direction="in", agent=app.state.trace_agent, task_id=task["id"])
            incoming, text = protocol.split_parts(stored["parts"])
            if incoming is None:
                raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["RESUME_DRIFT"], kind="bad_params")
            incoming = dict(incoming)
            incoming["_peer_text"] = text
            return _spawn_turn(task["id"], incoming)

        task = store.get_task(task_id)
        assert task is not None

        # R7: terminal tasks are immutable.
        if state_machine.is_terminal(task["state"]):
            raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R7"], kind="bad_params")

        # R8: new messageId while a turn is running.
        if task["state"] == "working":
            raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R8"], kind="bad_params")

        # R17-R19: role/type rows (v1.1, E-66).
        _decision_rows(task, offer, role)

        # R9: only spec messages carry a checked round (E-67).
        if (
            offer["type"] in protocol.SPEC_MESSAGE_TYPES
            and int(offer["round"]) != int(task["round"])
        ):
            raise _fault(
                config.JSONRPC_INVALID_PARAMS, config.REASONS["R9"], kind="bad_round"
            )

        # accept: critic completes the task (E-66; no terms, no agreement artifact).
        if offer["type"] == "accept":
            # R10: the accept must name the build that was actually produced,
            # at the task's current round. Terms don't exist here; the build
            # descriptor (artifactId + sha256, optional size) is the artifact
            # under acceptance, so a mismatched accept is a protocol fault.
            _check_accept_matches_build(task_id, offer, task)
            store.complete_with_accept(
                task_id, None,
                message_id=message.messageId,
                parts=[p.model_dump(exclude_none=True) for p in message.parts],
            )
            trace.trace("accept", direction="in", agent=app.state.trace_agent,
                        task_id=task_id, round=int(offer["round"]))
            view = store.task_view(task_id)
            assert view is not None
            return view

        # walk_away: designer mid-negotiation, or critic rejecting the build.
        if offer["type"] == "walk_away":
            store.commit_user_walk_away(
                task_id,
                message_id=message.messageId,
                parts=[p.model_dump(exclude_none=True) for p in message.parts],
            )
            trace.trace("walk_away", direction="in", agent=app.state.trace_agent,
                        task_id=task_id, round=int(offer["round"]))
            view = store.task_view(task_id)
            assert view is not None
            return view

        # R13/R14: a spec message on an input-required/submitted task.
        if task["state"] in {"input-required", "submitted"}:
            return _begin_turn(task_id, message, offer, role)

        raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R8"], kind="bad_params")

    def _check_accept_matches_build(
        task_id: str, offer: dict[str, Any], task: dict[str, Any]
    ) -> None:
        build = offer.get("build")
        last_agent = store.last_message(task_id, "agent")
        last_data, _ = (
            protocol.split_parts(last_agent["parts"]) if last_agent is not None else (None, None)
        )
        artifact = (
            last_data.get("artifact")
            if isinstance(last_data, dict) and last_data.get("type") == "build"
            else None
        )

        mismatch = not isinstance(build, dict) or not isinstance(artifact, dict)
        if isinstance(build, dict) and isinstance(artifact, dict):
            mismatch = (
                str(build.get("artifactId", "")) != str(artifact.get("artifactId", ""))
                or not artifact.get("sha256")
                or str(build.get("sha256", "")) != str(artifact.get("sha256"))
                or int(offer["round"]) != int(task["round"])
            )
            if not mismatch and "size" in build:
                mismatch = str(build.get("size")) != str(artifact.get("size"))

        if mismatch:
            trace.trace("accept", direction="in", agent=app.state.trace_agent,
                        task_id=task_id, round=int(offer["round"]),
                        severity="error", error_kind="bad_params")
            raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R10"], kind="bad_params")

    def _decision_rows(task: dict[str, Any], offer: dict[str, Any], role: str) -> None:
        type_ = offer["type"]
        # R17: spec_signed must come from the designer.
        if type_ == "spec_signed" and role != "designer":
            raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R17"], kind="bad_params")
        # R18: build never arrives unset pre-signoff (a client build is also
        # rejected this way — builds come from the service).
        if type_ == "build" and not bool(task["spec_signed"]):
            raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R18"], kind="bad_params")
        # E-68: one build per task. Re-signing an already-signed spec would
        # start a second build turn, so it is rejected at the message layer.
        if type_ == "spec_signed" and bool(task["spec_signed"]):
            raise _fault(
                config.JSONRPC_INVALID_PARAMS, config.REASONS["ALREADY_SIGNED"],
                kind="bad_params",
            )
        # R19: accept must be the critic AND after signoff.
        if type_ == "accept" and (role != "critic" or not bool(task["spec_signed"])):
            raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R19"], kind="bad_params")
        # role-type pairing for the remaining spec messages.
        if type_ in {"offer", "counter"} and role != "designer":
            raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R21"], kind="bad_params")

    def _begin_turn(
        task_id: str, message: protocol.Message, offer: dict[str, Any], role: str
    ) -> StartedTurn:
        parts = [p.model_dump(exclude_none=True) for p in message.parts]
        store.begin_turn(task_id, message.messageId, parts)
        trace.trace("offer" if offer["type"] == "offer" else offer["type"],
                    direction="in", agent=app.state.trace_agent,
                    task_id=task_id, round=int(offer["round"]))
        incoming = dict(offer)
        text_part = next((part.text for part in message.parts if part.text), None)
        if text_part:
            incoming["_peer_text"] = text_part
        return _spawn_turn(task_id, incoming)

    def _spawn_turn(task_id: str, incoming: dict[str, Any]) -> StartedTurn:
        view = store.task_view(task_id)
        assert view is not None
        own_history = store.own_utility_history(task_id)
        turn = asyncio.create_task(_run_turn(task_id, incoming, view, own_history))
        app.state.turns[turn] = None
        app.state.turn_tasks[task_id] = turn

        def _cleanup(done: asyncio.Task[None]) -> None:
            app.state.turns.pop(done, None)
            if app.state.turn_tasks.get(task_id) is done:
                app.state.turn_tasks.pop(task_id, None)

        turn.add_done_callback(_cleanup)
        return StartedTurn(task_id=task_id, turn=turn)

    async def _await_turn(started: StartedTurn) -> dict[str, Any]:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(started.turn), timeout=config.BLOCKING_TIMEOUT
            )
        current = store.task_view(started.task_id)
        assert current is not None
        return current

    async def _run_turn(
        task_id: str,
        incoming: dict[str, Any],
        view: dict[str, Any],
        own_history: list[float],
    ) -> None:
        try:
            result = await asyncio.to_thread(app.state.responder, view, incoming, own_history)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, decision.Decision):
                raise TypeError("responder must return a Decision")
            dec: decision.Decision = result

            def _commit(kind: str, state: str, sent: str, **kwargs: Any) -> None:
                committed = store.cas_commit(task_id, state, sent, **kwargs)
                if committed:
                    trace.trace(kind, direction="out", agent=app.state.trace_agent,
                                task_id=task_id, round=int(dec.round), own_u=dec.own_u,
                                source=dec.source,
                                tokens_in=dec.tokens_in, tokens_out=dec.tokens_out)
                else:
                    trace.trace("discarded", direction="out", agent=app.state.trace_agent,
                                task_id=task_id, round=int(dec.round))

            if dec.kind == "counter":
                counter_part = {
                    "type": "counter", "terms": dec.terms or {},
                    "round": int(dec.round),
                }
                if app.state.include_rationale and dec.rationale:
                    # X3 arm only; default keeps the pinned part shape.
                    counter_part["rationale"] = dec.rationale
                parts = [protocol.data_part(counter_part)]
                # NOTE: counter terms come from the responder, not the LLM here —
                # the negotiator decides; the LLM only proposes (E-68).
                _commit("counter", "input-required", "counter",
                        own_u=dec.own_u, agent_message_id=str(uuid.uuid4()), agent_parts=parts)
            elif dec.kind == "build":
                br = dec.build_result
                if br is None or not br.ok or not br.binary_bytes:
                    # E-68: build failed after retries -> walk_away(build_failed)
                    parts = [protocol.data_part({
                        "type": "walk_away", "round": int(dec.round), "reason": "build_failed",
                    })]
                    _commit("walk_away", "rejected", "walk_away",
                            agent_message_id=str(uuid.uuid4()), agent_parts=parts)
                    return
                descriptor = store.store_binary_artifact(
                    task_id, binary=br.binary_bytes, source_tar=br.source_tar,
                    name="binary",
                )
                parts = [protocol.data_part({
                    "type": "build",
                    "round": int(dec.round),
                    "artifact": descriptor,
                })]
                _commit("build", "input-required", "build",
                        agent_message_id=str(uuid.uuid4()), agent_parts=parts)
            elif dec.kind == "walk_away":
                parts = [protocol.data_part({
                    "type": "walk_away", "round": int(dec.round),
                    **({"reason": dec.reason} if dec.reason else {}),
                })]
                _commit("walk_away", "rejected", "walk_away",
                        agent_message_id=str(uuid.uuid4()), agent_parts=parts)
            else:
                raise ValueError(f"unknown decision kind {dec.kind!r}")
        except asyncio.CancelledError:
            raise
        except Exception:
            committed = store.cas_commit(task_id, "failed", "failed")
            if committed:
                trace.trace("turn_failed", direction="out", agent=app.state.trace_agent,
                            task_id=task_id, severity="error", error_kind="turn_failed")
            else:
                trace.trace("discarded", direction="out", agent=app.state.trace_agent,
                            task_id=task_id)

    # -- tasks/get and tasks/cancel ----------------------------------------

    def _get_task(params: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
        history_length = params.get("historyLength")
        if history_length is not None:
            if (
                not isinstance(history_length, int)
                or isinstance(history_length, bool)
                or history_length < 0
            ):
                raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
        else:
            history_length = config.MAX_HISTORY_RESPONSE

        view = store.task_view(task_id, history_length)
        if view is None:
            raise _fault(config.A2A_TASK_NOT_FOUND, "task not found")
        return view

    def _cancel_task(params: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
        task_id = params.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")

        _task, error = store.cancel_task(task_id)
        if error == config.A2A_TASK_NOT_FOUND:
            raise _fault(config.A2A_TASK_NOT_FOUND, "task not found")
        if error == config.A2A_TASK_NOT_CANCELABLE:
            raise _fault(config.A2A_TASK_NOT_CANCELABLE, "task not cancelable")

        # Cancel a running turn too: the responder thread may still finish,
        # but its CAS commit will be discarded against the canceled state.
        running = app.state.turn_tasks.get(task_id)
        if running is not None and not running.done():
            running.cancel()

        view = store.task_view(task_id)
        assert view is not None
        return view

    return app


def _validate_send_params(
    params: dict[str, Any] | None, role: str
) -> tuple[protocol.Message, dict[str, Any], bool]:
    """R3: schema and strict types, before any lock. Also enforces the role
    type pairing (E-66) up front so a zero-budget caller gets the stable fault."""

    if not isinstance(params, dict):
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
    raw_message = params.get("message")
    if not isinstance(raw_message, dict):
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")

    try:
        message = protocol.Message.model_validate(raw_message)
    except ValidationError:
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params") from None

    if message.internal_role != "user":
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")

    if any(part.raw is not None or part.url is not None for part in message.parts):
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")

    data_parts = [part for part in message.parts if part.data is not None]
    text_parts = [part for part in message.parts if part.text is not None]
    if len(data_parts) != 1 or len(text_parts) > 1:
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
    if text_parts and len(text_parts[0].text or "") > config.MAX_RATIONALE_CHARS:
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
    if not isinstance(data_parts[0].data, dict):
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")

    try:
        offer = protocol.OfferData.model_validate(data_parts[0].data)
    except ValidationError:
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params") from None

    if offer.type not in protocol.CLIENT_MESSAGE_TYPES:
        # build and friends never arrive from clients (E-18 class errors).
        raise _fault(config.JSONRPC_INVALID_PARAMS, config.REASONS["R18"], kind="bad_params")

    allowed = _ROLE_TYPES.get(role, set())
    if offer.type not in allowed:
        reason = config.REASONS["R19"] if offer.type == "accept" else (
            config.REASONS["R17"] if offer.type == "spec_signed" and role != "designer"
            else config.REASONS["R21"]
        )
        raise _fault(config.JSONRPC_INVALID_PARAMS, reason, kind="bad_params")

    configuration = params.get("configuration")
    if configuration is None:
        configuration = {}
    if not isinstance(configuration, dict):
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
    return_immediately = configuration.get("returnImmediately", False)
    legacy_blocking = configuration.get("blocking")
    if not isinstance(return_immediately, bool):
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
    if legacy_blocking is not None and not isinstance(legacy_blocking, bool):
        raise _fault(config.JSONRPC_INVALID_PARAMS, "invalid params", kind="bad_params")
    blocking = legacy_blocking if legacy_blocking is not None else not return_immediately

    return message, offer.as_dict(), blocking
