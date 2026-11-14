"""A2A 1.0 wire models for the game dev agent (pinned in docs/A2A_PIN.md).

Wire shapes follow A2A 1.0: no ``kind`` discriminator, ``ROLE_*`` roles,
``TASK_STATE_*`` states. Internal names map in :mod:`src.config`. Terms are
strict types on the wire (E-7): ints are reject-on-float/bool, enums are
members of the frozen categorical tables (E-72); never coerced.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, field_validator, model_validator

from src import config

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class Terms(BaseModel):
    """The negotiable game spec. Strict ints and frozen-table enums only."""

    model_config = ConfigDict(extra="ignore")

    genre: StrictStr
    grid_w: StrictInt
    grid_h: StrictInt
    tick_ms: StrictInt
    enemy_count: StrictInt
    win_condition: StrictStr
    reliability: StrictStr = "deterministic"

    @field_validator("genre")
    @classmethod
    def _genre_known(cls, value: str) -> str:
        if value not in config.GENRE_UTILITY:
            raise ValueError(f"unknown genre {value!r}")
        return value

    @field_validator("win_condition")
    @classmethod
    def _win_condition_known(cls, value: str) -> str:
        if value not in config.WIN_CONDITION_UTILITY:
            raise ValueError(f"unknown win_condition {value!r}")
        return value

    @field_validator("reliability")
    @classmethod
    def _reliability_known(cls, value: str) -> str:
        if value not in config.RELIABILITY_UTILITY:
            raise ValueError(f"unknown reliability {value!r}")
        return value

    @field_validator("grid_w", "grid_h", "tick_ms", "enemy_count")
    @classmethod
    def _ints_positive(cls, value: int) -> int:
        if value < 0:
            raise ValueError("spec ints must be >= 0")
        return value

    def as_dict(self) -> dict[str, Any]:
        return {
            "genre": self.genre,
            "grid_w": int(self.grid_w),
            "grid_h": int(self.grid_h),
            "tick_ms": int(self.tick_ms),
            "enemy_count": int(self.enemy_count),
            "win_condition": self.win_condition,
            "reliability": self.reliability,
        }


OfferType = Literal["offer", "counter", "spec_signed", "build", "accept", "walk_away"]

# E-67: the negotiation itself; each of these commits +1 to tasks.round.
SPEC_MESSAGE_TYPES = frozenset({"offer", "counter", "spec_signed", "walk_away"})
# Types a client may send at all; everything else (e.g. build) is rejected.
CLIENT_MESSAGE_TYPES = frozenset({"offer", "counter", "spec_signed", "accept", "walk_away"})


class OfferData(BaseModel):
    """The structured payload carried in a single ``data`` part.

    type            round    terms
    offer/counter   x        required
    spec_signed     x        required (the agreed spec is echoed)
    walk_away       x        optional (reason may carry build_failed etc.)
    accept          x        forbidden (critic accept carries build, not terms)
    build           x        forbidden (build messages carry artifact data)
    """

    model_config = ConfigDict(extra="ignore")

    type: OfferType
    round: StrictInt
    terms: Terms | None = None
    build: dict[str, Any] | None = None
    reason: str | None = None

    @field_validator("round")
    @classmethod
    def _round_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("round must be >= 0")
        return value

    @model_validator(mode="after")
    def _terms_and_build_shape(self) -> OfferData:
        if self.type in ("offer", "counter", "spec_signed") and self.terms is None:
            raise ValueError(f"{self.type} requires terms")
        if self.type in ("accept", "build") and self.terms is not None:
            raise ValueError(f"{self.type} must not carry terms")
        if self.type == "build" and not isinstance(self.build, dict):
            raise ValueError("build requires the artifact descriptor in 'build'")
        return self

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "round": int(self.round)}
        if self.terms is not None:
            out["terms"] = self.terms.as_dict()
        if self.build is not None:
            out["build"] = dict(self.build)
        if self.reason is not None:
            out["reason"] = self.reason
        return out


class Part(BaseModel):
    """A2A 1.0 Part: exactly one of text/raw/url/data."""

    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    raw: str | None = None
    url: str | None = None
    data: Any | None = None
    metadata: dict[str, Any] | None = None
    filename: str | None = None
    mediaType: str | None = None

    @model_validator(mode="after")
    def _exactly_one_content(self) -> Part:
        filled = sum(
            value is not None for value in (self.text, self.raw, self.url, self.data)
        )
        if filled != 1:
            raise ValueError("a Part MUST contain exactly one of text/raw/url/data")
        return self


class Message(BaseModel):
    """A2A 1.0 Message."""

    model_config = ConfigDict(extra="ignore")

    messageId: str
    role: str
    parts: list[Part]
    taskId: str | None = None
    contextId: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("role")
    @classmethod
    def _known_role(cls, value: str) -> str:
        if value not in {"ROLE_USER", "ROLE_AGENT", "user", "agent"}:
            raise ValueError("unknown role")
        return value

    @field_validator("messageId")
    @classmethod
    def _message_id_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("messageId must be non-empty")
        return value

    @property
    def internal_role(self) -> str:
        return "user" if self.role in {"ROLE_USER", "user"} else "agent"

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class TaskStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: str
    message: Message | None = None
    timestamp: str | None = None


class Artifact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifactId: str
    name: str | None = None
    description: str | None = None
    parts: list[Part]
    metadata: dict[str, Any] | None = None


class Task(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    contextId: str | None = None
    status: TaskStatus
    history: list[Message] | None = None
    artifacts: list[Artifact] | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Agent discovery
# ---------------------------------------------------------------------------


class AgentCapabilities(BaseModel):
    model_config = ConfigDict(extra="ignore")

    streaming: bool = False
    pushNotifications: bool = False


class AgentSkill(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    description: str
    tags: list[str]


class AgentInterface(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str
    protocolBinding: str
    protocolVersion: str


class AgentCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    supportedInterfaces: list[AgentInterface]
    version: str
    capabilities: AgentCapabilities
    defaultInputModes: list[str]
    defaultOutputModes: list[str]
    skills: list[AgentSkill]
    securitySchemes: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 envelopes
# ---------------------------------------------------------------------------


class JSONRPCRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    jsonrpc: Literal["2.0"]
    id: str | int | None = None
    method: str
    params: dict[str, Any] | None = None


class JSONRPCErrorObject(BaseModel):
    code: int
    message: str
    data: list[Any] | None = None


def jsonrpc_result(request_id: str | int | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def jsonrpc_error(
    request_id: str | int | None,
    code: int,
    message: str,
    data: list[Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def error_data(kind: str, detail: str | None = None) -> list[dict[str, Any]]:
    """Minimal ProtoJSON-Any-ish detail object used in error.data."""

    item: dict[str, Any] = {"@type": f"type.googleapis.com/a2a.gamedev.{kind}"}
    if detail is not None:
        item["detail"] = detail
    return [item]


# ---------------------------------------------------------------------------
# Part helpers
# ---------------------------------------------------------------------------


def text_part(text: str) -> dict[str, Any]:
    return {"text": text}


def data_part(data: dict[str, Any]) -> dict[str, Any]:
    return {"data": data}


def split_parts(parts: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    """Return (data payload, text) from raw wire parts."""

    data: dict[str, Any] | None = None
    text: str | None = None
    for part in parts:
        if "data" in part and part["data"] is not None:
            data = part["data"]
        elif "text" in part and part["text"] is not None:
            text = part["text"]
    return data, text


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Re-exported for convenience in server code.
MAX_MESSAGE_BYTES = config.MAX_MESSAGE_BYTES
