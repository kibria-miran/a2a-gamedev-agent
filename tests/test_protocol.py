"""Protocol model tests (E-7 strict types, exact-content parts)."""

import pytest
from pydantic import ValidationError

from src import config, protocol

TERMS_OK = {
    "genre": "invaders",
    "grid_w": 40,
    "grid_h": 20,
    "tick_ms": 100,
    "enemy_count": 8,
    "win_condition": "survive",
}


def test_terms_reference_brief_validates():
    terms = protocol.Terms.model_validate(TERMS_OK)
    assert terms.reliability == "deterministic"
    assert terms.as_dict()["enemy_count"] == 8


def test_terms_reject_unknown_genre():
    with pytest.raises(ValidationError):
        protocol.Terms.model_validate({**TERMS_OK, "genre": "doom-clone"})


def test_terms_reject_float_and_bool_ints():
    with pytest.raises(ValidationError):
        protocol.Terms.model_validate({**TERMS_OK, "grid_w": 40.0})
    with pytest.raises(ValidationError):
        protocol.Terms.model_validate({**TERMS_OK, "grid_w": True})


def test_terms_reject_bad_win_condition_and_reliability():
    with pytest.raises(ValidationError):
        protocol.Terms.model_validate({**TERMS_OK, "win_condition": "victory"})
    with pytest.raises(ValidationError):
        protocol.Terms.model_validate({**TERMS_OK, "reliability": "mostly"})


def offer_data(type_, round_=0, **kw):
    base = {"type": type_, "round": round_, **kw}
    return base


def test_offer_round_trip():
    offer = protocol.OfferData.model_validate(
        offer_data("counter", 3, terms=TERMS_OK)
    )
    assert offer.as_dict()["terms"]["genre"] == "invaders"


def test_spec_signed_carries_terms():
    data = protocol.OfferData.model_validate(
        offer_data("spec_signed", 2, terms=TERMS_OK)
    )
    assert data.terms is not None


def test_offer_requires_terms():
    with pytest.raises(ValidationError):
        protocol.OfferData.model_validate(offer_data("offer", 0))


def test_accept_must_not_carry_terms():
    with pytest.raises(ValidationError):
        protocol.OfferData.model_validate(offer_data("accept", 0, terms=TERMS_OK))


def test_build_requires_descriptor():
    with pytest.raises(ValidationError):
        protocol.OfferData.model_validate(offer_data("build", 3))
    data = protocol.OfferData.model_validate(
        offer_data("build", 3, build={"artifact": {"artifactId": "a"}})
    )
    assert data.type == "build"


def test_message_role_mapping():
    message = protocol.Message.model_validate(
        {"messageId": "m", "role": "ROLE_USER", "parts": [{"text": "hi"}]}
    )
    assert message.internal_role == "user"
    agent = protocol.Message.model_validate(
        {"messageId": "m", "role": "ROLE_AGENT", "parts": [{"text": "hi"}]}
    )
    assert agent.internal_role == "agent"


def test_part_must_have_exactly_one_content_type():
    with pytest.raises(ValidationError):
        protocol.Part.model_validate({"text": "x", "data": {"a": 1}})
    with pytest.raises(ValidationError):
        protocol.Part.model_validate({})


def test_build_sha256_helper():
    import hashlib

    assert protocol.build_sha256(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_jsonrpc_helpers():
    result = protocol.jsonrpc_result("1", {"task": {}})
    assert result == {"jsonrpc": "2.0", "id": "1", "result": {"task": {}}}
    error = protocol.jsonrpc_error("1", config.JSONRPC_INVALID_PARAMS, "oops")
    assert error["error"]["code"] == -32602
    assert protocol.error_data("test")[0]["@type"].startswith("type.googleapis.com/")
