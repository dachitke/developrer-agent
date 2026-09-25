"""Tests for the agent's structured response model."""

import pytest
from pydantic import ValidationError

from app.models.schemas import AgentResponse


def test_successful_response_with_tools() -> None:
    response = AgentResponse(
        answer="  25  ",
        tools_used=[" calculator ", "read_file"],
        tool_calls=2,
    )

    assert response.answer == "25"
    assert response.tools_used == ["calculator", "read_file"]
    assert response.tool_calls == 2
    assert response.success is True
    assert response.error is None


def test_successful_response_can_use_no_tools() -> None:
    response = AgentResponse(answer="LangChain is a framework for LLM apps.")

    assert response.tools_used == []
    assert response.tool_calls == 0
    assert response.success is True


def test_failed_response_requires_an_error() -> None:
    response = AgentResponse(
        answer="I couldn't read that file because it does not exist.",
        tools_used=["read_file"],
        tool_calls=1,
        success=False,
        error="  file does not exist: missing.txt  ",
    )

    assert response.success is False
    assert response.error == "file does not exist: missing.txt"


def test_response_can_be_read_as_a_dictionary() -> None:
    response = AgentResponse(answer="100", tools_used=["calculator"], tool_calls=1)

    assert response.model_dump() == {
        "answer": "100",
        "tools_used": ["calculator"],
        "tool_calls": 1,
        "success": True,
        "error": None,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"answer": "   "},
        {"answer": "100", "tool_calls": -1},
        {"answer": "100", "tools_used": ["calculator", ""]},
        {"answer": "100", "error": "should not be set"},
        {"answer": "something went wrong", "success": False},
        {"answer": "something went wrong", "success": False, "error": "   "},
    ],
)
def test_invalid_response_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AgentResponse.model_validate(payload)


def test_response_cannot_be_changed_after_creation() -> None:
    response = AgentResponse(answer="100")

    with pytest.raises(ValidationError):
        response.answer = "200"
