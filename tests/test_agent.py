"""Tests for the tool-calling loop with a fake chat model."""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from app.agent.agent import build_tools, run_agent
from app.config import PROJECT_ROOT


class ScriptedToolModel(FakeMessagesListChatModel):
    """Fake model that accepts bind_tools and still returns scripted messages.

    The real OpenAI model implements bind_tools. The stock fake model does not,
    so tests use this subclass. It does not choose tools itself; each test
    scripts the choice the real model would make.
    """

    def bind_tools(self, tools: list[BaseTool], **kwargs: object) -> "ScriptedToolModel":
        return self


def test_agent_answers_without_a_tool_when_the_model_does_not_call_one() -> None:
    llm = ScriptedToolModel(
        responses=[AIMessage(content="LangChain is a framework for building LLM apps.")]
    )

    response = run_agent("What is LangChain?", llm, build_tools(PROJECT_ROOT / "data"))

    assert response.success is True
    assert response.tools_used == []
    assert response.tool_calls == 0
    assert "LangChain" in response.answer


def test_agent_runs_the_tool_the_model_selects() -> None:
    llm = ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "125 / 5"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="125 / 5 is 25."),
        ]
    )
    progress: list[str] = []

    response = run_agent(
        "What is 125 / 5?",
        llm,
        build_tools(PROJECT_ROOT / "data"),
        on_progress=progress.append,
    )

    assert response.answer == "125 / 5 is 25."
    assert response.tools_used == ["calculator"]
    assert response.tool_calls == 1
    assert progress == ["Using calculator..."]


def test_agent_reads_a_file_when_the_model_selects_that_tool() -> None:
    llm = ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"filename": "numbers.json"},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The file lists the values 10, 20, 30, 40 and 50."),
        ]
    )
    progress: list[str] = []

    response = run_agent(
        "Read numbers.json",
        llm,
        build_tools(PROJECT_ROOT / "data"),
        on_progress=progress.append,
    )

    assert response.success is True
    assert response.tools_used == ["read_file"]
    assert progress == ["Reading numbers.json..."]
    assert "50" in response.answer
