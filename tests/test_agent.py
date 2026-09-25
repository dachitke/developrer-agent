"""Tests for the tool-calling loop with a fake chat model."""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from app.agent.agent import _read_tool_call, build_system_prompt, build_tools, run_agent
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


def _scripted(*responses: AIMessage) -> ScriptedToolModel:
    return ScriptedToolModel(responses=list(responses))


def _call(name: str, args: object, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def test_calculator_error_is_explained_instead_of_crashing() -> None:
    llm = _scripted(
        _call("calculator", {"expression": "10 / 0"}),
        AIMessage(content="I couldn't calculate that because it divides by zero."),
    )

    response = run_agent("What is 10 / 0?", llm, build_tools(PROJECT_ROOT / "data"))

    assert response.success is True
    assert "zero" in response.answer


def test_missing_file_is_explained_instead_of_crashing() -> None:
    llm = _scripted(
        _call("read_file", {"filename": "missing.txt"}),
        AIMessage(content="I couldn't read that file because it does not exist."),
    )

    response = run_agent("Read missing.txt", llm, build_tools(PROJECT_ROOT / "data"))

    assert response.success is True
    assert "does not exist" in response.answer


def test_invalid_tool_arguments_do_not_crash() -> None:
    llm = _scripted(
        _call("calculator", {"expression": 10}),
        AIMessage(content="I couldn't run the calculator because the expression was not text."),
    )

    response = run_agent("Calculate", llm, build_tools(PROJECT_ROOT / "data"))

    assert response.success is True
    assert "expression" in response.answer


def test_unknown_tool_name_does_not_crash() -> None:
    llm = _scripted(
        _call("web_search", {"query": "langchain"}),
        AIMessage(content="I don't have a web search tool."),
    )

    response = run_agent("Search the web", llm, build_tools(PROJECT_ROOT / "data"))

    assert response.success is True
    assert response.tools_used == ["web_search"]
    assert "web search" in response.answer


def test_unexpected_tool_exception_does_not_crash() -> None:
    from langchain_core.tools import tool

    @tool
    def explode(value: str) -> str:
        """A tool that always fails."""
        raise RuntimeError("disk on fire")

    llm = _scripted(
        _call("explode", {"value": "x"}),
        AIMessage(content="That tool failed, so I could not finish the request."),
    )

    response = run_agent("Use the broken tool", llm, [explode])

    assert response.success is True
    assert "failed" in response.answer


def test_model_failure_becomes_a_friendly_response() -> None:
    class BrokenModel(ScriptedToolModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("connection reset")

    response = run_agent("Hello", BrokenModel(responses=[]), build_tools(PROJECT_ROOT / "data"))

    assert response.success is False
    assert response.error == "language model request failed"
    assert "API key" in response.answer


def test_system_prompt_lists_installed_skills() -> None:
    prompt = build_system_prompt(PROJECT_ROOT / "skills")

    assert "quadratic-equation:" in prompt
    assert "read_skill" in prompt


def test_agent_reads_a_file_then_uses_the_calculator() -> None:
    llm = _scripted(
        _call("read_file", {"filename": "numbers.json"}, "call_1"),
        _call("calculator", {"expression": "10 + 20 + 30 + 40 + 50"}, "call_2"),
        AIMessage(content="The values add up to 150."),
    )

    response = run_agent("Add the values in numbers.json", llm, build_tools(PROJECT_ROOT / "data"))

    assert response.tools_used == ["read_file", "calculator"]
    assert response.tool_calls == 2
    assert "150" in response.answer


def test_agent_reads_a_skill_before_calculating() -> None:
    llm = _scripted(
        _call("read_skill", {"skill_name": "quadratic-equation"}, "call_1"),
        _call("calculator", {"expression": "20 ** 2 - 4 * 1 * 30"}, "call_2"),
        AIMessage(content="The discriminant is 280."),
    )
    progress: list[str] = []

    response = run_agent(
        "What are the roots when a=1, b=20 and c=30?",
        llm,
        build_tools(PROJECT_ROOT / "data"),
        on_progress=progress.append,
    )

    assert response.tools_used == ["read_skill", "calculator"]
    assert progress[0] == "Using skill quadratic-equation..."
    assert "280" in response.answer


def test_iteration_limit_stops_repeated_tool_calls() -> None:
    llm = _scripted(_call("read_file", {"filename": "example.txt"}))

    response = run_agent(
        "Keep reading",
        llm,
        build_tools(PROJECT_ROOT / "data"),
        max_iterations=2,
    )

    assert response.success is False
    assert response.error == "stopped after 2 steps"
    assert "too many steps" in response.answer


def test_malformed_tool_call_is_turned_into_an_error_observation() -> None:
    name, arguments, call_id = _read_tool_call({"args": {}, "id": "bad"})

    assert name == "unknown"
    assert arguments == {}
    assert call_id == "malformed"
