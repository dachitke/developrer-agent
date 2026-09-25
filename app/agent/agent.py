"""The autonomous agent.

The model decides whether a tool is needed. This module never checks the
user's words for keywords such as "calculate" or "read".
"""

import logging
from collections.abc import Callable
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from app.models.schemas import AgentResponse
from app.tools.calculator import calculator
from app.tools.file_reader import create_file_reader

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6

SYSTEM_PROMPT = """You are a developer assistant.
Decide for yourself whether a tool is needed.
Use the calculator tool for arithmetic instead of calculating it yourself.
Use the read_file tool for .txt, .md, and .json files in the data directory.
Known files include example.txt, notes.md, and numbers.json.
If no tool is needed, answer directly.
After a tool returns, reply to the user in a short sentence.
"""

ProgressCallback = Callable[[str], None]


def build_tools(data_dir: Path) -> list[BaseTool]:
    """Return the tools the model is allowed to call."""
    return [calculator, create_file_reader(data_dir)]


def run_agent(
    user_message: str,
    llm: BaseChatModel,
    tools: list[BaseTool],
    history: list[BaseMessage] | None = None,
    max_iterations: int = MAX_ITERATIONS,
    on_progress: ProgressCallback | None = None,
) -> AgentResponse:
    """Run the tool-calling loop and return the application's response.

    `history` is updated in place so the CLI can continue the same conversation.
    """
    messages = history if history is not None else []
    if not any(isinstance(message, SystemMessage) for message in messages):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    messages.append(HumanMessage(content=user_message))

    model = llm.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}
    tools_used: list[str] = []

    for _ in range(max_iterations):
        ai_message = model.invoke(messages)
        messages.append(ai_message)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            answer = _message_text(ai_message) or "I could not produce an answer."
            return AgentResponse(
                answer=answer,
                tools_used=tools_used,
                tool_calls=len(tools_used),
            )

        for call in tool_calls:
            name = call["name"]
            arguments = call.get("args") or {}
            _report(on_progress, _progress_line(name, arguments))
            observation = _run_tool(tools_by_name.get(name), arguments)
            tools_used.append(name)
            logger.info("Tool %s returned: %s", name, observation)
            messages.append(
                ToolMessage(content=observation, tool_call_id=call["id"])
            )

    logger.warning("Stopped after %s tool-calling steps", max_iterations)
    return AgentResponse(
        answer="I stopped because this request needed too many steps. Please try a simpler question.",
        tools_used=tools_used,
        tool_calls=len(tools_used),
        success=False,
        error=f"stopped after {max_iterations} steps",
    )


def _run_tool(tool: BaseTool | None, arguments: dict) -> str:
    if tool is None:
        return "Error: that tool is not available."
    result = tool.invoke(arguments)
    return str(result)


def _progress_line(name: str, arguments: dict) -> str:
    if name == "read_file":
        filename = arguments.get("filename", "a file")
        return f"Reading {filename}..."
    if name == "calculator":
        return "Using calculator..."
    return f"Using {name}..."


def _report(on_progress: ProgressCallback | None, line: str) -> None:
    if on_progress is not None:
        on_progress(line)


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()
