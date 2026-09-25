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
from app.tools.skills import DEFAULT_SKILLS_DIR, create_skill_reader, skill_catalog

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6

SYSTEM_PROMPT = """You are a developer assistant.
Decide for yourself whether a tool is needed.
If a skill below matches the request, call read_skill for that skill and follow its instructions.
Use the calculator tool for arithmetic instead of calculating it yourself.
Use the read_file tool for .txt, .md, and .json files in the data directory.
Known files include example.txt, notes.md, and numbers.json.
Do not open a data file to search for numbers the user already gave you.
If no tool is needed, answer directly.
After a tool returns, reply to the user in a short sentence.
"""

ProgressCallback = Callable[[str], None]


def build_system_prompt(skills_dir: Path = DEFAULT_SKILLS_DIR) -> str:
    """Base instructions plus the skill names the model can choose from."""
    return f"{SYSTEM_PROMPT}\nAvailable skills:\n{skill_catalog(skills_dir)}\n"


def build_tools(data_dir: Path, skills_dir: Path = DEFAULT_SKILLS_DIR) -> list[BaseTool]:
    """Return the tools the model is allowed to call."""
    return [calculator, create_file_reader(data_dir), create_skill_reader(skills_dir)]


def run_agent(
    user_message: str,
    llm: BaseChatModel,
    tools: list[BaseTool],
    history: list[BaseMessage] | None = None,
    skills_dir: Path = DEFAULT_SKILLS_DIR,
    max_iterations: int = MAX_ITERATIONS,
    on_progress: ProgressCallback | None = None,
) -> AgentResponse:
    """Run the tool-calling loop and return the application's response.

    `history` is updated in place so the CLI can continue the same conversation.
    """
    messages = history if history is not None else []
    if not any(isinstance(message, SystemMessage) for message in messages):
        messages.insert(0, SystemMessage(content=build_system_prompt(skills_dir)))
    messages.append(HumanMessage(content=user_message))

    model = llm.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}
    tools_used: list[str] = []

    for _ in range(max_iterations):
        try:
            ai_message = model.invoke(messages)
        except Exception:
            logger.exception("Language model request failed")
            return AgentResponse(
                answer=(
                    "I could not reach the language model. "
                    "Check your connection and API key, then try again."
                ),
                tools_used=tools_used,
                tool_calls=len(tools_used),
                success=False,
                error="language model request failed",
            )
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
            name, arguments, call_id = _read_tool_call(call)
            _report(on_progress, _progress_line(name, arguments))
            observation = _run_tool(tools_by_name.get(name), arguments)
            tools_used.append(name)
            logger.info("Tool %s returned: %s", name, observation)
            messages.append(ToolMessage(content=observation, tool_call_id=call_id))

    logger.warning("Stopped after %s tool-calling steps", max_iterations)
    return AgentResponse(
        answer="I stopped because this request needed too many steps. Please try a simpler question.",
        tools_used=tools_used,
        tool_calls=len(tools_used),
        success=False,
        error=f"stopped after {max_iterations} steps",
    )


def _read_tool_call(call: object) -> tuple[str, object, str]:
    if not isinstance(call, dict) or not call.get("name"):
        logger.error("Malformed tool call: %r", call)
        return "unknown", {}, "malformed"
    return str(call["name"]), call.get("args") or {}, str(call.get("id") or "missing")


def _run_tool(tool: BaseTool | None, arguments: object) -> str:
    if tool is None:
        return "Error: that tool is not available."
    if not isinstance(arguments, dict):
        return "Error: the tool arguments were not valid."
    try:
        return str(tool.invoke(arguments))
    except Exception:
        logger.exception("Tool %s failed", tool.name)
        return f"Error: the {tool.name} tool could not complete that request."


def _progress_line(name: str, arguments: object) -> str:
    details = arguments if isinstance(arguments, dict) else {}
    if name == "read_file":
        filename = details.get("filename", "a file")
        return f"Reading {filename}..."
    if name == "calculator":
        return "Using calculator..."
    if name == "read_skill":
        skill_name = details.get("skill_name", "a skill")
        return f"Using skill {skill_name}..."
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
