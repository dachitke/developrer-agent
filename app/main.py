"""Interactive CLI entry point.

Run with:  python -m app.main
"""

import logging

from langchain_core.messages import BaseMessage

from app.agent.agent import build_tools, run_agent
from app.agent.llm import create_llm
from app.config import ConfigError, load_settings, setup_logging

logger = logging.getLogger(__name__)

BANNER = """
=================================
   Developer Assistant Agent
=================================
Type your request or 'exit' to quit.
"""


def format_tools_used(tools_used: list[str]) -> str:
    """Show each tool once. A repeated tool is written as 'calculator X3'."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for name in tools_used:
        if name not in counts:
            order.append(name)
            counts[name] = 0
        counts[name] += 1
    parts = [name if counts[name] == 1 else f"{name} X{counts[name]}" for name in order]
    return ", ".join(parts)


def main() -> None:
    """Load settings, then accept requests until the user exits.

    A failed or cancelled request is removed from the conversation history,
    so a broken exchange cannot make later requests fail too.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc)
        return

    setup_logging(settings.log_level)
    llm = create_llm(settings)
    tools = build_tools(settings.data_dir)
    history: list[BaseMessage] = []

    print(BANNER)
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        start = len(history)
        try:
            response = run_agent(
                user_text,
                llm,
                tools,
                history=history,
                on_progress=lambda line: print(f"  {line}"),
            )
        except KeyboardInterrupt:
            del history[start:]
            print("\nAgent: Request cancelled.")
            continue
        except Exception:
            del history[start:]
            logger.exception("Unexpected agent failure")
            print("Agent: Something went wrong. Please try another request.")
            continue

        if not response.success:
            del history[start:]
        print(f"Agent: {response.answer}")
        if response.tools_used:
            print(f"Tools used: {format_tools_used(response.tools_used)}")


if __name__ == "__main__":
    main()
