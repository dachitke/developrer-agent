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


def main() -> None:
    """Load settings, then accept requests until the user exits."""
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

        try:
            response = run_agent(
                user_text,
                llm,
                tools,
                history=history,
                on_progress=lambda line: print(f"  {line}"),
            )
        except Exception:
            logger.exception("Unexpected agent failure")
            print("Agent: Something went wrong. Please try another request.")
            continue

        print(f"Agent: {response.answer}")


if __name__ == "__main__":
    main()
