"""OpenAI chat model setup.

The rest of the application calls create_llm() and does not know how the
provider client is constructed.
"""

from langchain_openai import ChatOpenAI

from app.config import Settings


def create_llm(settings: Settings) -> ChatOpenAI:
    """Build a chat model from validated settings. This does not call the API."""
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=settings.temperature,
    )
