"""OpenAI chat model setup.

The rest of the application calls create_llm() and does not know how the
provider client is constructed.
"""

from langchain_openai import ChatOpenAI

from app.config import Settings

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2


def create_llm(settings: Settings) -> ChatOpenAI:
    """Build a chat model from validated settings. This does not call the API.

    A timeout is set so a lost connection ends with an error message instead
    of leaving the CLI waiting forever.
    """
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=settings.temperature,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
