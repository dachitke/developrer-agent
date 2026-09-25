"""Tests that the chat model can be built without calling the API."""

from app.agent.llm import create_llm
from app.config import PROJECT_ROOT, Settings


def test_create_llm_uses_settings_and_does_not_call_the_api() -> None:
    settings = Settings(
        openai_api_key="sk-test",
        model_name="gpt-4o-mini",
        temperature=0,
        data_dir=PROJECT_ROOT / "data",
        log_level="INFO",
    )

    llm = create_llm(settings)

    assert llm.model_name == "gpt-4o-mini"
    assert llm.temperature == 0
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "sk-test"
