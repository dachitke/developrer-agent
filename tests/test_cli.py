"""Tests for the interactive CLI. The language model is not called."""

from app.config import PROJECT_ROOT, ConfigError, Settings
from app.main import main
from app.models.schemas import AgentResponse


def _settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        model_name="gpt-4o-mini",
        temperature=0,
        data_dir=PROJECT_ROOT / "data",
        log_level="INFO",
    )


def test_cli_prints_the_answer_and_exits(capsys, monkeypatch) -> None:
    replies = iter(["Calculate 125 / 5", "exit"])
    monkeypatch.setattr("app.main.load_settings", _settings)
    monkeypatch.setattr("app.main.create_llm", lambda settings: object())
    monkeypatch.setattr("app.main.setup_logging", lambda level: None)
    monkeypatch.setattr(
        "app.main.run_agent",
        lambda *args, **kwargs: AgentResponse(answer="25", tools_used=["calculator"], tool_calls=1),
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    main()
    output = capsys.readouterr().out

    assert "Developer Assistant Agent" in output
    assert "Agent: 25" in output
    assert "Tools used: calculator" in output
    assert "Goodbye!" in output


def test_cli_survives_a_failed_request(capsys, monkeypatch) -> None:
    replies = iter(["Read missing.txt", "quit"])

    def fail_once(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.main.load_settings", _settings)
    monkeypatch.setattr("app.main.create_llm", lambda settings: object())
    monkeypatch.setattr("app.main.setup_logging", lambda level: None)
    monkeypatch.setattr("app.main.run_agent", fail_once)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    main()
    output = capsys.readouterr().out

    assert "Something went wrong" in output
    assert "Goodbye!" in output
    assert "Traceback" not in output


def test_cli_reports_a_missing_api_key(capsys, monkeypatch) -> None:
    def missing_key() -> Settings:
        raise ConfigError("OPENAI_API_KEY is not set.")

    monkeypatch.setattr("app.main.load_settings", missing_key)

    main()

    assert "OPENAI_API_KEY is not set" in capsys.readouterr().out
