"""Tests for the interactive CLI. The language model is not called."""

from langchain_core.messages import HumanMessage

from app.config import PROJECT_ROOT, ConfigError, Settings
from app.main import format_tools_used, main
from app.models.schemas import AgentResponse


def _settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        model_name="gpt-4o-mini",
        temperature=0,
        data_dir=PROJECT_ROOT / "data",
        log_level="INFO",
    )


def test_repeated_tools_are_shown_once_with_a_count() -> None:
    assert format_tools_used(["read_skill", "calculator", "calculator", "calculator"]) == (
        "read_skill, calculator X3"
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


def test_ctrl_c_during_a_request_cancels_it_and_keeps_the_cli_open(capsys, monkeypatch) -> None:
    replies = iter(["What is 2 + 2?", "exit"])

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("app.main.load_settings", _settings)
    monkeypatch.setattr("app.main.create_llm", lambda settings: object())
    monkeypatch.setattr("app.main.setup_logging", lambda level: None)
    monkeypatch.setattr("app.main.run_agent", interrupted)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    main()
    output = capsys.readouterr().out

    assert "Request cancelled." in output
    assert "Goodbye!" in output
    assert "Traceback" not in output


def test_failed_request_is_removed_from_history(capsys, monkeypatch) -> None:
    replies = iter(["first", "second", "exit"])
    history_lengths: list[int] = []

    def fake_agent(user_text, llm, tools, history, on_progress):
        history_lengths.append(len(history))
        history.append(HumanMessage(content=user_text))
        if user_text == "first":
            return AgentResponse(answer="API down", success=False, error="language model request failed")
        return AgentResponse(answer="ok")

    monkeypatch.setattr("app.main.load_settings", _settings)
    monkeypatch.setattr("app.main.create_llm", lambda settings: object())
    monkeypatch.setattr("app.main.setup_logging", lambda level: None)
    monkeypatch.setattr("app.main.run_agent", fake_agent)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    main()

    assert history_lengths == [0, 0]


def test_cli_reports_a_missing_api_key(capsys, monkeypatch) -> None:
    def missing_key() -> Settings:
        raise ConfigError("OPENAI_API_KEY is not set.")

    monkeypatch.setattr("app.main.load_settings", missing_key)

    main()

    assert "OPENAI_API_KEY is not set" in capsys.readouterr().out
