"""Tests for configuration loading and validation."""

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, ConfigError, load_settings

CONFIG_VARS = ("OPENAI_API_KEY", "MODEL_NAME", "TEMPERATURE", "DATA_DIR", "LOG_LEVEL")


@pytest.fixture(autouse=True)
def isolated_env() -> Iterator[None]:
    """Start each test without config variables; restore the real environment after."""
    with patch.dict(os.environ):
        for name in CONFIG_VARS:
            os.environ.pop(name, None)
        yield


def test_missing_api_key_raises_clear_error() -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY is not set"):
        load_settings(env_file=None)


@pytest.mark.parametrize("value", ["", "   ", "your_api_key_here"])
def test_blank_or_placeholder_api_key_is_rejected(value: str) -> None:
    os.environ["OPENAI_API_KEY"] = value
    with pytest.raises(ConfigError, match="OPENAI_API_KEY is not set"):
        load_settings(env_file=None)


def test_defaults_are_used_when_only_api_key_is_set() -> None:
    os.environ["OPENAI_API_KEY"] = "sk-test"

    settings = load_settings(env_file=None)

    assert settings.openai_api_key.get_secret_value() == "sk-test"
    assert settings.model_name == "gpt-4o-mini"
    assert settings.temperature == 0.0
    assert settings.data_dir == (PROJECT_ROOT / "data").resolve()
    assert settings.log_level == "INFO"


def test_custom_values_are_read_from_environment(tmp_path: Path) -> None:
    os.environ.update(
        OPENAI_API_KEY="sk-test",
        MODEL_NAME="gpt-4.1-mini",
        TEMPERATURE="0.7",
        DATA_DIR=str(tmp_path),
        LOG_LEVEL="debug",
    )

    settings = load_settings(env_file=None)

    assert settings.model_name == "gpt-4.1-mini"
    assert settings.temperature == 0.7
    assert settings.data_dir == tmp_path.resolve()
    assert settings.log_level == "DEBUG"


def test_relative_data_dir_is_resolved_from_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    os.environ.update(OPENAI_API_KEY="sk-test", DATA_DIR="data")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(env_file=None)

    assert settings.data_dir == (PROJECT_ROOT / "data").resolve()


@pytest.mark.parametrize("value", ["hot", "-1", "2.5"])
def test_invalid_temperature_is_rejected(value: str) -> None:
    os.environ.update(OPENAI_API_KEY="sk-test", TEMPERATURE=value)
    with pytest.raises(ConfigError, match="TEMPERATURE"):
        load_settings(env_file=None)


def test_missing_data_dir_is_rejected(tmp_path: Path) -> None:
    os.environ.update(OPENAI_API_KEY="sk-test", DATA_DIR=str(tmp_path / "missing"))
    with pytest.raises(ConfigError, match="DATA_DIR: directory does not exist"):
        load_settings(env_file=None)


def test_unknown_log_level_is_rejected() -> None:
    os.environ.update(OPENAI_API_KEY="sk-test", LOG_LEVEL="LOUD")
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        load_settings(env_file=None)


def test_api_key_is_hidden_when_settings_are_printed() -> None:
    os.environ["OPENAI_API_KEY"] = "sk-very-secret"

    settings = load_settings(env_file=None)

    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in str(settings)


def test_settings_cannot_be_changed_after_loading() -> None:
    os.environ["OPENAI_API_KEY"] = "sk-test"
    settings = load_settings(env_file=None)

    with pytest.raises(ValidationError):
        settings.model_name = "something-else"


def test_settings_are_loaded_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-from-file\nMODEL_NAME=gpt-test\n", encoding="utf-8"
    )

    settings = load_settings(env_file=env_file)

    assert settings.openai_api_key.get_secret_value() == "sk-from-file"
    assert settings.model_name == "gpt-test"
