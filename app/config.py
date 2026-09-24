"""Application configuration.

Loads settings from environment variables (and the optional .env file),
validates them once at startup, and configures logging.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
LOG_FILE = PROJECT_ROOT / "agent.log"

DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_LEVEL = "INFO"

API_KEY_PLACEHOLDER = "your_api_key_here"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


class Settings(BaseModel):
    """Validated, read-only application settings."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    openai_api_key: SecretStr
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    data_dir: Path = Path(DEFAULT_DATA_DIR)
    log_level: str = DEFAULT_LOG_LEVEL

    @field_validator("data_dir")
    @classmethod
    def resolve_data_dir(cls, value: Path) -> Path:
        # Relative paths are anchored to the project root, not the current
        # working directory, so the app behaves the same wherever it is run from.
        path = value if value.is_absolute() else PROJECT_ROOT / value
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"directory does not exist: {path}")
        return path

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in VALID_LOG_LEVELS:
            raise ValueError(f"must be one of {', '.join(VALID_LOG_LEVELS)}")
        return level


def load_settings(env_file: Path | None = ENV_FILE) -> Settings:
    """Read settings from the environment and validate them.

    Variables already set in the real environment take priority over the
    .env file. Raises ConfigError with a readable message on any problem.
    """
    if env_file is not None:
        load_dotenv(env_file)

    api_key = _read_env("OPENAI_API_KEY")
    if not api_key or api_key == API_KEY_PLACEHOLDER:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env "
            "and replace the placeholder with your OpenAI API key."
        )

    try:
        return Settings.model_validate(
            {
                "openai_api_key": api_key,
                "model_name": _read_env("MODEL_NAME", DEFAULT_MODEL_NAME),
                "temperature": _read_env("TEMPERATURE", str(DEFAULT_TEMPERATURE)),
                "data_dir": _read_env("DATA_DIR", DEFAULT_DATA_DIR),
                "log_level": _read_env("LOG_LEVEL", DEFAULT_LOG_LEVEL),
            }
        )
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def setup_logging(level: str, log_file: Path = LOG_FILE) -> None:
    """Write log records to a file so technical details never clutter the CLI."""
    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # These libraries log every HTTP request at INFO level, which is noise here.
    for noisy_logger in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _read_env(name: str, default: str = "") -> str:
    """Return an environment variable, treating unset and blank the same way."""
    value = os.getenv(name, "").strip()
    return value or default


def _format_validation_error(exc: ValidationError) -> str:
    """Turn Pydantic's error list into one line per bad environment variable."""
    lines = ["Invalid configuration:"]
    for error in exc.errors():
        env_var = str(error["loc"][0]).upper()
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"  - {env_var}: {message}")
    return "\n".join(lines)
