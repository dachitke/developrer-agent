"""Structured output schemas.

AgentResponse is the application's own result object. The LLM is not asked
to print JSON; the agent fills this model after the tool loop finishes.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentResponse(BaseModel):
    """The final result returned to the CLI."""

    model_config = ConfigDict(frozen=True)

    answer: str
    tools_used: list[str] = Field(default_factory=list)
    tool_calls: int = Field(default=0, ge=0)
    success: bool = True
    error: str | None = None

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer cannot be empty")
        return stripped

    @field_validator("tools_used")
    @classmethod
    def tool_names_must_not_be_blank(cls, value: list[str]) -> list[str]:
        names = [name.strip() for name in value]
        if any(not name for name in names):
            raise ValueError("tool name cannot be empty")
        return names

    @field_validator("error")
    @classmethod
    def blank_error_becomes_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def error_must_match_success(self) -> "AgentResponse":
        if self.success and self.error is not None:
            raise ValueError("error must be empty when success is true")
        if not self.success and self.error is None:
            raise ValueError("error is required when success is false")
        return self
