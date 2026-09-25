"""Skill folder tool.

A skill is a folder under skills/ with one SKILL.md file. The file starts
with a name and a description, then instructions the model should follow.
The model chooses when to read a skill. This module only loads the file.
"""

import logging
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"
MAX_SKILL_CHARS = 8_000


class SkillError(ValueError):
    """Raised when a skill cannot be read safely."""


class SkillInfo(BaseModel):
    """The short description shown to the model before it opens a skill."""

    name: str
    description: str


class ReadSkillInput(BaseModel):
    """Arguments the LLM must provide when reading a skill."""

    skill_name: str = Field(
        description="Skill folder name, for example 'quadratic-equation'."
    )


def create_skill_reader(skills_dir: Path) -> BaseTool:
    """Build a read_skill tool locked to one skills directory."""
    root = Path(skills_dir)

    @tool("read_skill", args_schema=ReadSkillInput)
    def read_skill(skill_name: str) -> str:
        """Read the instructions for one skill from the skills folder.

        Call this when a listed skill matches the user's request, then follow
        those instructions. Use other tools if the skill tells you to.
        """
        try:
            return read_skill_text(skill_name, root)
        except SkillError as exc:
            logger.info("Skill reader rejected %r: %s", skill_name, exc)
            return f"Error: {exc}"

    return read_skill


def list_skills(skills_dir: Path) -> list[SkillInfo]:
    """Return the name and description of every skill in the folder."""
    root = Path(skills_dir)
    if not root.is_dir():
        return []

    skills: list[SkillInfo] = []
    for path in sorted(root.iterdir()):
        skill_file = path / "SKILL.md"
        if not path.is_dir() or not skill_file.is_file():
            continue
        name, description = _name_and_description(skill_file.read_text(encoding="utf-8"), path.name)
        skills.append(SkillInfo(name=name, description=description))
    return skills


def skill_catalog(skills_dir: Path) -> str:
    """Format the skill list that is added to the system prompt."""
    skills = list_skills(skills_dir)
    if not skills:
        return "No skills are installed."
    lines = [f"- {skill.name}: {skill.description}" for skill in skills]
    return "\n".join(lines)


def read_skill_text(skill_name: str, skills_dir: Path) -> str:
    """Read one SKILL.md file. Rejects paths that leave the skills folder."""
    name = skill_name.strip()
    if not name or any(part in name for part in ("/", "\\", "..")) or name in {".", ".."}:
        raise SkillError("invalid skill name")

    root = Path(skills_dir).resolve()
    if not root.is_dir():
        raise SkillError(f"skills directory does not exist: {root}")

    skill_file = (root / name / "SKILL.md").resolve()
    if skill_file == root or not skill_file.is_relative_to(root):
        raise SkillError("path is outside the skills directory")
    if not skill_file.is_file():
        raise SkillError(f"skill does not exist: {name}")

    try:
        content = skill_file.read_text(encoding="utf-8")
    except PermissionError:
        raise SkillError(f"permission denied: {name}") from None
    except UnicodeDecodeError:
        raise SkillError(f"skill is not valid UTF-8: {name}") from None
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise SkillError(f"could not read skill: {name} ({detail})") from None

    if len(content) > MAX_SKILL_CHARS:
        hidden = len(content) - MAX_SKILL_CHARS
        return content[:MAX_SKILL_CHARS] + f"\n\n[truncated: {hidden} more characters not shown]"
    return content


def _name_and_description(text: str, folder_name: str) -> tuple[str, str]:
    metadata, _body = _split_frontmatter(text)
    name = metadata.get("name") or folder_name
    description = metadata.get("description") or "No description provided."
    return name, description


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()

    end = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        return {}, text.strip()

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    body = "\n".join(lines[end + 1 :]).strip()
    return metadata, body
