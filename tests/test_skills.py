"""Tests for loading skills from the skills folder."""

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.tools.skills import (
    SkillError,
    create_skill_reader,
    list_skills,
    read_skill_text,
    skill_catalog,
)


def write_skill(skills_dir: Path, name: str, description: str, body: str) -> None:
    folder = skills_dir / name
    folder.mkdir(parents=True)
    folder.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    root.mkdir()
    write_skill(root, "quadratic-equation", "Find real roots.", "Use the calculator.")
    return root


def test_catalog_lists_name_and_description(skills_dir: Path) -> None:
    skills = list_skills(skills_dir)

    assert [(skill.name, skill.description) for skill in skills] == [
        ("quadratic-equation", "Find real roots.")
    ]
    assert "quadratic-equation: Find real roots." in skill_catalog(skills_dir)


def test_read_skill_returns_the_instructions(skills_dir: Path) -> None:
    result = create_skill_reader(skills_dir).invoke({"skill_name": "quadratic-equation"})

    assert "Use the calculator." in result
    assert result.startswith("---")


def test_missing_skill_returns_an_error(skills_dir: Path) -> None:
    result = create_skill_reader(skills_dir).invoke({"skill_name": "missing-skill"})

    assert result == "Error: skill does not exist: missing-skill"


@pytest.mark.parametrize(
    "skill_name",
    ["../secret", "..\\secret", "quadratic-equation/../../.env", "skills/../.env"],
)
def test_path_traversal_is_rejected(skills_dir: Path, skill_name: str) -> None:
    result = create_skill_reader(skills_dir).invoke({"skill_name": skill_name})

    assert result.startswith("Error:")
    assert "invalid skill name" in result


def test_read_skill_text_raises_for_a_missing_skill(skills_dir: Path) -> None:
    with pytest.raises(SkillError, match="skill does not exist"):
        read_skill_text("missing-skill", skills_dir)


def test_tool_exposes_name_and_argument_to_the_llm(skills_dir: Path) -> None:
    reader = create_skill_reader(skills_dir)

    assert reader.name == "read_skill"
    assert "skill" in reader.description
    assert list(reader.args) == ["skill_name"]


def test_project_skills_can_be_read() -> None:
    catalog = skill_catalog(PROJECT_ROOT / "skills")
    quadratic = read_skill_text("quadratic-equation", PROJECT_ROOT / "skills")

    assert "quadratic-equation:" in catalog
    assert "project-summary:" in catalog
    assert "discriminant" in quadratic
