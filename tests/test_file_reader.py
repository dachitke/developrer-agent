"""Tests for the local file reader tool."""

import json
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.tools.file_reader import (
    FileReaderError,
    create_file_reader,
    file_reader,
    read_data_file,
)


def write_file(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    write_file(directory, "example.txt", "hello from the data directory")
    write_file(directory, "notes.md", "# Notes\n\nA markdown file.")
    write_file(directory, "numbers.json", '{"values": [1, 2, 3]}')
    return directory


def test_reads_txt_md_and_json(data_dir: Path) -> None:
    reader = create_file_reader(data_dir)

    assert reader.invoke({"filename": "example.txt"}) == "hello from the data directory"
    assert reader.invoke({"filename": "notes.md"}).startswith("# Notes")
    assert json.loads(reader.invoke({"filename": "numbers.json"})) == {"values": [1, 2, 3]}


def test_relative_path_inside_the_directory_is_allowed(data_dir: Path) -> None:
    write_file(data_dir, "nested/info.txt", "nested file")

    assert read_data_file("./example.txt", data_dir) == "hello from the data directory"
    assert read_data_file("nested/info.txt", data_dir) == "nested file"


def test_missing_file_returns_an_error(data_dir: Path) -> None:
    result = create_file_reader(data_dir).invoke({"filename": "missing.txt"})

    assert result == "Error: file does not exist: missing.txt"


@pytest.mark.parametrize("filename", ["script.py", ".env", "notes", "data.exe.txt.bak"])
def test_unsupported_extension_is_rejected(data_dir: Path, filename: str) -> None:
    result = create_file_reader(data_dir).invoke({"filename": filename})

    assert result.startswith("Error: unsupported file type")
    assert ".json, .md, .txt" in result


@pytest.mark.parametrize(
    "filename",
    ["../secret.txt", "..\\secret.txt", "nested/../../secret.txt", "example.txt/../../../.env"],
)
def test_path_traversal_is_rejected(data_dir: Path, tmp_path: Path, filename: str) -> None:
    secret = write_file(tmp_path, "secret.txt", "top secret")
    write_file(tmp_path, ".env", "OPENAI_API_KEY=sk-real")

    result = create_file_reader(data_dir).invoke({"filename": filename})

    assert result.startswith("Error:")
    assert "path traversal is not allowed" in result
    assert "top secret" not in result
    assert "sk-real" not in result
    assert secret.is_file()


def test_absolute_path_is_rejected_even_when_the_file_exists(data_dir: Path, tmp_path: Path) -> None:
    secret = write_file(tmp_path, "secret.txt", "top secret")

    result = create_file_reader(data_dir).invoke({"filename": str(secret)})

    assert result == "Error: absolute paths are not allowed"
    assert "top secret" not in result


def test_sibling_directory_with_a_similar_name_cannot_be_read(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_file(tmp_path / "data_backup", "notes.txt", "backup secret")

    result = create_file_reader(data_dir).invoke({"filename": "../data_backup/notes.txt"})

    assert "path traversal is not allowed" in result
    assert "backup secret" not in result


def test_symlink_pointing_outside_is_rejected(data_dir: Path, tmp_path: Path) -> None:
    secret = write_file(tmp_path, "secret.txt", "linked secret")
    link = data_dir / "link.txt"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("this system does not allow creating symlinks")

    result = create_file_reader(data_dir).invoke({"filename": "link.txt"})

    assert result == "Error: path is outside the data directory"
    assert "linked secret" not in result


def test_permission_error_is_reported(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def deny_read(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", deny_read)

    result = create_file_reader(data_dir).invoke({"filename": "example.txt"})

    assert result == "Error: permission denied: example.txt"


def test_invalid_utf8_is_reported(data_dir: Path) -> None:
    (data_dir / "binary.txt").write_bytes(b"\xff\xfe not utf-8")

    result = create_file_reader(data_dir).invoke({"filename": "binary.txt"})

    assert result == "Error: file is not valid UTF-8: binary.txt"


def test_directory_is_not_read_as_a_file(data_dir: Path) -> None:
    (data_dir / "folder.txt").mkdir()

    result = create_file_reader(data_dir).invoke({"filename": "folder.txt"})

    assert result == "Error: not a file: folder.txt"


def test_long_file_is_truncated(data_dir: Path) -> None:
    write_file(data_dir, "long.txt", "a" * 8_050)

    result = read_data_file("long.txt", data_dir)

    assert result.startswith("a" * 8_000)
    assert result.endswith("[truncated: 50 more characters not shown]")
    assert len(result) < 8_050


@pytest.mark.parametrize("filename", ["", "   "])
def test_empty_filename_is_rejected(data_dir: Path, filename: str) -> None:
    with pytest.raises(FileReaderError, match="filename is empty"):
        read_data_file(filename, data_dir)


def test_read_data_file_raises_instead_of_returning_error_text(data_dir: Path) -> None:
    with pytest.raises(FileReaderError, match="file does not exist"):
        read_data_file("missing.txt", data_dir)


def test_tool_exposes_name_description_and_argument_to_the_llm(data_dir: Path) -> None:
    reader = create_file_reader(data_dir)

    assert reader.name == "read_file"
    assert "data directory" in reader.description
    assert list(reader.args) == ["filename"]


def test_project_sample_files_can_be_read() -> None:
    text = file_reader.invoke({"filename": "example.txt"})
    notes = file_reader.invoke({"filename": "notes.md"})
    numbers = json.loads(file_reader.invoke({"filename": "numbers.json"}))

    assert "LangChain" in text
    assert "calculator" in notes
    assert numbers["values"] == [10, 20, 30, 40, 50]
    assert file_reader.invoke({"filename": "../.env"}).startswith("Error:")
    assert (PROJECT_ROOT / "data").is_dir()
