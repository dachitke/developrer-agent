"""Local file reader tool.

Reads .txt, .md and .json files from one directory only.
The path is resolved and must stay inside that directory, so ../ traversal,
absolute paths and symlinks that point outside are rejected.
"""

import logging
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = frozenset({".txt", ".md", ".json"})
MAX_FILE_CHARS = 8_000
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class FileReaderError(ValueError):
    """Raised when a file cannot be read safely."""


class FileReaderInput(BaseModel):
    """Arguments the LLM must provide when calling the file reader tool."""

    filename: str = Field(
        description=(
            "Name of a file inside the data directory, for example "
            "'example.txt', 'notes.md' or 'numbers.json'."
        )
    )


def create_file_reader(data_dir: Path) -> BaseTool:
    """Build a read_file tool locked to one directory."""
    root = Path(data_dir)

    @tool("read_file", args_schema=FileReaderInput)
    def read_file(filename: str) -> str:
        """Read a .txt, .md or .json file from the project's data directory.

        Pass only the file name, for example 'example.txt' or 'numbers.json'.
        Files outside the data directory cannot be read.
        """
        try:
            return read_data_file(filename, root)
        except FileReaderError as exc:
            logger.info("File reader rejected %r: %s", filename, exc)
            return f"Error: {exc}"

    return read_file


file_reader = create_file_reader(DEFAULT_DATA_DIR)


def read_data_file(filename: str, data_dir: Path) -> str:
    """Read one allowed file from data_dir.

    Raises FileReaderError for empty names, unsafe paths, unsupported types,
    missing files, permission problems and files that are not UTF-8 text.
    """
    filename = filename.strip()
    if not filename:
        raise FileReaderError("filename is empty")

    requested = Path(filename)
    if requested.is_absolute():
        raise FileReaderError("absolute paths are not allowed")
    if ".." in requested.parts:
        raise FileReaderError("path traversal is not allowed")

    suffix = requested.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        shown = suffix or "(no extension)"
        raise FileReaderError(f"unsupported file type {shown} (allowed: {allowed})")

    root = data_dir.resolve()
    if not root.is_dir():
        raise FileReaderError(f"data directory does not exist: {root}")

    candidate = (root / requested).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise FileReaderError("path is outside the data directory")
    if not candidate.exists():
        raise FileReaderError(f"file does not exist: {requested.as_posix()}")
    if not candidate.is_file():
        raise FileReaderError(f"not a file: {requested.as_posix()}")

    try:
        content = candidate.read_text(encoding="utf-8")
    except PermissionError:
        raise FileReaderError(f"permission denied: {requested.as_posix()}") from None
    except UnicodeDecodeError:
        raise FileReaderError(f"file is not valid UTF-8: {requested.as_posix()}") from None
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise FileReaderError(f"could not read file: {requested.as_posix()} ({detail})") from None

    if len(content) > MAX_FILE_CHARS:
        hidden = len(content) - MAX_FILE_CHARS
        return content[:MAX_FILE_CHARS] + f"\n\n[truncated: {hidden} more characters not shown]"
    return content
