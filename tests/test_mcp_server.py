import asyncio
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


EXPECTED_TOOL_NAMES = (
    "import_book",
    "list_books",
    "library_status",
    "search_books",
    "get_passages",
    "save_reading_note",
)


@pytest.fixture
def server_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    root = tmp_path / "mcp-library"
    monkeypatch.setenv("BOOK_LIBRARY_ROOT", str(root))
    sys.modules.pop("book_agent.mcp_server", None)
    module = importlib.import_module("book_agent.mcp_server")
    yield module
    sys.modules.pop("book_agent.mcp_server", None)


def test_tool_names_and_actual_fastmcp_registration_are_exact(
    server_module: ModuleType,
) -> None:
    registered = asyncio.run(server_module.mcp.list_tools())

    assert server_module.TOOL_NAMES == EXPECTED_TOOL_NAMES
    assert tuple(tool.name for tool in registered) == EXPECTED_TOOL_NAMES
    assert all(tool.description for tool in registered)
    assert Path(server_module.ROOT).is_absolute()


def test_import_is_silent_and_does_not_start_the_server(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / "subprocess-library"
    environment = os.environ.copy()
    environment["BOOK_LIBRARY_ROOT"] = str(root)

    completed = subprocess.run(
        [sys.executable, "-c", "import book_agent.mcp_server"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert (root / "data" / "library.sqlite3").is_file()
