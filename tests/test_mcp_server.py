import asyncio
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tomllib
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
    obsidian_vault = tmp_path / "Obsidian_workspace"
    obsidian_vault.mkdir()
    monkeypatch.setenv("BOOK_LIBRARY_ROOT", str(root))
    monkeypatch.setenv("BOOK_LIBRARY_OBSIDIAN_VAULT", str(obsidian_vault))
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
    assert server_module.OBSIDIAN_VAULT == Path(
        os.environ["BOOK_LIBRARY_OBSIDIAN_VAULT"]
    )
    assert server_module.library_tools.paths.vault == server_module.OBSIDIAN_VAULT
    assert server_module.library_tools.paths.database.is_relative_to(
        server_module.ROOT
    )
    assert not (server_module.ROOT / "vault").exists()


def test_import_schema_requires_a_described_codex_attachment_file_path(
    server_module: ModuleType,
) -> None:
    registered = asyncio.run(server_module.mcp.list_tools())
    import_tool = next(tool for tool in registered if tool.name == "import_book")
    properties = import_tool.inputSchema["properties"]
    description = properties["file_path"]["description"].lower()

    assert import_tool.inputSchema["required"] == ["file_path"]
    assert "file_path" in properties
    assert "source" not in properties
    assert "codex" in description
    assert "attachment" in description
    assert "local" in description
    assert "absolute" in description


def test_schema_field_dependency_is_declared_directly() -> None:
    project = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert any(dependency.startswith("pydantic>=") for dependency in dependencies)


def test_import_is_silent_and_does_not_start_the_server(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / "subprocess-library"
    obsidian_vault = tmp_path / "subprocess-obsidian"
    obsidian_vault.mkdir()
    environment = os.environ.copy()
    environment["BOOK_LIBRARY_ROOT"] = str(root)
    environment["BOOK_LIBRARY_OBSIDIAN_VAULT"] = str(obsidian_vault)

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
    assert (root / "data" / "models").is_dir()
    for directory in (
        "00-待导入",
        "10-原始书籍",
        "20-解析文本",
        "30-AI读书笔记",
    ):
        assert (obsidian_vault / "书库" / directory).is_dir()
    assert not (root / "vault").exists()
