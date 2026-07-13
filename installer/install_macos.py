#!/usr/bin/env python3
"""Install the local book library for macOS from an extracted release bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


EXIT_SUCCESS = 0
EXIT_INSTALL_ERROR = 1

ENABLED_TOOLS = (
    "import_book",
    "list_books",
    "library_status",
    "search_books",
    "get_passages",
    "save_reading_note",
)

VAULT_DIRECTORIES = (
    Path("书库/00-待导入"),
    Path("书库/10-原始书籍"),
    Path("书库/20-解析文本"),
    Path("书库/30-AI读书笔记"),
)


class InstallError(RuntimeError):
    """An expected installation failure with a user-facing message."""


@dataclass(frozen=True)
class InstallResult:
    project_root: Path
    vault: Path
    config: Path
    python: Path


def default_project_root() -> Path:
    """Return the extracted distribution root containing this installer."""

    return Path(__file__).resolve().parents[1]


def _absolute(path: Path, base: Optional[Path] = None) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = (base or Path.cwd()) / expanded
    return expanded.resolve(strict=False)


def _toml_string(value: object) -> str:
    # JSON escapes quotes, backslashes, and C0 controls like TOML basic strings.
    # TOML also forbids DEL, which JSON otherwise leaves literal.
    return json.dumps(str(value), ensure_ascii=False).replace("\x7f", "\\u007f")


def render_codex_config(
    *, project_root: Path, vault: Path, python: Path
) -> str:
    tools = ",\n".join("  %s" % _toml_string(tool) for tool in ENABLED_TOOLS)
    return """[mcp_servers.book_library]
command = {python}
args = ["-m", "book_agent.mcp_server"]
cwd = {project_root}
required = true
enabled = true
enabled_tools = [
{tools}
]
startup_timeout_sec = 60
tool_timeout_sec = 120

[mcp_servers.book_library.env]
BOOK_LIBRARY_ROOT = {project_root}
BOOK_LIBRARY_OBSIDIAN_VAULT = {vault}
HF_HUB_OFFLINE = "1"
TRANSFORMERS_OFFLINE = "1"
""".format(
        python=_toml_string(python),
        project_root=_toml_string(project_root),
        vault=_toml_string(vault),
        tools=tools,
    )


def _write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".%s." % path.name,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _find_uv(
    project_root: Path,
    find_executable: Callable[[str], Optional[str]],
) -> Path:
    bundled_uv = project_root / "bin" / "uv"
    if bundled_uv.is_file() and os.access(str(bundled_uv), os.X_OK):
        return bundled_uv.resolve()

    path_uv = find_executable("uv")
    if path_uv:
        return _absolute(Path(path_uv))

    raise InstallError(
        "未找到 uv。请重新下载包含可执行 bin/uv 的完整 macOS Apple Silicon "
        "安装包，或先安装 uv 并重新运行。"
    )


def _sync_environment(
    *,
    project_root: Path,
    uv: Path,
    run_command: Callable[..., Any],
) -> None:
    command = [
        str(uv),
        "sync",
        "--frozen",
        "--extra",
        "semantic",
        "--python",
        "3.12",
    ]
    try:
        run_command(command, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise InstallError(
            "uv sync 安装依赖失败（退出码 %s）。请检查网络后重新运行。"
            % exc.returncode
        ) from exc
    except PermissionError as exc:
        raise InstallError(
            "uv 无法执行：%s。请确认安装包中的 bin/uv 具有执行权限。" % uv
        ) from exc
    except OSError as exc:
        raise InstallError("无法运行 uv：%s" % exc) from exc


def _create_runtime_directories(project_root: Path, vault: Path) -> None:
    try:
        for relative in VAULT_DIRECTORIES:
            (vault / relative).mkdir(parents=True, exist_ok=True)
        (project_root / "data").mkdir(parents=True, exist_ok=True)
        (project_root / "data" / "models").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InstallError("无法创建书库目录：%s" % exc) from exc


def install(
    *,
    project_root: Path,
    vault: Optional[Path] = None,
    skip_sync: bool = False,
    codex_config: Optional[Path] = None,
    python: Optional[Path] = None,
    find_executable: Optional[Callable[[str], Optional[str]]] = None,
    run_command: Optional[Callable[..., Any]] = None,
) -> InstallResult:
    """Install dependencies, create runtime directories, and write local config."""

    resolved_root = _absolute(Path(project_root))
    resolved_vault = _absolute(
        Path(vault) if vault is not None else resolved_root / "Obsidian书库"
    )
    resolved_config = _absolute(
        Path(codex_config)
        if codex_config is not None
        else resolved_root / ".codex" / "config.toml"
    )
    resolved_python = _absolute(
        Path(python)
        if python is not None
        else resolved_root / ".venv" / "bin" / "python"
    )

    if not skip_sync:
        uv = _find_uv(resolved_root, find_executable or shutil.which)
        _sync_environment(
            project_root=resolved_root,
            uv=uv,
            run_command=run_command or subprocess.run,
        )

    _create_runtime_directories(resolved_root, resolved_vault)
    try:
        _write_text_atomically(
            resolved_config,
            render_codex_config(
                project_root=resolved_root,
                vault=resolved_vault,
                python=resolved_python,
            ),
        )
    except OSError as exc:
        raise InstallError("无法写入 Codex 项目配置：%s" % exc) from exc

    return InstallResult(
        project_root=resolved_root,
        vault=resolved_vault,
        config=resolved_config,
        python=resolved_python,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="安装 Codex + Obsidian 本地书库")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
        help="发行包根目录（默认自动使用安装脚本所在发行包）",
    )
    parser.add_argument(
        "--vault",
        type=Path,
        help="Obsidian Vault；默认使用 <project-root>/Obsidian书库",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="跳过 uv 依赖安装（仅用于测试或已经准备好的环境）",
    )
    parser.add_argument(
        "--codex-config",
        type=Path,
        help="Codex 项目配置；默认使用 <project-root>/.codex/config.toml",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="写入配置的 Python；默认使用 <project-root>/.venv/bin/python",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = install(
            project_root=args.project_root,
            vault=args.vault,
            skip_sync=args.skip_sync,
            codex_config=args.codex_config,
            python=args.python,
        )
    except InstallError as exc:
        print("安装失败：%s" % exc, file=sys.stderr)
        return EXIT_INSTALL_ERROR

    print("安装完成。")
    print("Codex 项目配置：%s" % result.config)
    print("下一步：")
    print("1. 重启 Codex。")
    print("2. 用此项目新建任务。")
    print("3. 可以说：“检查书库状态”。")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
