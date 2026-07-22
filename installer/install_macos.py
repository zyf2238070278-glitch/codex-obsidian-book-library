#!/usr/bin/env python3
"""Install the local book library for macOS from an extracted release bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
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
    "start_ocr",
    "start_pending_ocr",
    "ocr_status",
    "pause_ocr",
    "sync_book_catalog",
)

VAULT_DIRECTORIES = (
    Path("书库/00-待导入"),
    Path("书库/10-原始书籍"),
    Path("书库/20-解析文本"),
    Path("书库/30-AI读书笔记"),
    Path("书库/40-OCR报告"),
    Path("书库/50-书目卡片"),
)


class InstallError(RuntimeError):
    """An expected installation failure with a user-facing message."""


VISION_HELPER_RELATIVE = Path("bin/book-vision-ocr")
VISION_HELPER_SCHEMA_VERSION = 2
VISION_HELPER_LANGUAGES = frozenset({"zh-Hans", "en-US"})
VISION_HELPER_LIPO = "/usr/bin/lipo"
VISION_HELPER_CODESIGN = "/usr/bin/codesign"
VISION_HELPER_MACHO_MAGICS = frozenset({b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"})
RAPIDOCR_MODEL_FILES = (
    "PP-OCRv6_det_small.onnx",
    "PP-OCRv6_rec_small.onnx",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
)
LIGHT_OCR_REQUIRED_RUNTIME_FILES = (
    Path("node_modules/@arcships/light-ocr/package.json"),
    Path("node_modules/@arcships/light-ocr-darwin-arm64/native/light_ocr_node.node"),
    Path("node_modules/@arcships/light-ocr-model-ppocrv6-small/bundle/manifest.json"),
    Path("node_modules/@arcships/light-ocr-model-ppocrv6-small/bundle/det/inference.onnx"),
    Path("node_modules/@arcships/light-ocr-model-ppocrv6-small/bundle/rec/inference.onnx"),
)


@dataclass(frozen=True)
class InstallResult:
    project_root: Path
    vault: Path
    config: Path
    python: Path
    light_ocr_node: Path | None = None


def default_project_root() -> Path:
    """Return the extracted distribution root containing this installer."""

    return Path(__file__).resolve().parents[1]


def _absolute(path: Path, base: Optional[Path] = None) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = (base or Path.cwd()) / expanded
    return expanded.resolve(strict=False)


def _absolute_without_symlink_resolution(
    path: Path, base: Optional[Path] = None
) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = (base or Path.cwd()) / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _toml_string(value: object) -> str:
    # JSON escapes quotes, backslashes, and C0 controls like TOML basic strings.
    # TOML also forbids DEL, which JSON otherwise leaves literal.
    return json.dumps(str(value), ensure_ascii=False).replace("\x7f", "\\u007f")


def render_codex_config(
    *,
    project_root: Path,
    vault: Path,
    python: Path,
    light_ocr_node: Path | None = None,
) -> str:
    tools = ",\n".join("  %s" % _toml_string(tool) for tool in ENABLED_TOOLS)
    light_ocr_environment = (
        "BOOK_LIBRARY_LIGHT_OCR_NODE = %s\n" % _toml_string(light_ocr_node)
        if light_ocr_node is not None
        else ""
    )
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
{light_ocr_environment}
""".format(
        python=_toml_string(python),
        project_root=_toml_string(project_root),
        vault=_toml_string(vault),
        tools=tools,
        light_ocr_environment=light_ocr_environment,
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
        "--extra",
        "ocr",
        "--python",
        "3.12",
    ]
    child_environment = os.environ.copy()
    child_environment["UV_PROJECT_ENVIRONMENT"] = str(project_root / ".venv")
    try:
        run_command(
            command,
            cwd=project_root,
            check=True,
            env=child_environment,
        )
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


def _validate_python(python: Path) -> None:
    if not python.is_file():
        raise InstallError("Python 解释器不存在或不是文件：%s" % python)
    if not os.access(str(python), os.X_OK):
        raise InstallError("Python 解释器不可执行：%s" % python)


def _prepare_semantic_model(
    *, project_root: Path, python: Path, run_command: Callable[..., Any]
) -> None:
    command = [
        str(python),
        "-m",
        "installer.model_assets",
        "--model-root",
        str(project_root / "data" / "models"),
        "--manifest",
        str(project_root / "distribution" / "model-manifest.json"),
    ]
    try:
        run_command(command, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise InstallError(
            f"语义模型下载或校验失败（退出码 {exc.returncode}）。"
            "请根据上方错误检查网络、磁盘或模型缓存后重试。"
        ) from exc
    except OSError as exc:
        raise InstallError(f"无法运行语义模型安装：{exc}") from exc


def _run_runtime_selftest(
    *,
    project_root: Path,
    vault: Path,
    python: Path,
    run_command: Callable[..., Any],
) -> None:
    command = [
        str(python),
        "-m",
        "installer.runtime_selftest",
        "--project-root",
        str(project_root),
        "--vault",
        str(vault),
    ]
    try:
        run_command(command, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise InstallError(
            f"安装自检失败（退出码 {exc.returncode}）。请根据上方错误修复后重试。"
        ) from exc
    except OSError as exc:
        raise InstallError(f"无法运行安装自检：{exc}") from exc


def _run_validation_command(
    runner: Callable[..., Any],
    argv: list[str],
    *,
    cwd: Path,
) -> Any:
    """Run a validation command with argv only (never through a shell).

    The small compatibility fallback keeps the injectable runner used by the
    installer tests useful even when it does not accept capture_output/text.
    """

    try:
        return runner(
            argv,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except TypeError as exc:
        try:
            return runner(argv, cwd=cwd, check=True)
        except TypeError:
            raise exc


def _validate_vision_helper(
    *,
    project_root: Path,
    helper: Optional[Path] = None,
    run_command: Callable[..., Any],
) -> Path:
    """Validate the packaged Apple Vision helper before publishing config."""

    helper = helper or (project_root / VISION_HELPER_RELATIVE)
    try:
        info = helper.lstat()
    except OSError as exc:
        raise InstallError(
            "缺少 Apple Vision OCR helper：%s。请重新下载完整 macOS 安装包。" % helper
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise InstallError("Apple Vision OCR helper 必须是普通文件：%s" % helper)
    if stat.S_IMODE(info.st_mode) != 0o755 or not os.access(str(helper), os.X_OK):
        raise InstallError("Apple Vision OCR helper 必须具有 0755 可执行权限：%s" % helper)
    try:
        with helper.open("rb") as source:
            if source.read(4) not in VISION_HELPER_MACHO_MAGICS:
                raise InstallError("Apple Vision OCR helper 不是 64 位 Mach-O 文件：%s" % helper)
    except OSError as exc:
        raise InstallError("无法读取 Apple Vision OCR helper：%s" % helper) from exc

    try:
        architecture = _run_validation_command(
            run_command, [VISION_HELPER_LIPO, "-archs", str(helper)], cwd=project_root
        )
        if (getattr(architecture, "stdout", "") or "").strip().split() != ["arm64"]:
            raise InstallError("Apple Vision OCR helper 必须是 arm64：%s" % helper)
        _run_validation_command(
            run_command,
            [VISION_HELPER_CODESIGN, "--verify", "--strict", str(helper)],
            cwd=project_root,
        )
        capabilities = _run_validation_command(
            run_command, [str(helper), "--capabilities"], cwd=project_root
        )
    except InstallError:
        raise
    except subprocess.CalledProcessError as exc:
        raise InstallError("Apple Vision OCR helper 校验失败：退出码 %s" % exc.returncode) from exc
    except OSError as exc:
        raise InstallError("无法校验 Apple Vision OCR helper：%s" % exc) from exc

    stderr = getattr(capabilities, "stderr", "") or ""
    if stderr:
        raise InstallError("Apple Vision OCR helper capabilities 输出了错误信息。")
    try:
        payload = json.loads(getattr(capabilities, "stdout", "") or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise InstallError("Apple Vision OCR helper capabilities 不是有效 JSON。") from exc
    if type(payload) is not dict or set(payload) != {"schema_version", "languages"}:
        raise InstallError("Apple Vision OCR helper capabilities schema 不受支持。")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != VISION_HELPER_SCHEMA_VERSION
    ):
        raise InstallError("Apple Vision OCR helper schema 版本不受支持。")
    languages = payload.get("languages")
    if (
        type(languages) is not list
        or any(type(language) is not str or not language for language in languages)
        or len(set(languages)) != len(languages)
        or not VISION_HELPER_LANGUAGES.issubset(languages)
    ):
        raise InstallError("Apple Vision OCR helper 不支持中文或英文识别。")
    return helper


def _create_runtime_directories(project_root: Path, vault: Path) -> None:
    try:
        for relative in VAULT_DIRECTORIES:
            (vault / relative).mkdir(parents=True, exist_ok=True)
        (project_root / "data").mkdir(parents=True, exist_ok=True)
        (project_root / "data" / "models").mkdir(parents=True, exist_ok=True)
        (project_root / "data" / "ocr-models").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InstallError("无法创建书库目录：%s" % exc) from exc


def _install_rapidocr_models(project_root: Path) -> None:
    """Copy RapidOCR's wheel-bundled models to the stable runtime location."""

    source = (
        project_root
        / ".venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "rapidocr"
        / "models"
    )
    destination = project_root / "data" / "ocr-models" / "rapidocr"
    try:
        for filename in RAPIDOCR_MODEL_FILES:
            model = source / filename
            if not model.is_file() or model.is_symlink():
                raise InstallError(
                    "RapidOCR 模型缺失：%s。请重新运行安装脚本。" % filename
                )
        destination.mkdir(parents=True, exist_ok=True)
        for filename in RAPIDOCR_MODEL_FILES:
            shutil.copy2(source / filename, destination / filename)
    except InstallError:
        raise
    except OSError as exc:
        raise InstallError("无法准备 RapidOCR 本地模型：%s" % exc) from exc


def _install_light_ocr_runtime(
    project_root: Path,
    *,
    find_executable: Callable[[str], Optional[str]],
    run_command: Callable[..., Any],
) -> Path:
    """Install and validate the pinned macOS Apple Silicon Light OCR runtime."""

    required_sources = (
        project_root / "package.json",
        project_root / "package-lock.json",
        project_root / "scripts" / "light_ocr_worker.mjs",
    )
    for source in required_sources:
        try:
            info = source.lstat()
        except OSError as exc:
            raise InstallError("Light OCR 安装文件缺失：%s" % source) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise InstallError("Light OCR 安装文件必须是普通文件：%s" % source)

    bundled_node = project_root / "runtime" / "node" / "bin" / "node"
    node_text = str(bundled_node) if bundled_node.is_file() else find_executable("node")
    npm_text = find_executable("npm")
    if not node_text or not npm_text:
        raise InstallError("Light OCR 需要 Node.js 22 或 24 以及 npm。")
    node = Path(node_text).expanduser().resolve()
    npm = Path(npm_text).expanduser().resolve()
    for executable, label in ((node, "Node.js"), (npm, "npm")):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise InstallError("%s 不存在或不可执行：%s" % (label, executable))

    try:
        version = _run_validation_command(
            run_command, [str(node), "--version"], cwd=project_root
        )
        match = re.fullmatch(
            r"v(\d+)\.\d+\.\d+",
            (getattr(version, "stdout", "") or "").strip(),
        )
        if match is None or int(match.group(1)) not in {22, 24}:
            raise InstallError("Light OCR 需要 Node.js 22 或 24。")
        architecture = _run_validation_command(
            run_command,
            [str(node), "-p", "process.arch"],
            cwd=project_root,
        )
        if (getattr(architecture, "stdout", "") or "").strip() != "arm64":
            raise InstallError("当前 macOS 版 Light OCR 只支持 arm64。")
        child_environment = os.environ.copy()
        child_environment["PATH"] = str(node.parent) + os.pathsep + child_environment.get(
            "PATH", ""
        )
        run_command(
            [str(npm), "ci", "--omit=dev", "--ignore-scripts"],
            cwd=project_root,
            check=True,
            env=child_environment,
        )
    except InstallError:
        raise
    except subprocess.CalledProcessError as exc:
        raise InstallError(
            "Light OCR 依赖安装失败（退出码 %s）。" % exc.returncode
        ) from exc
    except OSError as exc:
        raise InstallError("无法安装 Light OCR：%s" % exc) from exc

    for relative in LIGHT_OCR_REQUIRED_RUNTIME_FILES:
        path = project_root / relative
        try:
            info = path.lstat()
        except OSError as exc:
            raise InstallError("Light OCR 运行文件缺失：%s" % relative) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise InstallError("Light OCR 运行文件必须是普通文件：%s" % relative)
    return node


def install(
    *,
    project_root: Path,
    vault: Optional[Path] = None,
    skip_sync: bool = False,
    codex_config: Optional[Path] = None,
    python: Optional[Path] = None,
    vision_helper: Optional[Path] = None,
    find_executable: Optional[Callable[[str], Optional[str]]] = None,
    run_command: Optional[Callable[..., Any]] = None,
) -> InstallResult:
    """Install dependencies, create runtime directories, and write local config."""

    resolved_root = _absolute(Path(project_root))
    resolved_vault = _absolute_without_symlink_resolution(
        Path(vault) if vault is not None else resolved_root / "Obsidian书库"
    )
    resolved_config = _absolute(
        Path(codex_config)
        if codex_config is not None
        else resolved_root / ".codex" / "config.toml"
    )
    resolved_python = _absolute_without_symlink_resolution(
        Path(python)
        if python is not None
        else resolved_root / ".venv" / "bin" / "python"
    )
    light_ocr_node: Path | None = None

    command_runner = run_command or subprocess.run
    if not skip_sync:
        uv = _find_uv(resolved_root, find_executable or shutil.which)
        _sync_environment(
            project_root=resolved_root,
            uv=uv,
            run_command=command_runner,
        )
        _validate_python(resolved_python)
        _prepare_semantic_model(
            project_root=resolved_root,
            python=resolved_python,
            run_command=command_runner,
        )
        _install_rapidocr_models(resolved_root)
        if (resolved_root / "package.json").is_file():
            light_ocr_node = _install_light_ocr_runtime(
                resolved_root,
                find_executable=find_executable or shutil.which,
                run_command=command_runner,
            )

    # Validate the native helper before creating runtime directories or writing
    # Codex configuration.  A failed validation therefore leaves an existing
    # configuration untouched and never publishes a partial one.
    _validate_vision_helper(
        project_root=resolved_root,
        helper=(
            _absolute_without_symlink_resolution(Path(vision_helper), resolved_root)
            if vision_helper is not None
            else None
        ),
        run_command=command_runner,
    )

    if not skip_sync:
        _run_runtime_selftest(
            project_root=resolved_root,
            vault=resolved_vault,
            python=resolved_python,
            run_command=command_runner,
        )

    _create_runtime_directories(resolved_root, resolved_vault)
    try:
        _write_text_atomically(
            resolved_config,
            render_codex_config(
                project_root=resolved_root,
                vault=resolved_vault,
                python=resolved_python,
                light_ocr_node=light_ocr_node,
            ),
        )
    except OSError as exc:
        raise InstallError("无法写入 Codex 项目配置：%s" % exc) from exc

    return InstallResult(
        project_root=resolved_root,
        vault=resolved_vault,
        config=resolved_config,
        python=resolved_python,
        light_ocr_node=light_ocr_node,
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
