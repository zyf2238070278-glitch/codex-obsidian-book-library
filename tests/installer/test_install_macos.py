from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import pytest

from installer import install_macos


EXPECTED_TOOLS = [
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
]


@pytest.fixture(autouse=True)
def _legacy_vision_fixture(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep pre-OCR installer tests focused on their original behavior."""

    # New helper-validation tests deliberately exercise missing/corrupt files;
    # leave those calls untouched.
    if any(
        token in request.node.name.casefold()
        for token in ("vision", "helper", "ocr", "tesseract", "capabilities", "schema", "language")
    ):
        return
    original = install_macos.install

    def wrapped(*args: object, **kwargs: object) -> install_macos.InstallResult:
        project_root = Path(kwargs.get("project_root", args[0] if args else "."))
        helper = project_root / "bin" / "book-vision-ocr"
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 64)
        helper.chmod(0o755)
        original_runner = kwargs.get("run_command")

        def runner(argv: list[str], **runner_kwargs: object) -> subprocess.CompletedProcess[str]:
            if Path(argv[0]).name == "lipo":
                return subprocess.CompletedProcess(argv, 0, "arm64\n", "")
            if Path(argv[0]).name == "codesign":
                return subprocess.CompletedProcess(argv, 0, "", "")
            if argv[-1:] == ["--capabilities"]:
                return subprocess.CompletedProcess(
                    argv, 0,
                    '{"schema_version":2,"languages":["zh-Hans","en-US"]}',
                    "",
                )
            if original_runner is None:
                result = subprocess.run(argv, **runner_kwargs)
            else:
                result = original_runner(argv, **runner_kwargs)  # type: ignore[misc]
            if argv[1:2] == ["sync"]:
                models = (
                    project_root
                    / ".venv"
                    / "lib"
                    / "python3.12"
                    / "site-packages"
                    / "rapidocr"
                    / "models"
                )
                models.mkdir(parents=True, exist_ok=True)
                for model in install_macos.RAPIDOCR_MODEL_FILES:
                    (models / model).write_bytes(b"fixture model")
            return result

        kwargs["run_command"] = runner
        return original(*args, **kwargs)

    monkeypatch.setattr(install_macos, "install", wrapped)


def _create_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)


def test_install_rapidocr_models_copies_pinned_models_from_venv(tmp_path: Path) -> None:
    project_root = tmp_path / "Book Library Release"
    source = (
        project_root
        / ".venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "rapidocr"
        / "models"
    )
    source.mkdir(parents=True)
    for model in install_macos.RAPIDOCR_MODEL_FILES:
        (source / model).write_bytes(model.encode("utf-8"))

    install_macos._install_rapidocr_models(project_root)

    destination = project_root / "data" / "ocr-models" / "rapidocr"
    assert {
        path.name: path.read_bytes()
        for path in destination.iterdir()
    } == {
        model: model.encode("utf-8") for model in install_macos.RAPIDOCR_MODEL_FILES
    }


def test_default_project_root_is_distribution_root() -> None:
    assert install_macos.default_project_root() == Path(
        install_macos.__file__
    ).resolve().parents[1]


def test_install_uses_default_vault_and_creates_runtime_directories(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / 'Book Library "Release" \\ Apple Silicon'
    python = project_root / ".venv" / "bin" / "python"

    result = install_macos.install(
        project_root=project_root,
        python=python,
        skip_sync=True,
    )

    expected_vault = project_root / "Obsidian书库"
    assert result.vault == expected_vault.resolve()
    for relative in (
        "书库/00-待导入",
        "书库/10-原始书籍",
        "书库/20-解析文本",
        "书库/30-AI读书笔记",
    ):
        assert (expected_vault / relative).is_dir()
    assert (project_root / "data").is_dir()
    assert (project_root / "data" / "models").is_dir()
    assert (project_root / "data" / "ocr-models").is_dir()
    for obsolete in (
        "书库/00-原始书籍",
        "书库/10-解析文本",
        "书库/20-索引",
    ):
        assert not (expected_vault / obsolete).exists()




def test_install_uses_explicit_vault_and_codex_config(tmp_path: Path) -> None:
    project_root = tmp_path / "Release With Spaces"
    vault = tmp_path / "My Obsidian Vault"
    config_path = tmp_path / "machine config" / "book-library.toml"
    python = tmp_path / "Python With Spaces" / "python3"

    result = install_macos.install(
        project_root=project_root,
        vault=vault,
        codex_config=config_path,
        python=python,
        skip_sync=True,
    )

    assert result.vault == vault.resolve()
    assert result.config == config_path.resolve()
    assert config_path.is_file()
    assert not (project_root / "Obsidian书库").exists()


def test_generated_config_has_absolute_paths_offline_env_and_six_tools(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / 'Release "Quoted" \\ Root'
    vault = tmp_path / 'Vault "Quoted" \\ Notes'
    python = project_root / ".venv" / "bin" / "python"

    result = install_macos.install(
        project_root=project_root,
        vault=vault,
        python=python,
        skip_sync=True,
    )

    parsed = tomllib.loads(result.config.read_text(encoding="utf-8"))
    server = parsed["mcp_servers"]["book_library"]
    assert server["command"] == str(python.resolve())
    assert server["args"] == ["-m", "book_agent.mcp_server"]
    assert server["cwd"] == str(project_root.resolve())
    assert server["required"] is True
    assert server["enabled"] is True
    assert server["enabled_tools"] == EXPECTED_TOOLS
    assert server["env"] == {
        "BOOK_LIBRARY_ROOT": str(project_root.resolve()),
        "BOOK_LIBRARY_OBSIDIAN_VAULT": str(vault.resolve()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_generated_config_escapes_del_in_a_valid_macos_path(tmp_path: Path) -> None:
    project_root = tmp_path / "Release\x7fFolder"

    result = install_macos.install(
        project_root=project_root,
        python=project_root / ".venv" / "bin" / "python",
        skip_sync=True,
    )

    parsed = tomllib.loads(result.config.read_text(encoding="utf-8"))
    assert parsed["mcp_servers"]["book_library"]["cwd"] == str(
        project_root.resolve()
    )


def test_skip_sync_does_not_find_uv_or_run_external_commands(tmp_path: Path) -> None:
    def unexpected_find(_: str) -> str | None:
        raise AssertionError("skip-sync must not search PATH")

    def unexpected_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("skip-sync must not run an external command")

    install_macos.install(
        project_root=tmp_path / "Release",
        python=tmp_path / "fake-python",
        skip_sync=True,
        find_executable=unexpected_find,
        run_command=unexpected_run,
    )


def test_sync_prefers_bundled_uv_over_path_uv(tmp_path: Path) -> None:
    project_root = tmp_path / "Release With Spaces"
    bundled_uv = project_root / "bin" / "uv"
    bundled_uv.parent.mkdir(parents=True)
    bundled_uv.write_bytes(b"fake uv")
    bundled_uv.chmod(0o755)
    path_lookups: list[str] = []
    calls: list[tuple[list[str], Path, bool]] = []

    def find_executable(name: str) -> str | None:
        path_lookups.append(name)
        return "/opt/homebrew/bin/uv"

    def run_command(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd, check))
        _create_executable(project_root / ".venv" / "bin" / "python")
        return subprocess.CompletedProcess(command, 0)

    install_macos.install(
        project_root=project_root,
        python=project_root / ".venv" / "bin" / "python",
        find_executable=find_executable,
        run_command=run_command,
    )

    assert path_lookups == []
    assert calls == [
        (
            [
                str(bundled_uv.resolve()),
                "sync",
                "--frozen",
                "--extra",
                "semantic",
                "--extra",
                "ocr",
                "--python",
                "3.12",
            ],
            project_root.resolve(),
            True,
        )
    ]


def test_sync_falls_back_to_uv_on_path(tmp_path: Path) -> None:
    project_root = tmp_path / "Release"
    calls: list[list[str]] = []

    def run_command(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == project_root.resolve()
        assert check is True
        calls.append(command)
        _create_executable(project_root / ".venv" / "bin" / "python")
        return subprocess.CompletedProcess(command, 0)

    install_macos.install(
        project_root=project_root,
        python=project_root / ".venv" / "bin" / "python",
        find_executable=lambda name: "/usr/local/bin/uv" if name == "uv" else None,
        run_command=run_command,
    )

    assert calls == [
        [
            "/usr/local/bin/uv",
            "sync",
            "--frozen",
            "--extra",
            "semantic",
            "--extra",
            "ocr",
            "--python",
            "3.12",
        ]
    ]


def test_sync_forces_project_venv_without_discarding_other_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "Release With Spaces"
    external_environment = tmp_path / "redirected elsewhere"
    observed_environment: list[dict[str, str]] = []
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(external_environment))
    monkeypatch.setenv("BOOK_INSTALLER_TEST_SENTINEL", "keep-me")

    def run_command(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        observed_environment.append(dict(os.environ if env is None else env))
        _create_executable(project_root / ".venv" / "bin" / "python")
        return subprocess.CompletedProcess(command, 0)

    install_macos.install(
        project_root=project_root,
        find_executable=lambda _: "/usr/local/bin/uv",
        run_command=run_command,
    )

    assert observed_environment[0]["UV_PROJECT_ENVIRONMENT"] == str(
        (project_root / ".venv").resolve()
    )
    assert observed_environment[0]["BOOK_INSTALLER_TEST_SENTINEL"] == "keep-me"


def test_rerun_keeps_venv_python_symlink_path_in_config(tmp_path: Path) -> None:
    project_root = tmp_path / "Release With Spaces"
    base_python = tmp_path / "uv managed Python" / "python3.12"
    venv_python = project_root / ".venv" / "bin" / "python"
    _create_executable(base_python)
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    def run_command(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0)

    result = install_macos.install(
        project_root=project_root,
        find_executable=lambda _: "/usr/local/bin/uv",
        run_command=run_command,
    )

    expected_python = Path(os.path.abspath(venv_python))
    parsed = tomllib.loads(result.config.read_text(encoding="utf-8"))
    assert result.python == expected_python
    assert parsed["mcp_servers"]["book_library"]["command"] == str(
        expected_python
    )


@pytest.mark.parametrize(
    "created_mode",
    [None, 0o644],
    ids=["missing", "not-executable"],
)
def test_sync_does_not_publish_config_without_executable_python(
    tmp_path: Path, created_mode: int | None
) -> None:
    project_root = tmp_path / "Release"
    python = project_root / ".venv" / "bin" / "python"

    def run_command(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if created_mode is not None:
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(created_mode)
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(install_macos.InstallError, match="Python.*(不存在|不可执行)"):
        install_macos.install(
            project_root=project_root,
            find_executable=lambda _: "/usr/local/bin/uv",
            run_command=run_command,
        )

    assert not (project_root / ".codex" / "config.toml").exists()


def test_missing_bundled_and_path_uv_is_a_clear_chinese_error(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "Release"

    with pytest.raises(install_macos.InstallError, match="未找到 uv"):
        install_macos.install(
            project_root=project_root,
            python=project_root / ".venv" / "bin" / "python",
            find_executable=lambda _: None,
        )

    assert not (project_root / ".codex" / "config.toml").exists()


def test_main_returns_success_and_prints_next_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project_root = tmp_path / "Release With Spaces"

    exit_code = install_macos.main(
        [
            "--project-root",
            str(project_root),
            "--skip-sync",
            "--python",
            str(project_root / "test python"),
        ]
    )

    output = capsys.readouterr()
    assert exit_code == 0
    assert "重启 Codex" in output.out
    assert "用此项目新建任务" in output.out
    assert "检查书库状态" in output.out
    assert output.err == ""


def test_main_returns_install_failure_exit_code_when_uv_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "Release"
    monkeypatch.setenv("PATH", "")

    exit_code = install_macos.main(["--project-root", str(project_root)])

    output = capsys.readouterr()
    assert exit_code == install_macos.EXIT_INSTALL_ERROR
    assert output.out == ""
    assert "未找到 uv" in output.err
    assert "bin/uv" in output.err
