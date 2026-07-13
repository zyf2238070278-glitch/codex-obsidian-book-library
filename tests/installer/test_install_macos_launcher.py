from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_LAUNCHER = PROJECT_ROOT / "install-macos.command"


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _copy_launcher(tmp_path: Path) -> Path:
    release = tmp_path / "Book Library Release With Spaces"
    launcher = release / "install-macos.command"
    launcher.parent.mkdir(parents=True)
    shutil.copy2(SOURCE_LAUNCHER, launcher)
    launcher.chmod(0o755)
    (release / "installer").mkdir()
    (release / "installer" / "install_macos.py").write_text(
        "raise AssertionError('fake launcher test installer must not execute')\n",
        encoding="utf-8",
    )
    return launcher


def _capture_script() -> str:
    return """#!/bin/sh
: > "$CAPTURE_FILE"
for argument in "$@"; do
    printf '%s\\n' "$argument" >> "$CAPTURE_FILE"
done
exit "${FAKE_EXIT_CODE:-0}"
"""


def _read_arguments(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_launcher_prefers_python3_and_preserves_space_containing_paths(
    tmp_path: Path,
) -> None:
    launcher = _copy_launcher(tmp_path)
    release = launcher.parent
    fake_bin = tmp_path / "fake bin"
    python_capture = tmp_path / "python arguments.txt"
    uv_capture = tmp_path / "uv arguments.txt"
    _write_executable(fake_bin / "python3", _capture_script())
    _write_executable(release / "bin" / "uv", _capture_script())
    env = {
        **os.environ,
        "PATH": str(fake_bin),
        "CAPTURE_FILE": str(python_capture),
    }

    completed = subprocess.run(
        [str(launcher), "--skip-sync"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0
    assert _read_arguments(python_capture) == [
        str(release / "installer" / "install_macos.py"),
        "--project-root",
        str(release),
        "--skip-sync",
    ]
    assert not uv_capture.exists()


def test_launcher_uses_bundled_uv_when_python3_is_unavailable(
    tmp_path: Path,
) -> None:
    launcher = _copy_launcher(tmp_path)
    release = launcher.parent
    empty_path = tmp_path / "empty path"
    empty_path.mkdir()
    uv_capture = tmp_path / "uv arguments.txt"
    _write_executable(release / "bin" / "uv", _capture_script())
    env = {
        **os.environ,
        "PATH": str(empty_path),
        "CAPTURE_FILE": str(uv_capture),
    }

    completed = subprocess.run(
        [str(launcher), "--vault", str(tmp_path / "Vault With Spaces")],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0
    assert _read_arguments(uv_capture) == [
        "run",
        "--no-project",
        "--python",
        "3.12",
        str(release / "installer" / "install_macos.py"),
        "--project-root",
        str(release),
        "--vault",
        str(tmp_path / "Vault With Spaces"),
    ]


def test_launcher_exits_with_chinese_error_without_python_or_bundled_uv(
    tmp_path: Path,
) -> None:
    launcher = _copy_launcher(tmp_path)
    empty_path = tmp_path / "empty path"
    empty_path.mkdir()
    env = {**os.environ, "PATH": str(empty_path)}

    completed = subprocess.run(
        [str(launcher)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "未找到 python3" in completed.stderr
    assert "bin/uv" in completed.stderr
    assert "Apple Silicon" in completed.stderr


def test_launcher_preserves_child_failure_status_without_blocking_off_tty(
    tmp_path: Path,
) -> None:
    launcher = _copy_launcher(tmp_path)
    fake_bin = tmp_path / "fake bin"
    capture = tmp_path / "python arguments.txt"
    _write_executable(fake_bin / "python3", _capture_script())
    env = {
        **os.environ,
        "PATH": str(fake_bin),
        "CAPTURE_FILE": str(capture),
        "FAKE_EXIT_CODE": "23",
    }

    completed = subprocess.run(
        [str(launcher)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 23
    assert "按回车关闭" not in completed.stdout


def test_launcher_only_prompts_to_close_for_an_interactive_terminal() -> None:
    launcher = SOURCE_LAUNCHER.read_text(encoding="utf-8")

    assert '[ -t 0 ] && [ -t 1 ]' in launcher
    assert "按回车关闭" in launcher
    assert 'exit "$status"' in launcher
