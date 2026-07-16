from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_LAUNCHER = PROJECT_ROOT / "install-macos.command"
PINNED_UV_SHA256 = "c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554"


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


def _pin_fixture_digest(launcher: Path, executable: Path) -> None:
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    body = launcher.read_text(encoding="utf-8")
    assert PINNED_UV_SHA256 in body
    launcher.write_text(body.replace(PINNED_UV_SHA256, digest), encoding="utf-8")


def _substitute_platform_commands(
    launcher: Path, tmp_path: Path, *, system: str, machine: str, version: str
) -> None:
    fake_uname = tmp_path / "fixture commands" / "uname"
    fake_sw_vers = tmp_path / "fixture commands" / "sw_vers"
    _write_executable(
        fake_uname,
        f"#!/bin/sh\n[ \"$1\" = -s ] && printf '%s\\n' {system!r} || printf '%s\\n' {machine!r}\n",
    )
    _write_executable(fake_sw_vers, f"#!/bin/sh\nprintf '%s\\n' {version!r}\n")
    body = launcher.read_text(encoding="utf-8")
    body = body.replace("/usr/bin/uname", f'"{fake_uname}"')
    body = body.replace("/usr/bin/sw_vers", f'"{fake_sw_vers}"')
    launcher.write_text(body, encoding="utf-8")


def test_launcher_always_uses_project_uv_and_fixed_python(tmp_path: Path) -> None:
    launcher = _copy_launcher(tmp_path)
    release = launcher.parent
    capture = tmp_path / "uv arguments.txt"
    _write_executable(release / "bin" / "uv", _capture_script())
    # For this fake fixture, rewrite only the copied launcher's pinned digest
    # to the fake executable digest; never change production digest.
    _pin_fixture_digest(launcher, release / "bin" / "uv")
    completed = subprocess.run(
        [
            str(launcher),
            "--vault",
            str(tmp_path / "Vault With Spaces"),
            "--codex-config",
            str(tmp_path / "Config With Spaces" / "config.toml"),
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "CAPTURE_FILE": str(capture)},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert _read_arguments(capture) == [
        "run",
        "--no-project",
        "--python",
        "3.12",
        str(release / "installer" / "install_macos.py"),
        "--project-root",
        str(release),
        "--vault",
        str(tmp_path / "Vault With Spaces"),
        "--codex-config",
        str(tmp_path / "Config With Spaces" / "config.toml"),
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--project-root", "/private/tmp/attacker"],
        ["--python", "/private/tmp/attacker-python"],
        ["--vault", "first", "--vault", "second"],
        ["--codex-config", "first", "--codex-config", "second"],
        ["--vault"],
        ["--codex-config"],
        ["--vault", "--codex-config", "config.toml"],
        ["--vault", "-x"],
        ["--unknown", "value"],
        ["--vault=/private/tmp/vault"],
        ["positional"],
        ["--codex-config", "{CONFIG}", "--python", "{ATTACKER}"],
    ],
)
def test_launcher_rejects_unsafe_arguments_before_executing_uv(
    tmp_path: Path, arguments: list[str]
) -> None:
    launcher = _copy_launcher(tmp_path)
    uv_capture = tmp_path / "uv must not execute.txt"
    attacker_capture = tmp_path / "attacker python must not execute.txt"
    config = tmp_path / "must not write" / "config.toml"
    _write_executable(launcher.parent / "bin" / "uv", _capture_script())
    _pin_fixture_digest(launcher, launcher.parent / "bin" / "uv")
    _write_executable(
        tmp_path / "attacker-python",
        f"#!/bin/sh\ntouch {str(attacker_capture)!r}\n",
    )
    fixture_values = {
        "{CONFIG}": str(config),
        "{ATTACKER}": str(tmp_path / "attacker-python"),
    }
    arguments = [fixture_values.get(argument, argument) for argument in arguments]

    completed = subprocess.run(
        [str(launcher), *arguments],
        text=True,
        capture_output=True,
        env={**os.environ, "CAPTURE_FILE": str(uv_capture)},
        check=False,
    )

    assert completed.returncode != 0
    assert "参数" in completed.stderr
    assert not uv_capture.exists()
    assert not attacker_capture.exists()
    assert not config.exists()


def test_launcher_does_not_use_python_or_path_uv(tmp_path: Path) -> None:
    launcher = _copy_launcher(tmp_path)
    release = launcher.parent
    capture = tmp_path / "bundled uv.txt"
    path_capture = tmp_path / "path command.txt"
    fake_bin = tmp_path / "fake PATH"
    _write_executable(release / "bin" / "uv", _capture_script())
    _pin_fixture_digest(launcher, release / "bin" / "uv")
    _write_executable(fake_bin / "uv", f"#!/bin/sh\ntouch {str(path_capture)!r}\nexit 91\n")
    _write_executable(fake_bin / "python3", f"#!/bin/sh\ntouch {str(path_capture)!r}\nexit 92\n")

    completed = subprocess.run(
        [str(launcher)],
        text=True,
        capture_output=True,
        env={**os.environ, "PATH": str(fake_bin), "CAPTURE_FILE": str(capture)},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert capture.is_file()
    assert not path_capture.exists()


@pytest.mark.parametrize(
    ("system", "machine", "version", "message"),
    [
        ("Linux", "arm64", "16.0", "仅支持 macOS"),
        ("Darwin", "x86_64", "16.0", "Apple 芯片"),
        ("Darwin", "arm64", "15.9", "macOS 16"),
        ("Darwin", "arm64", "not-a-version", "无法识别 macOS 版本"),
    ],
)
def test_launcher_rejects_unsupported_platform_before_uv(
    tmp_path: Path, system: str, machine: str, version: str, message: str
) -> None:
    launcher = _copy_launcher(tmp_path)
    capture = tmp_path / "must not run.txt"
    _write_executable(launcher.parent / "bin" / "uv", _capture_script())
    _pin_fixture_digest(launcher, launcher.parent / "bin" / "uv")
    _substitute_platform_commands(
        launcher, tmp_path, system=system, machine=machine, version=version
    )

    completed = subprocess.run(
        [str(launcher)],
        text=True,
        capture_output=True,
        env={**os.environ, "CAPTURE_FILE": str(capture)},
        check=False,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not capture.exists()


@pytest.mark.parametrize("state", ["missing", "not-executable", "tampered"])
def test_launcher_rejects_untrusted_bundled_uv_before_execution(
    tmp_path: Path, state: str
) -> None:
    launcher = _copy_launcher(tmp_path)
    uv = launcher.parent / "bin" / "uv"
    capture = tmp_path / "must not run.txt"
    if state != "missing":
        _write_executable(uv, _capture_script())
        if state == "not-executable":
            uv.chmod(0o644)
        elif state == "tampered":
            uv.write_text(_capture_script() + "# changed\n", encoding="utf-8")

    completed = subprocess.run(
        [str(launcher)],
        text=True,
        capture_output=True,
        env={**os.environ, "CAPTURE_FILE": str(capture)},
        check=False,
    )

    assert completed.returncode != 0
    assert "bin/uv" in completed.stderr
    assert ("校验" in completed.stderr) if state == "tampered" else ("可执行" in completed.stderr)
    assert not capture.exists()


def test_launcher_preserves_child_failure_status_without_blocking_off_tty(
    tmp_path: Path,
) -> None:
    launcher = _copy_launcher(tmp_path)
    capture = tmp_path / "uv arguments.txt"
    _write_executable(launcher.parent / "bin" / "uv", _capture_script())
    _pin_fixture_digest(launcher, launcher.parent / "bin" / "uv")

    completed = subprocess.run(
        [str(launcher)],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "CAPTURE_FILE": str(capture),
            "FAKE_EXIT_CODE": "23",
        },
        check=False,
    )

    assert completed.returncode == 23
    assert "按回车关闭" not in completed.stdout


def test_launcher_has_absolute_platform_and_integrity_gates() -> None:
    body = SOURCE_LAUNCHER.read_text(encoding="utf-8")

    assert SOURCE_LAUNCHER.stat().st_mode & 0o111
    assert body.startswith("#!/bin/bash\n")
    assert "set -u" in body
    assert "/usr/bin/uname" in body
    assert "/usr/bin/sw_vers" in body
    assert "/usr/bin/shasum -a 256" in body
    assert PINNED_UV_SHA256 in body
    assert '[ -t 0 ] && [ -t 1 ]' in body
    assert "按回车关闭" in body
    assert 'exit "$status"' in body


def test_launcher_contains_no_legacy_bootstrap_logic() -> None:
    body = SOURCE_LAUNCHER.read_text(encoding="utf-8").casefold()

    forbidden = ("python3", " pip ", "homebrew", "brew ", "xcrun", "xcode", "clt")
    assert all(token not in body for token in forbidden)
