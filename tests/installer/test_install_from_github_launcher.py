from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "install-from-github.command"
ARCHIVE = "codex-obsidian-book-library-v0.1.0-beta.1-macos-arm64-all-in-one.zip"
TOP_LEVEL = "codex-obsidian-book-library-v0.1.0-beta.1-macos-arm64"


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _make_fake_commands(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "fake-bin"
    _write_executable(
        fake_bin / "uname",
        """#!/bin/sh
if [ "$1" = "-s" ]; then
  printf '%s\n' Darwin
else
  printf '%s\n' arm64
fi
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/bin/sh
output=
url=
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) shift; output=$1 ;;
    http://*|https://*) url=$1 ;;
  esac
  shift
done
printf '%s\n' "$url" >> "$DOWNLOAD_LOG"
case "$url" in
  */SHA256SUMS) cp "$FIXTURE_DIR/SHA256SUMS" "$output" ;;
  *.zip) cp "$FIXTURE_DIR/release.zip" "$output" ;;
  *) exit 41 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "ditto",
        """#!/bin/sh
destination=$4
mkdir -p "$destination"
cp -R "$FIXTURE_BUNDLE" "$destination/$TOP_LEVEL_NAME"
""",
    )
    return fake_bin


def _make_fixture_bundle(tmp_path: Path) -> tuple[Path, Path]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    archive_bytes = b"fake deterministic release archive"
    (fixture / "release.zip").write_bytes(archive_bytes)
    digest = hashlib.sha256(archive_bytes).hexdigest()
    (fixture / "SHA256SUMS").write_text(
        f"{digest}  {ARCHIVE}\n",
        encoding="utf-8",
    )

    bundle = tmp_path / "fixture-bundle"
    (bundle / "bin").mkdir(parents=True)
    (bundle / "data" / "models").mkdir(parents=True)
    _write_executable(bundle / "bin" / "uv", "#!/bin/sh\nexit 0\n")
    (bundle / "data" / "models" / "model.safetensors").write_bytes(b"model")
    _write_executable(
        bundle / "install-macos.command",
        """#!/bin/sh
printf '%s\n' "$PWD" > "$INSTALL_LOG"
exit 0
""",
    )
    return fixture, bundle


def _run_launcher(tmp_path: Path, *, install_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    fake_bin = _make_fake_commands(tmp_path)
    fixture, bundle = _make_fixture_bundle(tmp_path)
    download_log = tmp_path / "downloads.txt"
    install_log = tmp_path / "install.txt"
    selected_install_dir = install_dir or tmp_path / "installed library"
    temp_root = tmp_path / "temporary files"
    temp_root.mkdir()
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "TMPDIR": str(temp_root),
        "BOOK_LIBRARY_REPOSITORY": "example-owner/example-repo",
        "BOOK_LIBRARY_INSTALL_DIR": str(selected_install_dir),
        "DOWNLOAD_LOG": str(download_log),
        "INSTALL_LOG": str(install_log),
        "FIXTURE_DIR": str(fixture),
        "FIXTURE_BUNDLE": str(bundle),
        "TOP_LEVEL_NAME": TOP_LEVEL,
    }
    completed = subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    completed.download_log = download_log  # type: ignore[attr-defined]
    completed.install_log = install_log  # type: ignore[attr-defined]
    completed.install_dir = selected_install_dir  # type: ignore[attr-defined]
    return completed


def test_launcher_downloads_pinned_release_verifies_and_installs(tmp_path: Path) -> None:
    completed = _run_launcher(tmp_path)

    assert completed.returncode == 0, completed.stderr
    urls = completed.download_log.read_text(encoding="utf-8").splitlines()  # type: ignore[attr-defined]
    assert urls == [
        "https://github.com/example-owner/example-repo/releases/download/v0.1.0-beta.1/SHA256SUMS",
        f"https://github.com/example-owner/example-repo/releases/download/v0.1.0-beta.1/{ARCHIVE}",
    ]
    install_dir = completed.install_dir  # type: ignore[attr-defined]
    assert (install_dir / "install-macos.command").is_file()
    assert completed.install_log.read_text(encoding="utf-8").strip() == str(install_dir)  # type: ignore[attr-defined]
    assert "安装完成" in completed.stdout
    assert "打开并信任" in completed.stdout


def test_launcher_rejects_bad_checksum_before_installing(tmp_path: Path) -> None:
    fake_bin = _make_fake_commands(tmp_path)
    fixture, bundle = _make_fixture_bundle(tmp_path)
    (fixture / "SHA256SUMS").write_text(
        f"{'0' * 64}  {ARCHIVE}\n",
        encoding="utf-8",
    )
    install_dir = tmp_path / "must-not-exist"
    temp_root = tmp_path / "temporary"
    temp_root.mkdir()
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "TMPDIR": str(temp_root),
        "BOOK_LIBRARY_REPOSITORY": "example-owner/example-repo",
        "BOOK_LIBRARY_INSTALL_DIR": str(install_dir),
        "DOWNLOAD_LOG": str(tmp_path / "downloads.txt"),
        "INSTALL_LOG": str(tmp_path / "install.txt"),
        "FIXTURE_DIR": str(fixture),
        "FIXTURE_BUNDLE": str(bundle),
        "TOP_LEVEL_NAME": TOP_LEVEL,
    }

    completed = subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "SHA-256" in completed.stderr
    assert not install_dir.exists()
    assert not Path(env["INSTALL_LOG"]).exists()


def test_launcher_reuses_complete_existing_install_without_downloading(tmp_path: Path) -> None:
    install_dir = tmp_path / "existing library"
    (install_dir / "bin").mkdir(parents=True)
    (install_dir / "data" / "models").mkdir(parents=True)
    _write_executable(install_dir / "bin" / "uv", "#!/bin/sh\nexit 0\n")
    (install_dir / "data" / "models" / "model.safetensors").write_bytes(b"model")
    install_log = tmp_path / "existing-install.txt"
    _write_executable(
        install_dir / "install-macos.command",
        """#!/bin/sh
printf '%s\n' "$PWD" > "$INSTALL_LOG"
exit 0
""",
    )
    fake_bin = tmp_path / "fake-bin"
    _write_executable(
        fake_bin / "uname",
        "#!/bin/sh\n[ \"$1\" = -s ] && echo Darwin || echo arm64\n",
    )
    _write_executable(fake_bin / "curl", "#!/bin/sh\nexit 99\n")
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "BOOK_LIBRARY_REPOSITORY": "example-owner/example-repo",
        "BOOK_LIBRARY_INSTALL_DIR": str(install_dir),
        "INSTALL_LOG": str(install_log),
    }

    completed = subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert install_log.read_text(encoding="utf-8").strip() == str(install_dir)
    assert "已有完整安装" in completed.stdout


def test_launcher_is_a_safe_executable_shell_entrypoint() -> None:
    body = LAUNCHER.read_text(encoding="utf-8")

    assert LAUNCHER.stat().st_mode & 0o111
    assert "set -eu" in body
    assert "mktemp -d" in body
    assert "trap" in body
    assert "eval " not in body
    assert "curl" in body
    assert "| sh" not in body
