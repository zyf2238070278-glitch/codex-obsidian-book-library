from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_WRAPPER = PROJECT_ROOT / "install-from-github.command"


def _copy_wrapper(tmp_path: Path) -> Path:
    release = tmp_path / "Compatibility Wrapper With Spaces"
    release.mkdir()
    wrapper = release / "install-from-github.command"
    shutil.copy2(SOURCE_WRAPPER, wrapper)
    wrapper.chmod(0o755)
    return wrapper


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_wrapper_execs_sibling_launcher_with_exact_arguments(tmp_path: Path) -> None:
    wrapper = _copy_wrapper(tmp_path)
    capture = tmp_path / "arguments.txt"
    _write_executable(
        wrapper.parent / "install-macos.command",
        """#!/bin/sh
: > "$CAPTURE_FILE"
printf '%s\n' "$0" >> "$CAPTURE_FILE"
for argument in "$@"; do printf '%s\n' "$argument" >> "$CAPTURE_FILE"; done
exit "${FAKE_EXIT_CODE:-0}"
""",
    )

    completed = subprocess.run(
        [str(wrapper), "--vault", str(tmp_path / "Vault With Spaces")],
        text=True,
        capture_output=True,
        env={**os.environ, "CAPTURE_FILE": str(capture)},
        check=False,
    )

    assert completed.returncode == 0
    assert capture.read_text(encoding="utf-8").splitlines() == [
        str(wrapper.parent / "install-macos.command"),
        "--vault",
        str(tmp_path / "Vault With Spaces"),
    ]


def test_wrapper_preserves_sibling_launcher_exit_code(tmp_path: Path) -> None:
    wrapper = _copy_wrapper(tmp_path)
    capture = tmp_path / "arguments.txt"
    _write_executable(
        wrapper.parent / "install-macos.command",
        "#!/bin/sh\nexit \"${FAKE_EXIT_CODE:-0}\"\n",
    )

    completed = subprocess.run(
        [str(wrapper)],
        env={**os.environ, "CAPTURE_FILE": str(capture), "FAKE_EXIT_CODE": "29"},
        check=False,
    )

    assert completed.returncode == 29


def test_wrapper_is_only_a_safe_compatibility_exec() -> None:
    body = SOURCE_WRAPPER.read_text(encoding="utf-8")
    lowered = body.casefold()

    assert SOURCE_WRAPPER.stat().st_mode & 0o111
    assert body.startswith("#!/bin/bash\n")
    assert "set -u" in body
    assert 'pwd -P' in body
    assert 'exec "$PROJECT_ROOT/install-macos.command" "$@"' in body
    for forbidden in ("tag=", "github.com", ".zip", "curl", "ditto"):
        assert forbidden not in lowered
