from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "native" / "book_vision_ocr" / "main.swift"
DEFAULT_OUTPUT = PROJECT_ROOT / "bin" / "book-vision-ocr"
TARGET = "arm64-apple-macos13.0"
REQUIRED_LANGUAGES = frozenset({"zh-Hans", "en-US"})
MACHO_64_MAGICS = frozenset({b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"})


class VisionHelperBuildError(ValueError):
    """The native Apple Vision helper could not be built safely."""


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _default_run_command(argv: list[str]) -> subprocess.CompletedProcess[str]:
    module_cache = (
        Path(tempfile.gettempdir())
        / f"book-vision-module-cache-{os.getuid()}"
    )
    if module_cache.is_symlink():
        raise OSError(f"module cache must not be a symlink: {module_cache}")
    module_cache.mkdir(mode=0o700, parents=False, exist_ok=True)
    if not module_cache.is_dir():
        raise OSError(f"module cache is not a directory: {module_cache}")
    module_cache.chmod(0o700)
    environment = os.environ.copy()
    environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
    environment["SWIFT_MODULECACHE_PATH"] = str(module_cache)
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        env=environment,
    )


def _validate_runner_result(
    result: object,
    *,
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    if not isinstance(result, subprocess.CompletedProcess):
        raise VisionHelperBuildError(
            f"command runner returned an invalid result for {command[0]}"
        )
    if type(result.returncode) is not int:
        raise VisionHelperBuildError(
            f"command runner returned an invalid exit code for {command[0]}"
        )
    if type(result.stdout) is not str or type(result.stderr) is not str:
        raise VisionHelperBuildError(
            f"command runner returned non-text output for {command[0]}"
        )
    return result


def _run_checked(
    command: list[str],
    *,
    run_command: CommandRunner,
) -> subprocess.CompletedProcess[str]:
    if type(command) is not list or not command or not all(
        type(argument) is str and argument for argument in command
    ):
        raise VisionHelperBuildError("internal command arguments are invalid")
    try:
        raw_result = run_command(list(command))
    except (OSError, subprocess.SubprocessError) as exc:
        raise VisionHelperBuildError(
            f"failed to run {command[0]}: {exc}"
        ) from exc
    result = _validate_runner_result(raw_result, command=command)
    if result.returncode != 0:
        diagnostic = (
            result.stderr.strip()
            or result.stdout.strip()
            or "no diagnostic output"
        )
        display_command = " ".join(command[:2])
        raise VisionHelperBuildError(
            f"{display_command} failed with exit code {result.returncode}: {diagnostic}"
        )
    return result


def _validate_source(source: object) -> Path:
    if not isinstance(source, Path):
        raise TypeError("source must be a pathlib.Path")
    if source.is_symlink():
        raise VisionHelperBuildError("source must not be a symlink")
    if not source.is_file():
        raise VisionHelperBuildError(f"source is not a regular file: {source}")
    return source.resolve(strict=True)


def _absolute_output(output: object) -> Path:
    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")
    return Path(os.path.abspath(os.fspath(output)))


def _validate_existing_output(output: Path) -> None:
    if output.is_symlink():
        raise VisionHelperBuildError(f"output must not be a symlink: {output}")
    if output.exists() and not output.is_file():
        raise VisionHelperBuildError(f"output must be a regular file: {output}")


def _validate_compiled_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise VisionHelperBuildError(
            "swiftc did not create the requested helper output"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise VisionHelperBuildError(
            "swiftc output must be a regular file, not a symlink"
        )
    try:
        with path.open("rb") as compiled:
            magic = compiled.read(4)
    except OSError as exc:
        raise VisionHelperBuildError(
            f"could not inspect swiftc output: {exc}"
        ) from exc
    if magic not in MACHO_64_MAGICS:
        raise VisionHelperBuildError("swiftc output is not a 64-bit Mach-O executable")


def _reject_nonfinite_json(value: str) -> object:
    raise VisionHelperBuildError(
        f"capabilities must contain only finite JSON values, got {value}"
    )


def _validate_capabilities(stdout: str) -> None:
    try:
        payload = json.loads(stdout, parse_constant=_reject_nonfinite_json)
    except VisionHelperBuildError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise VisionHelperBuildError(f"invalid capabilities JSON: {exc}") from exc
    if type(payload) is not dict:
        raise VisionHelperBuildError("capabilities JSON must be an object")
    if set(payload) != {"schema_version", "languages"}:
        raise VisionHelperBuildError(
            "capabilities JSON has missing or extra fields"
        )
    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise VisionHelperBuildError(
            "capabilities schema_version must be integer 1"
        )
    languages = payload["languages"]
    if type(languages) is not list or not all(
        type(language) is str and language for language in languages
    ):
        raise VisionHelperBuildError(
            "capabilities languages must be nonblank strings"
        )
    if len(set(languages)) != len(languages):
        raise VisionHelperBuildError(
            "capabilities languages must not contain duplicates"
        )
    missing = sorted(REQUIRED_LANGUAGES.difference(languages))
    if missing:
        raise VisionHelperBuildError(
            f"capabilities missing required language {missing[0]}"
        )


def build_vision_helper(
    *,
    source: Path,
    output: Path,
    run_command: CommandRunner = _default_run_command,
) -> Path:
    """Build, sign, validate, and atomically install the native Vision helper."""

    if not callable(run_command):
        raise TypeError("run_command must be callable")
    source_path = _validate_source(source)
    output_path = _absolute_output(output)
    _validate_existing_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.parent.is_symlink() or not output_path.parent.is_dir():
        raise VisionHelperBuildError("output parent must be a real directory")

    with tempfile.TemporaryDirectory(
        prefix=".book-vision-ocr-build-",
        dir=output_path.parent,
    ) as temporary_directory:
        temporary_output = Path(temporary_directory) / output_path.name
        _run_checked(
            [
                "xcrun",
                "swiftc",
                "-O",
                "-target",
                TARGET,
                str(source_path),
                "-o",
                str(temporary_output),
            ],
            run_command=run_command,
        )
        _validate_compiled_file(temporary_output)
        temporary_output.chmod(0o755)

        _run_checked(
            [
                "codesign",
                "--force",
                "--sign",
                "-",
                str(temporary_output),
            ],
            run_command=run_command,
        )
        architectures = _run_checked(
            ["lipo", "-archs", str(temporary_output)],
            run_command=run_command,
        ).stdout.split()
        if architectures != ["arm64"]:
            rendered = " ".join(architectures) or "none"
            raise VisionHelperBuildError(
                f"helper must be a thin arm64 binary; lipo reported: {rendered}"
            )
        _run_checked(
            ["codesign", "--verify", "--strict", str(temporary_output)],
            run_command=run_command,
        )
        capabilities = _run_checked(
            [str(temporary_output), "--capabilities"],
            run_command=run_command,
        )
        if capabilities.stderr:
            raise VisionHelperBuildError(
                "capabilities command wrote unexpected stderr output"
            )
        _validate_capabilities(capabilities.stdout)
        _validate_compiled_file(temporary_output)
        if stat.S_IMODE(temporary_output.stat().st_mode) != 0o755:
            raise VisionHelperBuildError("helper executable mode must be exactly 0755")

        _validate_existing_output(output_path)
        os.replace(temporary_output, output_path)

    return output_path.resolve(strict=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the native Apple Vision OCR helper."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        output = build_vision_helper(
            source=arguments.source,
            output=arguments.output,
        )
    except (OSError, VisionHelperBuildError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
