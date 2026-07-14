from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "native" / "book_vision_ocr" / "main.swift"
DEFAULT_OUTPUT = PROJECT_ROOT / "bin" / "book-vision-ocr"
TARGET = "arm64-apple-macos13.0"
REQUIRED_LANGUAGES = frozenset({"zh-Hans", "en-US"})
MACHO_64_MAGICS = frozenset({b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"})
XCRUN = "/usr/bin/xcrun"
CODESIGN = "/usr/bin/codesign"
LIPO = "/usr/bin/lipo"
SAFE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
COMPILE_TIMEOUT_SECONDS = 180.0
TOOL_TIMEOUT_SECONDS = 30.0
HELPER_TIMEOUT_SECONDS = 30.0
MAXIMUM_COMMAND_OUTPUT_BYTES = 64 * 1024
MAXIMUM_DIAGNOSTIC_BYTES = 4 * 1024
MAXIMUM_HELPER_BYTES = 256 * 1024 * 1024
OUTPUT_LIMIT_EXIT_CODE = 125
PIPE_READ_CHUNK_BYTES = 16 * 1024


class VisionHelperBuildError(ValueError):
    """The native Apple Vision helper could not be built safely."""


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: list[str],
        *,
        environment: Mapping[str, str],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class _ArtifactFingerprint:
    device: int
    inode: int
    size: int
    mode: int
    modified_ns: int
    sha256: str


def _render_output(value: bytearray, *, truncated: bool) -> str:
    rendered = bytes(value).decode("utf-8", errors="replace")
    if truncated:
        rendered += "\n...[truncated]"
    return rendered


def _truncate_text(value: str, maximum_bytes: int) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= maximum_bytes:
        return value
    return (
        encoded[:maximum_bytes].decode("utf-8", errors="ignore")
        + "\n...[truncated]"
    )


def _default_run_command(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=dict(environment),
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated_stream: str | None = None
    deadline = time.monotonic() + timeout

    for name, stream in streams.items():
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)

    def terminate_process_group() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_process_group()
                raise subprocess.TimeoutExpired(argv, timeout)
            events = selector.select(timeout=min(remaining, 0.1))
            for key, _ in events:
                try:
                    chunk = os.read(key.fd, PIPE_READ_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                captured = sum(len(buffer) for buffer in buffers.values())
                available = MAXIMUM_COMMAND_OUTPUT_BYTES - captured
                if len(chunk) > available:
                    if available > 0:
                        buffers[key.data].extend(chunk[:available])
                    truncated_stream = key.data
                    terminate_process_group()
                    break
                buffers[key.data].extend(chunk)
            if truncated_stream is not None:
                break

        if truncated_stream is not None:
            return subprocess.CompletedProcess(
                argv,
                OUTPUT_LIMIT_EXIT_CODE,
                _render_output(
                    buffers["stdout"],
                    truncated=truncated_stream == "stdout",
                ),
                _render_output(
                    buffers["stderr"],
                    truncated=truncated_stream == "stderr",
                ),
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            terminate_process_group()
            raise subprocess.TimeoutExpired(argv, timeout)
        return_code = process.wait(timeout=remaining)
        return subprocess.CompletedProcess(
            argv,
            return_code,
            _render_output(buffers["stdout"], truncated=False),
            _render_output(buffers["stderr"], truncated=False),
        )
    except subprocess.TimeoutExpired:
        terminate_process_group()
        raise
    finally:
        selector.close()
        for stream in streams.values():
            stream.close()


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
    return subprocess.CompletedProcess(
        command,
        result.returncode,
        _truncate_text(result.stdout, MAXIMUM_COMMAND_OUTPUT_BYTES),
        _truncate_text(result.stderr, MAXIMUM_COMMAND_OUTPUT_BYTES),
    )


def _run_checked(
    command: list[str],
    *,
    run_command: CommandRunner,
    environment: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    if type(command) is not list or not command or not all(
        type(argument) is str and argument for argument in command
    ):
        raise VisionHelperBuildError("internal command arguments are invalid")
    try:
        raw_result = run_command(
            list(command),
            environment=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise VisionHelperBuildError(
            f"{command[0]} timed out after {timeout:g} seconds"
        ) from exc
    except (OSError, subprocess.SubprocessError, TypeError) as exc:
        raise VisionHelperBuildError(
            f"failed to run {command[0]}: {exc}"
        ) from exc
    result = _validate_runner_result(raw_result, command=command)
    if result.returncode != 0:
        diagnostic = _truncate_text(
            result.stderr.strip()
            or result.stdout.strip()
            or "no diagnostic output",
            MAXIMUM_DIAGNOSTIC_BYTES,
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


def _open_compiled_file(path: Path) -> int:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except FileNotFoundError as exc:
        raise VisionHelperBuildError(
            "swiftc did not create the requested helper output"
        ) from exc
    except OSError as exc:
        raise VisionHelperBuildError(
            "swiftc output must be a regular file, not a symlink"
        ) from exc
    metadata = os.fstat(descriptor)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise VisionHelperBuildError(
            "swiftc output must be a regular file, not a symlink"
        )
    if metadata.st_size <= 0 or metadata.st_size > MAXIMUM_HELPER_BYTES:
        os.close(descriptor)
        raise VisionHelperBuildError("swiftc output has an unsafe file size")
    return descriptor


def _validate_compiled_file(path: Path) -> None:
    descriptor = _open_compiled_file(path)
    try:
        magic = os.read(descriptor, 4)
    except OSError as exc:
        raise VisionHelperBuildError(
            f"could not inspect swiftc output: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    if magic not in MACHO_64_MAGICS:
        raise VisionHelperBuildError("swiftc output is not a 64-bit Mach-O executable")


def _artifact_fingerprint(path: Path) -> _ArtifactFingerprint:
    descriptor = _open_compiled_file(path)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            stat.S_IMODE(before.st_mode),
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            stat.S_IMODE(after.st_mode),
            after.st_mtime_ns,
        ):
            raise VisionHelperBuildError(
                "helper changed while its fingerprint was calculated"
            )
        return _ArtifactFingerprint(
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mode=stat.S_IMODE(after.st_mode),
            modified_ns=after.st_mtime_ns,
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def _validate_architecture(
    path: Path,
    *,
    run_command: CommandRunner,
    environment: Mapping[str, str],
) -> None:
    architectures = _run_checked(
        [LIPO, "-archs", str(path)],
        run_command=run_command,
        environment=environment,
        timeout=TOOL_TIMEOUT_SECONDS,
    ).stdout.split()
    if architectures != ["arm64"]:
        rendered = " ".join(architectures) or "none"
        raise VisionHelperBuildError(
            f"helper must be a thin arm64 binary; lipo reported: {rendered}"
        )


def _verify_signature(
    path: Path,
    *,
    run_command: CommandRunner,
    environment: Mapping[str, str],
) -> None:
    _run_checked(
        [CODESIGN, "--verify", "--strict", str(path)],
        run_command=run_command,
        environment=environment,
        timeout=TOOL_TIMEOUT_SECONDS,
    )


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
    text_budget_source = _validate_source(source_path.with_name("TextBudget.swift"))
    output_path = _absolute_output(output)
    _validate_existing_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.parent.is_symlink() or not output_path.parent.is_dir():
        raise VisionHelperBuildError("output parent must be a real directory")

    with tempfile.TemporaryDirectory(
        prefix=".book-vision-ocr-build-",
        dir=output_path.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_output = temporary_root / output_path.name
        module_cache = temporary_root / "module-cache"
        module_cache.mkdir(mode=0o700)
        # Intentionally omit inherited DEVELOPER_DIR, SDKROOT, and DYLD_*.
        # Absolute /usr/bin/xcrun therefore uses the administrator-selected
        # Xcode toolchain, while every compiler cache stays private to this build.
        environment = {
            "PATH": SAFE_PATH,
            "LANG": "en_US.UTF-8",
            "TMPDIR": str(temporary_root),
            "CLANG_MODULE_CACHE_PATH": str(module_cache),
            "SWIFT_MODULECACHE_PATH": str(module_cache),
        }
        _run_checked(
            [
                XCRUN,
                "swiftc",
                "-O",
                "-target",
                TARGET,
                str(text_budget_source),
                str(source_path),
                "-o",
                str(temporary_output),
            ],
            run_command=run_command,
            environment=environment,
            timeout=COMPILE_TIMEOUT_SECONDS,
        )
        _validate_compiled_file(temporary_output)
        temporary_output.chmod(0o755)

        _run_checked(
            [
                CODESIGN,
                "--force",
                "--sign",
                "-",
                str(temporary_output),
            ],
            run_command=run_command,
            environment=environment,
            timeout=TOOL_TIMEOUT_SECONDS,
        )
        _validate_architecture(
            temporary_output,
            run_command=run_command,
            environment=environment,
        )
        _verify_signature(
            temporary_output,
            run_command=run_command,
            environment=environment,
        )
        fingerprint_before_capabilities = _artifact_fingerprint(temporary_output)
        capabilities = _run_checked(
            [str(temporary_output), "--capabilities"],
            run_command=run_command,
            environment=environment,
            timeout=HELPER_TIMEOUT_SECONDS,
        )
        if capabilities.stderr:
            raise VisionHelperBuildError(
                "capabilities command wrote unexpected stderr output"
            )
        _validate_capabilities(capabilities.stdout)
        fingerprint_after_capabilities = _artifact_fingerprint(temporary_output)
        if fingerprint_after_capabilities != fingerprint_before_capabilities:
            raise VisionHelperBuildError(
                "helper changed during capabilities validation"
            )
        _validate_architecture(
            temporary_output,
            run_command=run_command,
            environment=environment,
        )
        _verify_signature(
            temporary_output,
            run_command=run_command,
            environment=environment,
        )
        if _artifact_fingerprint(temporary_output) != fingerprint_before_capabilities:
            raise VisionHelperBuildError(
                "helper changed during final architecture/signature verification"
            )
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
