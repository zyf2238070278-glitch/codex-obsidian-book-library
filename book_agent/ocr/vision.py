from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import fitz

from book_agent.ocr.models import (
    BoundingBox,
    OCR_SCHEMA_VERSION,
    VisionLine,
    VisionPageResult,
)


TARGET_DPI = 300.0
MAXIMUM_LONG_EDGE_PIXELS = 12_000
MAXIMUM_PAGE_PIXELS = 20_000_000
HELPER_TIMEOUT_SECONDS = 120.0
MAXIMUM_STDOUT_BYTES = 1024 * 1024
MAXIMUM_STDERR_BYTES = 16 * 1024
MAXIMUM_DIAGNOSTIC_BYTES = 4 * 1024
MAXIMUM_HELPER_BYTES = 256 * 1024 * 1024
MAXIMUM_TEXT_UTF8_BYTES = 400_000
MAXIMUM_TEXT_UNICODE_SCALARS = 100_000
SAFE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
CODESIGN = "/usr/bin/codesign"
_CODESIGN_TIMEOUT_SECONDS = 30.0
_LANGUAGES = "zh-Hans,en-US"
_PIPE_CHUNK_BYTES = 16 * 1024
_BOUNDING_BOX_EPSILON = 1e-6
_READING_ORDER_VERTICAL_TOLERANCE = 0.0125


class VisionOcrError(ValueError):
    """A PDF page could not be rendered or recognized safely."""


@dataclass(frozen=True)
class RunRequest:
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    cwd: Path
    timeout: float
    pass_fds: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.argv) is not tuple or not self.argv or not all(
            type(value) is str and value for value in self.argv
        ):
            raise ValueError("argv must be a nonempty tuple of strings")
        if type(self.environment) is not tuple or not all(
            type(pair) is tuple
            and len(pair) == 2
            and all(type(value) is str and value for value in pair)
            for pair in self.environment
        ):
            raise ValueError("environment must contain string pairs")
        if len({key for key, _ in self.environment}) != len(self.environment):
            raise ValueError("environment keys must be unique")
        if not isinstance(self.cwd, Path) or not self.cwd.is_absolute():
            raise ValueError("cwd must be an absolute pathlib.Path")
        if (
            not isinstance(self.timeout, (int, float))
            or isinstance(self.timeout, bool)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be finite and greater than zero")
        if type(self.pass_fds) is not tuple or len(set(self.pass_fds)) != len(
            self.pass_fds
        ):
            raise ValueError("pass_fds must be a tuple of unique descriptors")
        for descriptor in self.pass_fds:
            if type(descriptor) is not int or descriptor < 0:
                raise ValueError("pass_fds must contain native file descriptors")
            try:
                metadata = os.fstat(descriptor)
            except OSError as exc:
                raise ValueError("pass_fds contains a closed file descriptor") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("pass_fds descriptors must refer to regular files")


class HelperRunner(Protocol):
    def __call__(self, request: RunRequest) -> subprocess.CompletedProcess[bytes | str]: ...


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    owner: int
    group: int
    size: int
    mode: int
    modified_ns: int


@dataclass(frozen=True)
class _ExecutableSnapshot:
    directory: Path
    directory_identity: _FileIdentity
    path: Path
    descriptor: int
    identity: _FileIdentity
    digest: str


@dataclass(frozen=True)
class _DefaultRunOutcome:
    result: subprocess.CompletedProcess[bytes]
    cleanup_diagnostic: str | None


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_bounded_command(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
    timeout: float,
    executable: Path | None = None,
    pass_fds: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[bytes]:
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    streams: dict[str, object] | None = None
    stdout: bytearray | None = None
    stderr: bytearray | None = None
    stderr_truncated = False
    deadline: float | None = None

    try:
        popen_arguments: dict[str, object] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
            "cwd": os.fspath(cwd),
            "env": dict(environment),
            "close_fds": True,
            "start_new_session": True,
            "pass_fds": pass_fds,
        }
        if executable is not None:
            popen_arguments["executable"] = os.fspath(executable)
        process = subprocess.Popen(argv, **popen_arguments)  # type: ignore[arg-type]
        if process.stdout is None or process.stderr is None:
            raise VisionOcrError("Vision helper process pipes were not created")
        streams = {"stdout": process.stdout, "stderr": process.stderr}
        stdout = bytearray()
        stderr = bytearray()
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        for name, stream in streams.items():
            os.set_blocking(stream.fileno(), False)  # type: ignore[attr-defined]
            selector.register(stream, selectors.EVENT_READ, name)

        assert deadline is not None and stdout is not None and stderr is not None
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            events = selector.select(timeout=min(remaining, 0.1))
            for key, _ in events:
                try:
                    chunk = os.read(key.fd, _PIPE_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    if len(stdout) + len(chunk) > MAXIMUM_STDOUT_BYTES:
                        raise VisionOcrError(
                            f"Vision helper output exceeded {MAXIMUM_STDOUT_BYTES} bytes"
                        )
                    stdout.extend(chunk)
                else:
                    available = MAXIMUM_STDERR_BYTES - len(stderr)
                    if available > 0:
                        stderr.extend(chunk[:available])
                    if len(chunk) > available:
                        stderr_truncated = True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(argv, timeout)
        returncode = process.wait(timeout=remaining)
        if stderr_truncated:
            stderr.extend(b"\n...[truncated]")
        return subprocess.CompletedProcess(argv, returncode, bytes(stdout), bytes(stderr))
    except BaseException:
        if process is not None:
            _terminate_process_group(process)
        raise
    finally:
        try:
            if selector is not None:
                try:
                    selector.close()
                except Exception:
                    pass
        finally:
            if process is not None:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass


def _identity_from_stat(metadata: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        owner=metadata.st_uid,
        group=metadata.st_gid,
        size=metadata.st_size,
        mode=metadata.st_mode,
        modified_ns=metadata.st_mtime_ns,
    )


def _mode_allows_effective_execute(identity: _FileIdentity) -> bool:
    effective_uid = os.geteuid()
    if effective_uid == 0:
        return bool(identity.mode & 0o111)
    if effective_uid == identity.owner:
        return bool(identity.mode & stat.S_IXUSR)
    effective_groups = {os.getegid(), *os.getgroups()}
    if identity.group in effective_groups:
        return bool(identity.mode & stat.S_IXGRP)
    return bool(identity.mode & stat.S_IXOTH)


def _same_directory_identity(
    current: _FileIdentity,
    expected: _FileIdentity,
) -> bool:
    return (
        current.device == expected.device
        and current.inode == expected.inode
        and current.owner == expected.owner
        and current.group == expected.group
        and current.mode == expected.mode
        and stat.S_ISDIR(current.mode)
    )


def _same_created_regular_file(
    current: _FileIdentity,
    expected: _FileIdentity,
) -> bool:
    return (
        current.device == expected.device
        and current.inode == expected.inode
        and stat.S_ISREG(current.mode)
    )


def _open_regular_nofollow(path: Path, *, description: str) -> tuple[int, _FileIdentity]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise VisionOcrError(f"{description} must not be a symlink") from exc
        raise VisionOcrError(f"could not open {description}: {exc}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise VisionOcrError(f"{description} must be a regular file")
    return descriptor, _identity_from_stat(metadata)


def _path_has_identity(path: Path, expected: _FileIdentity) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError:
        return False
    return _identity_from_stat(metadata) == expected


def _hash_descriptor(
    descriptor: int,
    *,
    maximum_bytes: int,
) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > maximum_bytes:
            raise VisionOcrError("Vision helper file size exceeds the safe limit")
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), total


def _path_matches_digest(
    path: Path,
    expected: _FileIdentity,
    expected_digest: str,
) -> bool:
    descriptor = -1
    try:
        descriptor, opened = _open_regular_nofollow(path, description="Vision helper")
        if opened != expected:
            return False
        digest, size = _hash_descriptor(
            descriptor,
            maximum_bytes=MAXIMUM_HELPER_BYTES,
        )
        return (
            size == expected.size
            and digest == expected_digest
            and _identity_from_stat(os.fstat(descriptor)) == expected
            and _path_has_identity(path, expected)
        )
    except (OSError, VisionOcrError):
        return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _add_exception_note(error: BaseException, diagnostic: str) -> None:
    try:
        error.add_note(diagnostic)
    except AttributeError:
        pass


def _cleanup_unexposed_named_file(
    path: Path,
    expected: _FileIdentity | None,
) -> str | None:
    """Remove a just-created private entry before any callback can observe it."""

    if expected is not None:
        try:
            current = _identity_from_stat(os.lstat(path))
        except OSError as exc:
            return f"OCR temporary PNG cleanup failed: {exc}"
        if not _same_created_regular_file(current, expected):
            return "OCR temporary PNG cleanup refused: file identity changed"
    try:
        os.unlink(path)
    except OSError as exc:
        return f"OCR temporary PNG cleanup failed: {exc}"
    return None


def _cleanup_private_snapshot(snapshot: _ExecutableSnapshot) -> str | None:
    """Remove only the unexposed private snapshot created for this call.

    Identity checks prevent accidental replacement cleanup. As documented at
    launch, this path-based boundary does not resist a hostile same-UID process
    racing the final check and unlink on Darwin.
    """

    try:
        directory_metadata = os.lstat(snapshot.directory)
    except OSError as exc:
        return f"Vision helper private cleanup failed: call directory: {exc}"
    if not _same_directory_identity(
        _identity_from_stat(directory_metadata),
        snapshot.directory_identity,
    ):
        return (
            "Vision helper private cleanup refused: call directory identity changed"
        )
    if not _path_matches_digest(snapshot.path, snapshot.identity, snapshot.digest):
        return "Vision helper private cleanup refused: snapshot identity changed"
    try:
        os.unlink(snapshot.path)
    except OSError as exc:
        return f"Vision helper private cleanup failed: snapshot unlink: {exc}"
    try:
        current_directory = _identity_from_stat(os.lstat(snapshot.directory))
        if not _same_directory_identity(
            current_directory,
            snapshot.directory_identity,
        ):
            return (
                "Vision helper private cleanup refused: call directory identity changed"
            )
        os.rmdir(snapshot.directory)
    except OSError as exc:
        return f"Vision helper private cleanup failed: call directory removal: {exc}"
    return None


def _verify_codesign(snapshot: _ExecutableSnapshot) -> None:
    if not _path_matches_digest(snapshot.path, snapshot.identity, snapshot.digest):
        raise VisionOcrError("Vision helper snapshot changed before code-sign verification")
    command = [CODESIGN, "--verify", "--strict", os.fspath(snapshot.path)]
    try:
        result = _run_bounded_command(
            command,
            environment={"PATH": SAFE_PATH, "LANG": "C", "LC_ALL": "C"},
            cwd=Path("/"),
            timeout=_CODESIGN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise VisionOcrError(
            "Vision helper code-sign verification timed out after "
            f"{_CODESIGN_TIMEOUT_SECONDS:g} seconds"
        ) from exc
    if result.returncode != 0:
        raise VisionOcrError(
            "Vision helper snapshot failed code-sign verification: "
            f"{_safe_diagnostic(result.stderr)}"
        )
    if not _path_matches_digest(snapshot.path, snapshot.identity, snapshot.digest):
        raise VisionOcrError("Vision helper snapshot changed during code-sign verification")


def _default_run_helper(
    request: RunRequest,
    snapshot: _ExecutableSnapshot,
) -> _DefaultRunOutcome:
    result: subprocess.CompletedProcess[bytes] | None = None
    primary_error: BaseException | None = None
    cleanup_diagnostic: str | None = None
    try:
        _verify_codesign(snapshot)
        if not _path_matches_digest(snapshot.path, snapshot.identity, snapshot.digest):
            raise VisionOcrError(
                "Vision helper snapshot changed immediately before launch"
            )

        # Darwin has no public fexecve/execveat and does not execute Mach-O from
        # /dev/fd. The random 0700 directory is therefore the strongest public-API
        # boundary available here. It is not claimed to resist a hostile same-UID
        # process. Swift/Foundation may lazily map the executable, so the private
        # name must remain until stdout/stderr are drained and the child is reaped.
        result = _run_bounded_command(
            list(request.argv),
            environment=dict(request.environment),
            cwd=request.cwd,
            timeout=request.timeout,
            executable=snapshot.path,
            pass_fds=request.pass_fds,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_diagnostic = _cleanup_private_snapshot(snapshot)
        if cleanup_diagnostic is not None and primary_error is not None:
            _add_exception_note(primary_error, cleanup_diagnostic)
    assert result is not None
    return _DefaultRunOutcome(
        result=result,
        cleanup_diagnostic=cleanup_diagnostic,
    )


def _strict_json_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _native_number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VisionOcrError(f"{name} must be a finite native number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise VisionOcrError(f"{name} must be a finite native number") from exc
    if not math.isfinite(number):
        raise VisionOcrError(f"{name} must be a finite native number")
    return number


def _normalized_coordinate(value: object, *, name: str) -> float:
    number = _native_number(value, name=name)
    if number < -_BOUNDING_BOX_EPSILON or number > 1.0 + _BOUNDING_BOX_EPSILON:
        raise VisionOcrError(f"{name} must be normalized between 0 and 1")
    return min(1.0, max(0.0, number))


def _parse_box(value: object) -> BoundingBox:
    if type(value) is not dict or set(value) != {"x", "y", "width", "height"}:
        raise VisionOcrError("line box has an invalid schema")
    x = _normalized_coordinate(value["x"], name="box.x")
    y = _normalized_coordinate(value["y"], name="box.y")
    width = _normalized_coordinate(value["width"], name="box.width")
    height = _normalized_coordinate(value["height"], name="box.height")
    if width <= 0.0 or height <= 0.0:
        raise VisionOcrError("box width and height must be greater than zero")
    if x + width > 1.0 + _BOUNDING_BOX_EPSILON:
        raise VisionOcrError("box x + width must not exceed 1")
    if y + height > 1.0 + _BOUNDING_BOX_EPSILON:
        raise VisionOcrError("box y + height must not exceed 1")
    width = min(width, 1.0 - x)
    height = min(height, 1.0 - y)
    if width <= 0.0 or height <= 0.0:
        raise VisionOcrError("box width and height must remain positive")
    return BoundingBox(x=x, y=y, width=width, height=height)


def _order_lines(lines: list[VisionLine]) -> tuple[VisionLine, ...]:
    rows: list[list[tuple[int, VisionLine]]] = []
    row_y: list[float] = []
    indexed = sorted(
        enumerate(lines),
        key=lambda item: (-item[1].box.y, item[1].box.x, item[0]),
    )
    for original_index, line in indexed:
        row_index = next(
            (
                index
                for index, y in enumerate(row_y)
                if abs(line.box.y - y)
                <= _READING_ORDER_VERTICAL_TOLERANCE + 1e-12
            ),
            None,
        )
        if row_index is None:
            rows.append([(original_index, line)])
            row_y.append(line.box.y)
        else:
            rows[row_index].append((original_index, line))
    return tuple(
        line
        for row in rows
        for _, line in sorted(row, key=lambda item: (item[1].box.x, item[0]))
    )


def _parse_helper_output(stdout: bytes | str) -> VisionPageResult:
    if isinstance(stdout, str):
        try:
            raw = stdout.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise VisionOcrError("Vision helper output is not valid UTF-8") from exc
    elif isinstance(stdout, bytes):
        raw = stdout
    else:
        raise VisionOcrError("Vision helper returned an invalid output type")
    if len(raw) > MAXIMUM_STDOUT_BYTES:
        raise VisionOcrError(
            f"Vision helper output exceeded {MAXIMUM_STDOUT_BYTES} bytes"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise VisionOcrError("Vision helper output is not valid UTF-8") from exc
    try:
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object_pairs,
        )
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise VisionOcrError(f"Vision helper returned invalid JSON: {exc}") from exc
    if type(payload) is not dict or set(payload) != {"schema_version", "lines"}:
        raise VisionOcrError("Vision helper response has an invalid schema")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != OCR_SCHEMA_VERSION
    ):
        raise VisionOcrError("Vision helper response has an unsupported schema version")
    raw_lines = payload["lines"]
    if type(raw_lines) is not list:
        raise VisionOcrError("Vision helper lines must be a list")

    lines: list[VisionLine] = []
    total_scalars = 0
    total_utf8_bytes = 0
    for raw_line in raw_lines:
        if type(raw_line) is not dict or set(raw_line) != {"text", "confidence", "box"}:
            raise VisionOcrError("Vision helper line has an invalid schema")
        line_text = raw_line["text"]
        if type(line_text) is not str or not line_text.strip():
            raise VisionOcrError("Vision helper line text must not be blank")
        try:
            encoded_text = line_text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise VisionOcrError("Vision helper line text is not valid Unicode") from exc
        total_scalars += len(line_text)
        total_utf8_bytes += len(encoded_text)
        if (
            total_scalars > MAXIMUM_TEXT_UNICODE_SCALARS
            or total_utf8_bytes > MAXIMUM_TEXT_UTF8_BYTES
        ):
            raise VisionOcrError("Vision helper text budget was exceeded")
        confidence = _native_number(raw_line["confidence"], name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise VisionOcrError("confidence must be between 0 and 1")
        lines.append(
            VisionLine(
                text=line_text,
                confidence=confidence,
                box=_parse_box(raw_line["box"]),
            )
        )
    return VisionPageResult(
        schema_version=OCR_SCHEMA_VERSION,
        lines=_order_lines(lines),
    )


def _safe_diagnostic(value: bytes | str) -> str:
    if isinstance(value, bytes):
        raw = value[:MAXIMUM_DIAGNOSTIC_BYTES]
        rendered = raw.decode("utf-8", errors="replace")
        truncated = len(value) > len(raw)
    elif isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
        raw = encoded[:MAXIMUM_DIAGNOSTIC_BYTES]
        rendered = raw.decode("utf-8", errors="ignore")
        truncated = len(encoded) > len(raw)
    else:
        return "invalid diagnostic output"
    if truncated:
        rendered += "\n...[truncated]"
    return rendered.strip() or "no diagnostic output"


class VisionOcrEngine:
    def __init__(
        self,
        *,
        helper: Path,
        temp_root: Path,
        run_helper: HelperRunner | None = None,
    ) -> None:
        if not isinstance(helper, Path):
            raise TypeError("helper must be a pathlib.Path")
        if not isinstance(temp_root, Path):
            raise TypeError("temp_root must be a pathlib.Path")
        if run_helper is not None and not callable(run_helper):
            raise TypeError("run_helper must be callable")
        self._helper = helper
        self._temp_root = temp_root
        self._run_helper = run_helper

    def _validate_helper(self) -> tuple[int, _FileIdentity, str]:
        if not self._helper.is_absolute():
            raise VisionOcrError("Vision helper path must be absolute")
        descriptor, identity = _open_regular_nofollow(
            self._helper,
            description="Vision helper",
        )
        try:
            if identity.size <= 0:
                raise VisionOcrError("Vision helper must not be empty")
            if identity.size > MAXIMUM_HELPER_BYTES:
                raise VisionOcrError("Vision helper file size exceeds the safe limit")
            if not _mode_allows_effective_execute(identity):
                raise VisionOcrError("Vision helper must be executable")
            try:
                accessible = os.access(
                    self._helper,
                    os.X_OK,
                    effective_ids=True,
                )
            except TypeError:
                accessible = os.access(self._helper, os.X_OK)
            if not accessible or not _path_has_identity(self._helper, identity):
                raise VisionOcrError("Vision helper must be executable")
            digest, size = _hash_descriptor(
                descriptor,
                maximum_bytes=MAXIMUM_HELPER_BYTES,
            )
            after = _identity_from_stat(os.fstat(descriptor))
            if after != identity or size != identity.size:
                raise VisionOcrError("Vision helper changed while being validated")
            return descriptor, identity, digest
        except BaseException:
            os.close(descriptor)
            raise

    def _prepare_temp_root(self) -> Path:
        if not self._temp_root.is_absolute():
            raise VisionOcrError("OCR temporary directory path must be absolute")
        if self._temp_root.is_symlink():
            raise VisionOcrError("OCR temporary directory must not be a symlink")
        try:
            self._temp_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = os.lstat(self._temp_root)
            if not stat.S_ISDIR(metadata.st_mode):
                raise VisionOcrError("OCR temporary path must be a directory")
            if self._temp_root.is_symlink():
                raise VisionOcrError("OCR temporary directory must not be a symlink")
            os.chmod(self._temp_root, 0o700, follow_symlinks=False)
        except VisionOcrError:
            raise
        except OSError as exc:
            raise VisionOcrError(f"could not prepare OCR temporary directory: {exc}") from exc
        return self._temp_root

    @staticmethod
    def _render_page(pdf: Path, page_index: int) -> tuple[fitz.Pixmap, int]:
        if type(page_index) is not int:
            raise TypeError("page_index must be a native integer")
        if page_index < 0:
            raise VisionOcrError("page_index must not be negative")
        if not isinstance(pdf, Path):
            raise TypeError("pdf must be a pathlib.Path")
        if not pdf.is_absolute():
            raise VisionOcrError("PDF path must be absolute")
        descriptor, identity = _open_regular_nofollow(pdf, description="PDF")
        os.close(descriptor)
        document: fitz.Document | None = None
        try:
            document = fitz.open(pdf)
            if document.needs_pass and not document.authenticate(""):
                raise VisionOcrError("PDF is encrypted and cannot be opened")
            if page_index >= len(document):
                raise VisionOcrError("page_index is outside the PDF page range")
            page = document.load_page(page_index)
            width_points = float(page.rect.width)
            height_points = float(page.rect.height)
            if (
                not math.isfinite(width_points)
                or not math.isfinite(height_points)
                or width_points <= 0.0
                or height_points <= 0.0
            ):
                raise VisionOcrError("PDF page has invalid dimensions")

            scale = TARGET_DPI / 72.0
            desired_long_edge = max(width_points, height_points) * scale
            desired_pixels = width_points * height_points * scale * scale
            if not math.isfinite(desired_long_edge) or not math.isfinite(desired_pixels):
                raise VisionOcrError("PDF page dimensions are too large")
            if desired_long_edge > MAXIMUM_LONG_EDGE_PIXELS:
                scale *= MAXIMUM_LONG_EDGE_PIXELS / desired_long_edge
            scaled_pixels = width_points * height_points * scale * scale
            if scaled_pixels > MAXIMUM_PAGE_PIXELS:
                scale *= math.sqrt(MAXIMUM_PAGE_PIXELS / scaled_pixels)
            if not math.isfinite(scale) or scale <= 0.0:
                raise VisionOcrError("PDF page cannot be rendered at a safe scale")

            pixmap: fitz.Pixmap | None = None
            for _ in range(4):
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    colorspace=fitz.csGRAY,
                    alpha=False,
                )
                if (
                    pixmap.width > 0
                    and pixmap.height > 0
                    and max(pixmap.width, pixmap.height) <= MAXIMUM_LONG_EDGE_PIXELS
                    and pixmap.width * pixmap.height <= MAXIMUM_PAGE_PIXELS
                ):
                    break
                next_scale = scale * min(
                    MAXIMUM_LONG_EDGE_PIXELS / max(pixmap.width, pixmap.height),
                    math.sqrt(MAXIMUM_PAGE_PIXELS / (pixmap.width * pixmap.height)),
                ) * (1.0 - 1e-9)
                if next_scale <= 0.0 or next_scale >= scale:
                    raise VisionOcrError("PDF page render exceeded the safe pixel limits")
                scale = next_scale
            else:
                raise VisionOcrError("PDF page render exceeded the safe pixel limits")
            assert pixmap is not None
            pixmap.set_dpi(max(1, round(scale * 72)), max(1, round(scale * 72)))
            if not _path_has_identity(pdf, identity):
                raise VisionOcrError("PDF changed while its page was being rendered")
            return pixmap, round(scale * 72)
        except VisionOcrError:
            raise
        except (fitz.FileDataError, RuntimeError, ValueError, OverflowError, MemoryError) as exc:
            raise VisionOcrError(f"could not render PDF page: {exc}") from exc
        finally:
            if document is not None:
                document.close()

    @staticmethod
    def _create_call_directory(temp_root: Path) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="vision-call-", dir=temp_root))
        try:
            os.chmod(directory, 0o700, follow_symlinks=False)
            metadata = os.lstat(directory)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise VisionOcrError("OCR call directory is not private")
            return directory
        except OSError as exc:
            try:
                os.rmdir(directory)
            except OSError:
                pass
            raise VisionOcrError(
                f"could not prepare OCR call directory: {exc}"
            ) from exc
        except BaseException:
            try:
                os.rmdir(directory)
            except OSError:
                pass
            raise

    @staticmethod
    def _write_anonymous_png(pixmap: fitz.Pixmap, directory: Path) -> tuple[int, _FileIdentity]:
        descriptor, raw_path = tempfile.mkstemp(
            prefix="vision-page-",
            suffix=".png",
            dir=directory,
        )
        path = Path(raw_path)
        named_identity: _FileIdentity | None = None
        name_is_linked = True
        primary_error: BaseException | None = None
        try:
            named_identity = _identity_from_stat(os.lstat(path))
            if not stat.S_ISREG(named_identity.mode):
                raise VisionOcrError("OCR temporary PNG must be a regular file")
            os.fchmod(descriptor, 0o600)
            identity = _identity_from_stat(os.fstat(descriptor))
            if not _same_created_regular_file(identity, named_identity):
                raise VisionOcrError("OCR temporary PNG identity changed before unlink")
            os.unlink(path)
            name_is_linked = False
            try:
                png = pixmap.tobytes("png")
            except (RuntimeError, MemoryError) as exc:
                raise VisionOcrError(
                    f"could not encode OCR temporary PNG: {exc}"
                ) from exc
            view = memoryview(png)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short PNG write")
                view = view[written:]
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size <= 0
                or os.pread(descriptor, 8, 0) != b"\x89PNG\r\n\x1a\n"
                or metadata.st_dev != identity.device
                or metadata.st_ino != identity.inode
            ):
                raise VisionOcrError("OCR anonymous PNG has an unsafe identity")
            return descriptor, _identity_from_stat(metadata)
        except OSError as exc:
            primary_error = VisionOcrError(
                f"could not create OCR temporary PNG: {exc}"
            )
            raise primary_error from exc
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error: VisionOcrError | None = None
            if name_is_linked:
                cleanup_diagnostic = _cleanup_unexposed_named_file(
                    path,
                    named_identity,
                )
                if cleanup_diagnostic is not None and primary_error is not None:
                    _add_exception_note(primary_error, cleanup_diagnostic)
                elif cleanup_diagnostic is not None:
                    cleanup_error = VisionOcrError(cleanup_diagnostic)
            if primary_error is not None or cleanup_error is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if cleanup_error is not None:
                raise cleanup_error

    @staticmethod
    def _snapshot_helper(
        source_descriptor: int,
        directory: Path,
        expected: _FileIdentity,
        expected_digest: str,
    ) -> _ExecutableSnapshot:
        """Copy a verified helper fd into an independent private executable."""

        snapshot = directory / "book-vision-ocr"
        descriptor = -1
        succeeded = False
        try:
            directory_identity = _identity_from_stat(os.lstat(directory))
            if (
                not stat.S_ISDIR(directory_identity.mode)
                or stat.S_IMODE(directory_identity.mode) != 0o700
            ):
                raise VisionOcrError("Vision helper call directory is not private")
            descriptor = os.open(
                snapshot,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o700,
            )
            os.fchmod(descriptor, 0o700)

            if _identity_from_stat(os.fstat(source_descriptor)) != expected:
                raise VisionOcrError("Vision helper changed before snapshot creation")
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            copied_digest = hashlib.sha256()
            copied = 0
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > MAXIMUM_HELPER_BYTES:
                    raise VisionOcrError("Vision helper file size exceeds the safe limit")
                copied_digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short helper snapshot write")
                    view = view[written:]
            os.fsync(descriptor)
            snapshot_identity = _identity_from_stat(os.fstat(descriptor))
            source_after = _identity_from_stat(os.fstat(source_descriptor))
            source_digest, source_size = _hash_descriptor(
                source_descriptor,
                maximum_bytes=MAXIMUM_HELPER_BYTES,
            )
            snapshot_digest = copied_digest.hexdigest()
            if (
                source_after != expected
                or source_size != expected.size
                or source_digest != expected_digest
                or copied != expected.size
                or snapshot_digest != expected_digest
            ):
                raise VisionOcrError("Vision helper changed during snapshot creation")
            if (
                not stat.S_ISREG(snapshot_identity.mode)
                or stat.S_IMODE(snapshot_identity.mode) != 0o700
                or snapshot_identity.size != expected.size
            ):
                raise VisionOcrError("Vision helper snapshot has an unsafe file identity")
            if not _path_has_identity(snapshot, snapshot_identity):
                raise VisionOcrError("Vision helper snapshot changed while being created")
            os.lseek(descriptor, 0, os.SEEK_SET)
            succeeded = True
            return _ExecutableSnapshot(
                directory=directory,
                directory_identity=directory_identity,
                path=snapshot,
                descriptor=descriptor,
                identity=snapshot_identity,
                digest=snapshot_digest,
            )
        finally:
            if not succeeded and descriptor >= 0:
                os.close(descriptor)
            if not succeeded:
                try:
                    os.unlink(snapshot)
                except FileNotFoundError:
                    pass

    def recognize_page(self, pdf: Path, *, page_index: int) -> VisionPageResult:
        helper_descriptor = -1
        helper_identity: _FileIdentity | None = None
        helper_digest: str | None = None
        png_descriptor = -1
        call_directory: Path | None = None
        snapshot: _ExecutableSnapshot | None = None
        snapshot_cleanup_attempted = False
        default_cleanup_diagnostic: str | None = None
        primary_error: BaseException | None = None
        try:
            helper_descriptor, helper_identity, helper_digest = self._validate_helper()
            temp_root = self._prepare_temp_root()
            pixmap, _rendered_dpi = self._render_page(pdf, page_index)
            call_directory = self._create_call_directory(temp_root)
            png_descriptor, _png_identity = self._write_anonymous_png(
                pixmap,
                call_directory,
            )
            snapshot = self._snapshot_helper(
                helper_descriptor,
                call_directory,
                helper_identity,
                helper_digest,
            )
            request = RunRequest(
                argv=(
                    os.fspath(self._helper),
                    "--image",
                    f"/dev/fd/{png_descriptor}",
                    "--languages",
                    _LANGUAGES,
                ),
                environment=(
                    ("PATH", SAFE_PATH),
                    ("LANG", "en_US.UTF-8"),
                    ("LC_ALL", "en_US.UTF-8"),
                ),
                cwd=Path("/"),
                timeout=HELPER_TIMEOUT_SECONDS,
                pass_fds=(png_descriptor,),
            )
            if self._run_helper is None:
                snapshot_cleanup_attempted = True
                outcome = _default_run_helper(request, snapshot)
                result: subprocess.CompletedProcess[bytes | str] = outcome.result
                default_cleanup_diagnostic = outcome.cleanup_diagnostic
            else:
                # Test/injected runners receive only the public descriptor contract.
                # Remove every private name before invoking external callback code.
                snapshot_cleanup_attempted = True
                cleanup_diagnostic = _cleanup_private_snapshot(snapshot)
                if cleanup_diagnostic is not None:
                    raise VisionOcrError(cleanup_diagnostic)
                result = self._run_helper(request)

            source_digest, source_size = _hash_descriptor(
                helper_descriptor,
                maximum_bytes=MAXIMUM_HELPER_BYTES,
            )
            if (
                not _path_has_identity(self._helper, helper_identity)
                or _identity_from_stat(os.fstat(helper_descriptor)) != helper_identity
                or source_size != helper_identity.size
                or source_digest != helper_digest
            ):
                raise VisionOcrError("Vision helper changed while recognition was running")
            snapshot_digest, snapshot_size = _hash_descriptor(
                snapshot.descriptor,
                maximum_bytes=MAXIMUM_HELPER_BYTES,
            )
            if (
                snapshot_size != snapshot.identity.size
                or snapshot_digest != snapshot.digest
                or _identity_from_stat(os.fstat(snapshot.descriptor)) != snapshot.identity
            ):
                raise VisionOcrError("Vision helper snapshot changed during recognition")
            if not isinstance(result, subprocess.CompletedProcess):
                raise VisionOcrError("Vision helper runner returned an invalid result")
            if type(result.returncode) is not int:
                raise VisionOcrError("Vision helper returned an invalid exit code")
            if result.returncode != 0:
                raise VisionOcrError(
                    f"Vision helper failed with exit code {result.returncode}: "
                    f"{_safe_diagnostic(result.stderr)}"
                )
            parsed = _parse_helper_output(result.stdout)
            if default_cleanup_diagnostic is not None:
                cleanup_error = VisionOcrError(default_cleanup_diagnostic)
                default_cleanup_diagnostic = None
                raise cleanup_error
            return parsed
        except subprocess.TimeoutExpired as exc:
            primary_error = VisionOcrError(
                f"Vision helper timed out after {HELPER_TIMEOUT_SECONDS:g} seconds"
            )
            for note in getattr(exc, "__notes__", ()):
                _add_exception_note(primary_error, note)
            raise primary_error from exc
        except VisionOcrError as exc:
            primary_error = exc
            if default_cleanup_diagnostic is not None:
                _add_exception_note(exc, default_cleanup_diagnostic)
            raise
        except (OSError, subprocess.SubprocessError, TypeError) as exc:
            primary_error = VisionOcrError(f"could not run Vision helper: {exc}")
            for note in getattr(exc, "__notes__", ()):
                _add_exception_note(primary_error, note)
            raise primary_error from exc
        except BaseException as exc:
            primary_error = exc
            if default_cleanup_diagnostic is not None:
                _add_exception_note(exc, default_cleanup_diagnostic)
            raise
        finally:
            for descriptor in (
                png_descriptor,
                snapshot.descriptor if snapshot else -1,
                helper_descriptor,
            ):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            final_cleanup_diagnostic: str | None = None
            if snapshot is not None and not snapshot_cleanup_attempted:
                final_cleanup_diagnostic = _cleanup_private_snapshot(snapshot)
            elif snapshot is None and call_directory is not None:
                try:
                    os.rmdir(call_directory)
                except OSError as exc:
                    final_cleanup_diagnostic = (
                        "Vision helper private cleanup failed: "
                        f"call directory removal: {exc}"
                    )
            if final_cleanup_diagnostic is not None:
                if primary_error is not None:
                    _add_exception_note(primary_error, final_cleanup_diagnostic)
                else:
                    raise VisionOcrError(final_cleanup_diagnostic)


__all__ = [
    "HELPER_TIMEOUT_SECONDS",
    "MAXIMUM_LONG_EDGE_PIXELS",
    "MAXIMUM_HELPER_BYTES",
    "MAXIMUM_PAGE_PIXELS",
    "MAXIMUM_STDOUT_BYTES",
    "RunRequest",
    "VisionOcrEngine",
    "VisionOcrError",
]
