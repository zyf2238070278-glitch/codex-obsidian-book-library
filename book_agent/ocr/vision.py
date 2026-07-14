from __future__ import annotations

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
MAXIMUM_TEXT_UTF8_BYTES = 400_000
MAXIMUM_TEXT_UNICODE_SCALARS = 100_000
SAFE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
_LANGUAGES = "zh-Hans,en-US"
_PIPE_CHUNK_BYTES = 16 * 1024
_BOUNDING_BOX_EPSILON = 1e-6
_READING_ORDER_VERTICAL_TOLERANCE = 0.0125


class VisionOcrError(ValueError):
    """A PDF page could not be rendered or recognized safely."""


class HelperRunner(Protocol):
    def __call__(
        self,
        argv: list[str],
        *,
        environment: Mapping[str, str],
        cwd: Path,
        timeout: float,
        executable: Path,
    ) -> subprocess.CompletedProcess[bytes | str]: ...


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mode: int
    modified_ns: int


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


def _default_run_helper(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
    timeout: float,
    executable: Path,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        cwd=os.fspath(cwd),
        env=dict(environment),
        close_fds=True,
        start_new_session=True,
        executable=os.fspath(executable),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    stdout = bytearray()
    stderr = bytearray()
    stderr_truncated = False
    deadline = time.monotonic() + timeout

    for name, stream in streams.items():
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)

    try:
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
        _terminate_process_group(process)
        raise
    finally:
        selector.close()
        for stream in streams.values():
            stream.close()


def _identity_from_stat(metadata: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mode=metadata.st_mode,
        modified_ns=metadata.st_mtime_ns,
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
    except (json.JSONDecodeError, ValueError) as exc:
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
        run_helper: HelperRunner = _default_run_helper,
    ) -> None:
        if not isinstance(helper, Path):
            raise TypeError("helper must be a pathlib.Path")
        if not isinstance(temp_root, Path):
            raise TypeError("temp_root must be a pathlib.Path")
        if not callable(run_helper):
            raise TypeError("run_helper must be callable")
        self._helper = helper
        self._temp_root = temp_root
        self._run_helper = run_helper

    def _validate_helper(self) -> _FileIdentity:
        if not self._helper.is_absolute():
            raise VisionOcrError("Vision helper path must be absolute")
        descriptor, identity = _open_regular_nofollow(
            self._helper,
            description="Vision helper",
        )
        os.close(descriptor)
        if identity.size <= 0:
            raise VisionOcrError("Vision helper must not be empty")
        if identity.mode & 0o111 == 0 or not os.access(self._helper, os.X_OK):
            raise VisionOcrError("Vision helper must be executable")
        return identity

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
    def _write_png(pixmap: fitz.Pixmap, temp_root: Path) -> tuple[Path, _FileIdentity]:
        descriptor = -1
        path: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                prefix="vision-page-",
                suffix=".png",
                dir=temp_root,
            )
            path = Path(raw_path)
            os.fchmod(descriptor, 0o600)
            png = pixmap.tobytes("png")
            view = memoryview(png)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise VisionOcrError("OCR temporary PNG must be a regular file")
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise VisionOcrError("OCR temporary PNG must have mode 0600")
            identity = _identity_from_stat(metadata)
            if not _path_has_identity(path, identity):
                raise VisionOcrError("OCR temporary PNG changed while being written")
            return path, identity
        except BaseException:
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _pin_helper(
        helper: Path,
        temp_root: Path,
        expected: _FileIdentity,
    ) -> Path:
        """Hard-link the verified helper so path replacement cannot change execution."""

        descriptor, raw_path = tempfile.mkstemp(
            prefix="vision-helper-",
            suffix=".pinned",
            dir=temp_root,
        )
        os.close(descriptor)
        pinned = Path(raw_path)
        try:
            pinned.unlink()
            os.link(helper, pinned, follow_symlinks=False)
            metadata = os.lstat(pinned)
            if not stat.S_ISREG(metadata.st_mode):
                raise VisionOcrError("pinned Vision helper must be a regular file")
            if _identity_from_stat(metadata) != expected:
                raise VisionOcrError("Vision helper changed before recognition started")
            return pinned
        except VisionOcrError:
            pinned.unlink(missing_ok=True)
            raise
        except OSError as exc:
            pinned.unlink(missing_ok=True)
            raise VisionOcrError(f"could not pin Vision helper safely: {exc}") from exc

    def recognize_page(self, pdf: Path, *, page_index: int) -> VisionPageResult:
        helper_identity = self._validate_helper()
        temp_root = self._prepare_temp_root()
        pixmap, _rendered_dpi = self._render_page(pdf, page_index)
        image_path: Path | None = None
        image_identity: _FileIdentity | None = None
        pinned_helper: Path | None = None
        try:
            image_path, image_identity = self._write_png(pixmap, temp_root)
            pinned_helper = self._pin_helper(
                self._helper,
                temp_root,
                helper_identity,
            )
            argv = [
                os.fspath(self._helper),
                "--image",
                os.fspath(image_path),
                "--languages",
                _LANGUAGES,
            ]
            environment = {
                "PATH": SAFE_PATH,
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
            }
            try:
                result = self._run_helper(
                    argv,
                    environment=environment,
                    cwd=Path("/"),
                    timeout=HELPER_TIMEOUT_SECONDS,
                    executable=pinned_helper,
                )
            except subprocess.TimeoutExpired as exc:
                raise VisionOcrError(
                    f"Vision helper timed out after {HELPER_TIMEOUT_SECONDS:g} seconds"
                ) from exc
            except VisionOcrError:
                raise
            except (OSError, subprocess.SubprocessError, TypeError) as exc:
                raise VisionOcrError(f"could not run Vision helper: {exc}") from exc

            if not _path_has_identity(self._helper, helper_identity):
                raise VisionOcrError("Vision helper changed while recognition was running")
            if not _path_has_identity(pinned_helper, helper_identity):
                raise VisionOcrError("pinned Vision helper changed during recognition")
            if not _path_has_identity(image_path, image_identity):
                raise VisionOcrError("OCR temporary PNG changed during recognition")
            if not isinstance(result, subprocess.CompletedProcess):
                raise VisionOcrError("Vision helper runner returned an invalid result")
            if type(result.returncode) is not int:
                raise VisionOcrError("Vision helper returned an invalid exit code")
            if result.returncode != 0:
                raise VisionOcrError(
                    f"Vision helper failed with exit code {result.returncode}: "
                    f"{_safe_diagnostic(result.stderr)}"
                )
            return _parse_helper_output(result.stdout)
        finally:
            if image_path is not None:
                try:
                    image_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if pinned_helper is not None:
                try:
                    pinned_helper.unlink(missing_ok=True)
                except OSError:
                    pass


__all__ = [
    "HELPER_TIMEOUT_SECONDS",
    "MAXIMUM_LONG_EDGE_PIXELS",
    "MAXIMUM_PAGE_PIXELS",
    "MAXIMUM_STDOUT_BYTES",
    "VisionOcrEngine",
    "VisionOcrError",
]
