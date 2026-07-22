from __future__ import annotations

import json
import math
import os
import select
import stat
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from book_agent.ocr.models import BoundingBox, OcrPageResult, VisionLine


class LightOcrError(RuntimeError):
    """A bounded Light OCR process, protocol, or recognition failure."""


def _readline_with_timeout(stream: TextIO, timeout: float) -> str:
    ready, _, _ = select.select([stream], [], [], timeout)
    if not ready:
        raise LightOcrError(f"Light OCR did not respond within {timeout:g} seconds")
    return stream.readline()


class LightOcrEngine:
    def __init__(
        self,
        *,
        node: Path,
        worker: Path,
        timeout_seconds: float = 90.0,
        process_factory: Callable[..., Any] = subprocess.Popen,
        response_reader: Callable[[TextIO, float], str] = _readline_with_timeout,
        request_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        if not isinstance(node, Path) or not isinstance(worker, Path):
            raise ValueError("node and worker must be Path values")
        if (
            type(timeout_seconds) not in (int, float)
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        self.node = node
        self.worker = worker
        self.timeout_seconds = float(timeout_seconds)
        self._process_factory = process_factory
        self._response_reader = response_reader
        self._request_id_factory = request_id_factory
        self._process: Any | None = None
        self._closed = False

    def recognize_image(self, image: Path) -> OcrPageResult:
        if self._closed:
            raise LightOcrError("Light OCR engine is closed")
        image = self._validate_image(image)
        process = self._ensure_process()
        request_id = self._request_id_factory()
        if type(request_id) is not str or not request_id.strip():
            raise LightOcrError("Light OCR request ID must be a nonblank string")
        request = {
            "id": request_id,
            "op": "recognize",
            "image": str(image.resolve()),
        }
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            raw = self._response_reader(process.stdout, self.timeout_seconds)
        except LightOcrError:
            self._abort()
            raise
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._abort()
            raise LightOcrError("Light OCR worker communication failed") from exc
        if not raw:
            self._abort()
            raise LightOcrError("Light OCR worker closed without a response")
        if len(raw) > 8 * 1024 * 1024:
            self._abort()
            raise LightOcrError("Light OCR response exceeded the protocol limit")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._abort()
            raise LightOcrError("Light OCR response must be valid JSON") from exc
        if type(payload) is not dict:
            self._abort()
            raise LightOcrError("Light OCR response must be a JSON object")
        if payload.get("id") != request_id:
            self._abort()
            raise LightOcrError("Light OCR response request ID does not match")
        if payload.get("ok") is False:
            detail = payload.get("error")
            message = detail.strip() if isinstance(detail, str) else "recognition failed"
            raise LightOcrError(f"Light OCR: {message[:500]}")
        if payload.get("ok") is not True or type(payload.get("lines")) is not list:
            self._abort()
            raise LightOcrError("Light OCR response lines must be a list")
        raw_lines = payload["lines"]
        if len(raw_lines) > 10000:
            self._abort()
            raise LightOcrError("Light OCR response contained too many lines")
        lines: list[VisionLine] = []
        discarded = 0
        for item in raw_lines:
            try:
                lines.append(self._line(item))
            except (TypeError, ValueError):
                discarded += 1
        return OcrPageResult(
            engine="light_ocr",
            lines=tuple(lines),
            discarded_observations=discarded,
        )

    @staticmethod
    def _validate_image(image: Path) -> Path:
        if not isinstance(image, Path):
            raise ValueError("image must be a Path")
        if image.suffix.casefold() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Light OCR accepts PNG or JPEG images")
        try:
            info = image.lstat()
        except OSError as exc:
            raise ValueError(f"Light OCR image is unavailable: {image}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("Light OCR image must be a regular file")
        return image

    @staticmethod
    def _line(item: object) -> VisionLine:
        if type(item) is not dict:
            raise TypeError("line must be an object")
        text = item.get("text")
        confidence = item.get("confidence")
        box = item.get("box")
        if type(text) is not str or type(box) is not dict:
            raise ValueError("line text and box are required")
        if type(confidence) not in (int, float) or isinstance(confidence, bool):
            raise ValueError("line confidence must be numeric")
        values = tuple(box.get(key) for key in ("x", "y", "width", "height"))
        if any(type(value) not in (int, float) or isinstance(value, bool) for value in values):
            raise ValueError("line box values must be numeric")
        x, top, width, height = (float(value) for value in values)
        bottom = 1.0 - top - height
        return VisionLine(
            text=text,
            confidence=float(confidence),
            box=BoundingBox(x=x, y=bottom, width=width, height=height),
        )

    def _ensure_process(self) -> Any:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if self._process is not None:
            self._process = None
        environment = os.environ.copy()
        environment.setdefault("LIGHT_OCR_EXECUTION", "cpu")
        try:
            process = self._process_factory(
                [str(self.node), str(self.worker)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=str(self.worker.parent),
                env=environment,
            )
        except (OSError, ValueError) as exc:
            raise LightOcrError("Light OCR worker could not be started") from exc
        if process.stdin is None or process.stdout is None:
            self._terminate(process)
            raise LightOcrError("Light OCR worker pipes are unavailable")
        self._process = process
        return process

    def _abort(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            self._terminate(process)

    @staticmethod
    def _terminate(process: Any) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, TimeoutError):
                process.kill()
                process.wait(timeout=2.0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.write('{"op":"close"}\n')
                process.stdin.flush()
                process.wait(timeout=5.0)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired, TimeoutError):
                self._terminate(process)

    def __enter__(self) -> LightOcrEngine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = ["LightOcrEngine", "LightOcrError"]
