from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitz

from book_agent.ocr.models import BoundingBox, OcrPageResult, VisionLine


REQUIRED_MODEL_FILES = (
    "PP-OCRv6_det_small.onnx",
    "PP-OCRv6_rec_small.onnx",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
)


class RapidOcrError(ValueError):
    """The packaged offline RapidOCR runtime could not recognize a page."""


class RapidOcrEngine:
    def __init__(
        self,
        model_root: Path,
        *,
        factory: Callable[..., Any] | None = None,
    ) -> None:
        self._model_root = model_root
        self._factory = factory
        self._runtime: Any | None = None

    def available(self) -> bool:
        return self._model_root.is_dir() and all(
            (self._model_root / name).is_file() for name in REQUIRED_MODEL_FILES
        )

    def _load_runtime(self) -> Any:
        if not self.available():
            raise RapidOcrError("RapidOCR model is missing from the offline package")
        if self._runtime is None:
            factory = self._factory
            if factory is None:
                try:
                    from rapidocr import RapidOCR  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise RapidOcrError("RapidOCR runtime is not installed") from exc
                factory = RapidOCR
            try:
                # RapidOCR resolves all detector/classifier/recognizer assets
                # beneath this explicit root.  Supplying no root would make the
                # library fall back to its own cache/download behavior.
                self._runtime = factory(
                    params={"Global.model_root_dir": str(self._model_root)}
                )
            except Exception as exc:
                raise RapidOcrError(f"could not initialize offline RapidOCR: {exc}") from exc
        return self._runtime

    @staticmethod
    def _image_size(image: Path) -> tuple[int, int]:
        try:
            pixmap = fitz.Pixmap(image)
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise RapidOcrError(f"could not read rendered OCR image: {exc}") from exc
        if pixmap.width <= 0 or pixmap.height <= 0:
            raise RapidOcrError("rendered OCR image has invalid dimensions")
        return pixmap.width, pixmap.height

    @staticmethod
    def _normalise(
        raw: Any, *, width: int, height: int
    ) -> OcrPageResult:
        candidates = raw[0] if isinstance(raw, tuple) and raw else raw
        if candidates is None:
            candidates = []
        if not isinstance(candidates, (list, tuple)):
            raise RapidOcrError("RapidOCR returned an invalid result")
        lines: list[VisionLine] = []
        discarded = 0
        for candidate in candidates:
            try:
                polygon, text, confidence = candidate[:3]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError
                numeric_confidence = float(confidence)
                if not math.isfinite(numeric_confidence) or not 0 <= numeric_confidence <= 1:
                    raise ValueError
                points = [(float(point[0]), float(point[1])) for point in polygon]
                if len(points) < 4 or not all(
                    math.isfinite(x) and math.isfinite(y) for x, y in points
                ):
                    raise ValueError
                left = min(x for x, _ in points)
                right = max(x for x, _ in points)
                top = min(y for _, y in points)
                bottom = max(y for _, y in points)
                box = BoundingBox(
                    max(0.0, left / width),
                    max(0.0, top / height),
                    min(1.0, right / width) - max(0.0, left / width),
                    min(1.0, bottom / height) - max(0.0, top / height),
                )
                lines.append(VisionLine(text.strip(), numeric_confidence, box))
            except (IndexError, TypeError, ValueError, OverflowError):
                discarded += 1
        return OcrPageResult(
            engine="rapidocr",
            lines=tuple(lines),
            discarded_observations=discarded,
        )

    def recognize_image(self, image: Path) -> OcrPageResult:
        if not isinstance(image, Path) or not image.is_absolute():
            raise RapidOcrError("rendered OCR image path must be absolute")
        width, height = self._image_size(image)
        try:
            raw = self._load_runtime()(str(image))
        except RapidOcrError:
            raise
        except Exception as exc:
            raise RapidOcrError(f"RapidOCR recognition failed: {exc}") from exc
        return self._normalise(raw, width=width, height=height)


__all__ = ["REQUIRED_MODEL_FILES", "RapidOcrEngine", "RapidOcrError"]
