from __future__ import annotations

import csv
import math
import os
import stat
import subprocess
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any

import fitz

from book_agent.ocr.models import BoundingBox, OcrPageResult, VisionLine


LANGUAGES = "chi_sim+chi_tra+eng"
REQUIRED_TESSDATA = ("chi_sim.traineddata", "chi_tra.traineddata", "eng.traineddata")
SAFE_ENVIRONMENT = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}


class TesseractError(ValueError):
    """The packaged Tesseract fallback could not recognize a rendered page."""


class TesseractEngine:
    def __init__(
        self,
        *,
        binary: Path,
        tessdata: Path,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._binary = binary
        self._tessdata = tessdata
        self._runner = subprocess.run if runner is None else runner

    def available(self) -> bool:
        try:
            binary_info = self._binary.stat()
        except OSError:
            return False
        return (
            self._binary.is_file()
            and stat.S_ISREG(binary_info.st_mode)
            and bool(binary_info.st_mode & stat.S_IXUSR)
            and self._tessdata.is_dir()
            and all((self._tessdata / name).is_file() for name in REQUIRED_TESSDATA)
        )

    @staticmethod
    def _image_size(image: Path) -> tuple[int, int]:
        try:
            pixmap = fitz.Pixmap(image)
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise TesseractError(f"could not read rendered OCR image: {exc}") from exc
        if pixmap.width <= 0 or pixmap.height <= 0:
            raise TesseractError("rendered OCR image has invalid dimensions")
        return pixmap.width, pixmap.height

    @staticmethod
    def _parse_tsv(tsv: str, *, width: int, height: int) -> OcrPageResult:
        if not isinstance(tsv, str):
            raise TesseractError("Tesseract returned non-text TSV output")
        reader = csv.DictReader(StringIO(tsv), delimiter="\t")
        if reader.fieldnames is None or set(reader.fieldnames) != {
            "level", "page_num", "block_num", "par_num", "line_num", "word_num",
            "left", "top", "width", "height", "conf", "text",
        }:
            raise TesseractError("Tesseract returned invalid TSV columns")
        lines: list[VisionLine] = []
        discarded = 0
        for row in reader:
            try:
                text = row["text"].strip()
                confidence = float(row["conf"])
                left = float(row["left"])
                top = float(row["top"])
                item_width = float(row["width"])
                item_height = float(row["height"])
                if (
                    not text
                    or not all(math.isfinite(value) for value in (confidence, left, top, item_width, item_height))
                    or not 0.0 <= confidence <= 100.0
                    or item_width <= 0.0
                    or item_height <= 0.0
                ):
                    raise ValueError
                x = max(0.0, left / width)
                y = max(0.0, top / height)
                right = min(1.0, (left + item_width) / width)
                bottom = min(1.0, (top + item_height) / height)
                lines.append(
                    VisionLine(
                        text=text,
                        confidence=confidence / 100.0,
                        box=BoundingBox(x, y, right - x, bottom - y),
                    )
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                discarded += 1
        return OcrPageResult(
            engine="tesseract",
            lines=tuple(lines),
            discarded_observations=discarded,
        )

    def recognize_image(self, image: Path) -> OcrPageResult:
        if not isinstance(image, Path) or not image.is_absolute():
            raise TesseractError("rendered OCR image path must be absolute")
        if not self.available():
            raise TesseractError("packaged Tesseract binary or language data is missing")
        width, height = self._image_size(image)
        command = [
            os.fspath(self._binary),
            os.fspath(image),
            "stdout",
            "--tessdata-dir",
            os.fspath(self._tessdata),
            "-l",
            LANGUAGES,
            "tsv",
        ]
        try:
            completed = self._runner(
                command,
                shell=False,
                check=False,
                text=True,
                capture_output=True,
                timeout=120,
                env=SAFE_ENVIRONMENT.copy(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TesseractError(f"could not run packaged Tesseract: {exc}") from exc
        if completed.returncode != 0:
            raise TesseractError(f"Tesseract failed with exit code {completed.returncode}")
        return self._parse_tsv(completed.stdout, width=width, height=height)


__all__ = ["TesseractEngine", "TesseractError"]
