from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


OCR_SCHEMA_VERSION = 1
_READING_ORDER_VERTICAL_TOLERANCE = 0.0125
_OcrStatus = Literal["queued", "running", "paused", "failed", "completed"]
_OCR_STATUSES = ("queued", "running", "paused", "failed", "completed")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.x <= 1.0:
            raise ValueError("x must be between 0 and 1")
        if not 0.0 <= self.y <= 1.0:
            raise ValueError("y must be between 0 and 1")
        if not 0.0 < self.width <= 1.0:
            raise ValueError("width must be greater than 0 and at most 1")
        if not 0.0 < self.height <= 1.0:
            raise ValueError("height must be greater than 0 and at most 1")
        if self.x + self.width > 1.0:
            raise ValueError("x + width must not exceed 1")
        if self.y + self.height > 1.0:
            raise ValueError("y + height must not exceed 1")


@dataclass(frozen=True)
class VisionLine:
    text: str
    confidence: float
    box: BoundingBox

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must not be blank")
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and between 0 and 1")
        if type(self.box) is not BoundingBox:
            raise ValueError("box must be a BoundingBox")


def _group_lines(
    lines: tuple[VisionLine, ...], *, vertical_tolerance: float
) -> list[list[VisionLine]]:
    """Group Vision coordinates into rows using a normalized 0.0125 y tolerance.

    Apple Vision uses a bottom-left origin, so larger ``y`` coordinates are read
    first. Each row is anchored to its first (topmost) line to prevent cumulative
    drift from merging neighboring rows.
    """

    rows: list[list[VisionLine]] = []
    row_y_positions: list[float] = []
    for line in sorted(lines, key=lambda item: (-item.box.y, item.box.x)):
        matching_row = next(
            (
                index
                for index, row_y in enumerate(row_y_positions)
                if abs(line.box.y - row_y) <= vertical_tolerance + 1e-12
            ),
            None,
        )
        if matching_row is None:
            rows.append([line])
            row_y_positions.append(line.box.y)
        else:
            rows[matching_row].append(line)
    return rows


@dataclass(frozen=True)
class VisionPageResult:
    schema_version: int
    lines: tuple[VisionLine, ...]

    def __post_init__(self) -> None:
        if (
            not _is_int(self.schema_version)
            or self.schema_version != OCR_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be exactly {OCR_SCHEMA_VERSION}"
            )
        if not isinstance(self.lines, tuple) or not all(
            type(line) is VisionLine for line in self.lines
        ):
            raise ValueError("lines must be a tuple of VisionLine values")

    def ordered_text(self) -> str:
        rows = _group_lines(
            self.lines,
            vertical_tolerance=_READING_ORDER_VERTICAL_TOLERANCE,
        )
        return "\n".join(
            " ".join(
                line.text for line in sorted(row, key=lambda item: item.box.x)
            )
            for row in rows
        ).strip()

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)


@dataclass(frozen=True)
class OcrJobSummary:
    book_id: str
    title: str
    status: _OcrStatus
    total_pages: int
    completed_pages: int = 0
    current_page: int | None = None
    queue_position: int | None = None
    updated_at: str | None = None
    error: str | None = None
    estimated_remaining_seconds: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.book_id, str) or not self.book_id.strip():
            raise ValueError("book_id must not be blank")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must not be blank")
        if type(self.status) is not str or self.status not in _OCR_STATUSES:
            raise ValueError("status is not a supported OCR job state")
        if not _is_int(self.total_pages) or self.total_pages <= 0:
            raise ValueError("total_pages must be greater than zero")
        if (
            not _is_int(self.completed_pages)
            or not 0 <= self.completed_pages <= self.total_pages
        ):
            raise ValueError("completed_pages must be between 0 and total_pages")
        if self.current_page is not None and (
            not _is_int(self.current_page)
            or not 1 <= self.current_page <= self.total_pages
        ):
            raise ValueError("current_page must be between 1 and total_pages")
        if self.queue_position is not None and (
            not _is_int(self.queue_position) or self.queue_position <= 0
        ):
            raise ValueError("queue_position must be greater than zero")
        for name in ("updated_at", "error"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be a nonblank string or None")
        if self.estimated_remaining_seconds is not None and (
            not _is_int(self.estimated_remaining_seconds)
            or self.estimated_remaining_seconds < 0
        ):
            raise ValueError(
                "estimated_remaining_seconds must be zero or greater"
            )

    @property
    def percent_complete(self) -> float:
        return round(self.completed_pages / self.total_pages * 100.0, 2)
