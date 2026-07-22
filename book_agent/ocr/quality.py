from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from book_agent.ocr.models import OcrPageOutcome, VisionLine


_BLANK_INK_RATIO = 0.002
_MAX_CONTROL_RATIO = 0.10
_MAX_REPLACEMENT_RATIO = 0.10
_MAX_REPEATED_RUN = 24


@dataclass(frozen=True)
class QualityVerdict:
    accepted: bool
    outcome: OcrPageOutcome | None
    score: float
    reason: str


def _longest_repeated_run(text: str) -> int:
    longest = 0
    previous: str | None = None
    current = 0
    for character in text:
        if character == previous:
            current += 1
        else:
            previous = character
            current = 1
        longest = max(longest, current)
    return longest


def assess_page(
    text: str,
    lines: Iterable[VisionLine],
    image_ink_ratio: float,
    *,
    terminal: bool = False,
) -> QualityVerdict:
    """Make bounded, deterministic fallback decisions for one OCR attempt."""

    if type(text) is not str:
        raise ValueError("text must be a native string")
    if (
        type(image_ink_ratio) not in (int, float)
        or isinstance(image_ink_ratio, bool)
        or not math.isfinite(image_ink_ratio)
        or not 0.0 <= image_ink_ratio <= 1.0
    ):
        raise ValueError("image_ink_ratio must be finite and between 0 and 1")
    materialized_lines = tuple(lines)
    if not all(type(line) is VisionLine for line in materialized_lines):
        raise ValueError("lines must contain VisionLine values")
    stripped = text.strip()
    if not stripped:
        if image_ink_ratio <= _BLANK_INK_RATIO:
            return QualityVerdict(
                accepted=True,
                outcome=OcrPageOutcome("blank", None, "blank_page"),
                score=1.0,
                reason="blank_page",
            )
        if terminal:
            return QualityVerdict(
                accepted=True,
                outcome=OcrPageOutcome("image_only", None, "no_text_expected"),
                score=1.0,
                reason="no_text_expected",
            )
        return QualityVerdict(False, None, 0.0, "unexpected_empty_text")

    controls = sum(
        unicodedata.category(character).startswith("C")
        for character in stripped
    )
    replacements = stripped.count("\ufffd")
    length = len(stripped)
    if controls / length > _MAX_CONTROL_RATIO:
        return QualityVerdict(False, None, 0.0, "control_character_heavy")
    if replacements / length > _MAX_REPLACEMENT_RATIO:
        return QualityVerdict(False, None, 0.0, "replacement_character_heavy")
    if _longest_repeated_run(stripped) > _MAX_REPEATED_RUN:
        return QualityVerdict(False, None, 0.0, "repeated_character_run")
    non_control_ratio = 1.0 - controls / length
    line_bonus = min(len(materialized_lines), 8) / 8
    score = round(0.8 * non_control_ratio + 0.2 * line_bonus, 4)
    return QualityVerdict(True, None, score, "accepted")


__all__ = ["QualityVerdict", "assess_page"]
