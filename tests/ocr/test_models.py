import json
from collections import UserString
from dataclasses import FrozenInstanceError, asdict
from math import inf, nan

import pytest

from book_agent.ocr import (
    BoundingBox,
    OcrJobSummary,
    VisionLine,
    VisionPageResult,
)


def _line(
    text: str = "文字",
    confidence: float = 0.9,
    *,
    x: float = 0.1,
    y: float = 0.7,
) -> VisionLine:
    return VisionLine(text, confidence, BoundingBox(x, y, 0.2, 0.05))


def _box(**overrides: object) -> BoundingBox:
    values: dict[str, object] = {
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.4,
    }
    values.update(overrides)
    return BoundingBox(**values)  # type: ignore[arg-type]


def _summary(**overrides: object) -> OcrJobSummary:
    values: dict[str, object] = {
        "book_id": "book-1",
        "title": "扫描书",
        "status": "running",
        "total_pages": 8,
        "completed_pages": 2,
        "current_page": 3,
        "queue_position": None,
        "updated_at": "2026-07-14T12:00:00Z",
        "error": None,
        "estimated_remaining_seconds": 12,
    }
    values.update(overrides)
    return OcrJobSummary(**values)  # type: ignore[arg-type]


def test_vision_page_result_orders_top_to_bottom_then_left_to_right() -> None:
    result = VisionPageResult(
        schema_version=1,
        lines=(
            _line("右", 0.9, x=0.60, y=0.70),
            _line("下一行", 0.8, x=0.10, y=0.50),
            _line("左", 0.95, x=0.10, y=0.70),
        ),
    )

    assert result.ordered_text() == "左 右\n下一行"
    assert result.mean_confidence == pytest.approx(0.8833333333)


def test_vision_page_result_groups_lines_within_vertical_tolerance() -> None:
    result = VisionPageResult(
        schema_version=1,
        lines=(
            _line("右", x=0.6, y=0.6900),
            _line("左", x=0.1, y=0.7025),
            _line("下一行", x=0.1, y=0.6800),
        ),
    )

    assert result.ordered_text() == "左 右\n下一行"


def test_empty_vision_page_has_empty_text_and_zero_mean_confidence() -> None:
    result = VisionPageResult(schema_version=1, lines=())

    assert result.ordered_text() == ""
    assert result.mean_confidence == 0.0


@pytest.mark.parametrize("confidence", [-0.01, 1.01, nan, inf, -inf])
def test_vision_line_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        _line(confidence=confidence)


@pytest.mark.parametrize("text", ["", " ", "\t\n"])
def test_vision_line_rejects_blank_text(text: str) -> None:
    with pytest.raises(ValueError, match="text"):
        _line(text=text)


@pytest.mark.parametrize("coordinate", [nan, inf, -inf])
def test_bounding_box_rejects_nonfinite_coordinates(coordinate: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BoundingBox(coordinate, 0.1, 0.2, 0.05)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x", True),
        ("y", False),
        ("width", True),
        ("height", False),
        ("x", "0.1"),
        ("y", object()),
        ("width", "0.3"),
        ("height", object()),
    ],
)
def test_bounding_box_rejects_bool_and_nonnumeric_values(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        _box(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x", -0.01),
        ("x", 1.01),
        ("y", -0.01),
        ("y", 1.01),
        ("width", 0.0),
        ("width", -0.01),
        ("width", 1.01),
        ("height", 0.0),
        ("height", -0.01),
        ("height", 1.01),
    ],
)
def test_bounding_box_rejects_values_outside_normalized_ranges(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        _box(**{field: value})


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"x": 0.8, "width": 0.2000000001}, "width"),
        ({"y": 0.7, "height": 0.3000000001}, "height"),
    ],
)
def test_bounding_box_rejects_extents_beyond_normalized_image(
    values: dict[str, float], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        _box(**values)


def test_bounding_box_accepts_exact_normalized_image_bounds() -> None:
    assert BoundingBox(0.0, 0.0, 1.0, 1.0) == BoundingBox(
        x=0.0,
        y=0.0,
        width=1.0,
        height=1.0,
    )


@pytest.mark.parametrize("schema_version", [0, 2, -1])
def test_vision_page_result_requires_schema_version_one(schema_version: int) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        VisionPageResult(schema_version=schema_version, lines=())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("book_id", " "),
        ("title", "\n"),
        ("status", "unknown"),
        ("total_pages", 0),
        ("completed_pages", -1),
        ("completed_pages", 9),
        ("current_page", 0),
        ("current_page", 9),
        ("queue_position", 0),
        ("estimated_remaining_seconds", -1),
    ],
)
def test_ocr_job_summary_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _summary(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("updated_at", 123),
        ("updated_at", nan),
        ("updated_at", ""),
        ("updated_at", " \t"),
        ("error", 123),
        ("error", nan),
        ("error", ""),
        ("error", " \n"),
    ],
)
def test_ocr_job_summary_rejects_invalid_optional_text_metadata(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        _summary(**{field: value})


def test_ocr_job_summary_computes_deterministic_percent_complete() -> None:
    assert _summary(total_pages=3, completed_pages=1).percent_complete == 33.33
    assert _summary(total_pages=8, completed_pages=8, current_page=8).percent_complete == 100.0


@pytest.mark.parametrize(
    "status", ["queued", "running", "paused", "failed", "completed"]
)
def test_ocr_job_summary_accepts_supported_statuses(status: str) -> None:
    assert _summary(status=status).status == status


def test_ocr_job_summary_rejects_string_like_status() -> None:
    with pytest.raises(ValueError, match="status"):
        _summary(status=UserString("queued"))


def test_ocr_models_are_immutable_and_json_safe() -> None:
    line = _line()
    result = VisionPageResult(schema_version=1, lines=(line,))
    summary = _summary()

    json.dumps(asdict(result), ensure_ascii=False, allow_nan=False)
    json.dumps(asdict(summary), ensure_ascii=False, allow_nan=False)

    with pytest.raises(FrozenInstanceError):
        line.box.x = 0.2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        line.text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.schema_version = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        summary.status = "completed"  # type: ignore[misc]
