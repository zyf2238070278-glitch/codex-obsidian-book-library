from __future__ import annotations

from pathlib import Path

import fitz

from book_agent.ocr.models import BoundingBox, OcrPageResult, VisionLine
from book_agent.ocr.router import LocalOcrRouter


def _pdf(path: Path, *, visual: bool = False) -> Path:
    document = fitz.open()
    page = document.new_page(width=100, height=100)
    if visual:
        page.draw_rect(fitz.Rect(10, 10, 90, 90), color=(0, 0, 0), fill=(0, 0, 0))
    document.save(path)
    document.close()
    return path


def _result(text: str, *, engine: str) -> OcrPageResult:
    return OcrPageResult(
        engine=engine,
        lines=(VisionLine(text, 0.9, BoundingBox(0.1, 0.1, 0.3, 0.1)),),
    )


class _Vision:
    def __init__(self, result: OcrPageResult | Exception) -> None:
        self.result = result

    def recognize_page(self, pdf: Path, *, page_index: int) -> OcrPageResult:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _ImageEngine:
    def __init__(self, result: OcrPageResult | Exception) -> None:
        self.result = result
        self.called = False

    def recognize_image(self, image: Path) -> OcrPageResult:
        self.called = True
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_router_stops_after_accepted_vision_result(tmp_path: Path) -> None:
    rapid = _ImageEngine(AssertionError("must not run"))
    decision = LocalOcrRouter(
        vision=_Vision(_result("Vision text", engine="apple_vision")),
        rapid=rapid,
    ).recognize_page(_pdf(tmp_path / "book.pdf"), page_index=0)

    assert decision.outcome.engine == "apple_vision"
    assert decision.text == "Vision text"
    assert not rapid.called


def test_router_uses_rapid_after_low_quality_vision(tmp_path: Path) -> None:
    rapid = _ImageEngine(_result("Rapid text", engine="rapidocr"))
    decision = LocalOcrRouter(
        vision=_Vision(_result("\ufffd\ufffd", engine="apple_vision")),
        rapid=rapid,
    ).recognize_page(_pdf(tmp_path / "book.pdf"), page_index=0)

    assert decision.outcome.engine == "rapidocr"
    assert decision.text == "Rapid text"
    assert rapid.called


def test_router_returns_skipped_after_all_local_strategies_fail(tmp_path: Path) -> None:
    decision = LocalOcrRouter(
        vision=_Vision(RuntimeError("vision unavailable")),
        rapid=_ImageEngine(RuntimeError("rapid unavailable")),
    ).recognize_page(_pdf(tmp_path / "book.pdf"), page_index=0)

    assert decision.outcome.status == "skipped"
    assert decision.text == ""
    assert "vision unavailable" in (decision.outcome.detail or "")


def test_router_uses_light_after_rapid_low_quality(tmp_path: Path) -> None:
    rapid = _ImageEngine(_result("\ufffd\ufffd", engine="rapidocr"))
    light = _ImageEngine(_result("Light text", engine="light_ocr"))

    decision = LocalOcrRouter(
        vision=_Vision(RuntimeError("vision unavailable")),
        rapid=rapid,
        light=light,
    ).recognize_page(_pdf(tmp_path / "book.pdf"), page_index=0)

    assert decision.outcome.engine == "light_ocr"
    assert decision.text == "Light text"
    assert decision.attempts == (
        "standard:apple_vision",
        "enhanced:rapidocr",
        "enhanced:light_ocr",
    )
    assert rapid.called is True
    assert light.called is True


def test_router_stops_before_light_when_rapid_is_accepted(tmp_path: Path) -> None:
    light = _ImageEngine(AssertionError("must not run"))

    decision = LocalOcrRouter(
        vision=_Vision(RuntimeError("vision unavailable")),
        rapid=_ImageEngine(_result("Rapid text", engine="rapidocr")),
        light=light,
    ).recognize_page(_pdf(tmp_path / "book.pdf"), page_index=0)

    assert decision.outcome.engine == "rapidocr"
    assert light.called is False


def test_router_classifies_visual_page_with_three_empty_results_as_image_only(
    tmp_path: Path,
) -> None:
    empty_rapid = OcrPageResult(engine="rapidocr", lines=())
    empty_light = OcrPageResult(engine="light_ocr", lines=())

    decision = LocalOcrRouter(
        vision=_Vision(OcrPageResult(engine="apple_vision", lines=())),
        rapid=_ImageEngine(empty_rapid),
        light=_ImageEngine(empty_light),
    ).recognize_page(_pdf(tmp_path / "book.pdf", visual=True), page_index=0)

    assert decision.outcome.status == "image_only"
    assert decision.outcome.strategy == "no_text_expected"
    assert decision.text == ""
