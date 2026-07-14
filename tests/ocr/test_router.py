from __future__ import annotations

from pathlib import Path

import fitz

from book_agent.ocr.models import BoundingBox, OcrPageResult, VisionLine
from book_agent.ocr.router import LocalOcrRouter


def _pdf(path: Path) -> Path:
    document = fitz.open()
    document.new_page(width=100, height=100)
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
    tesseract = _ImageEngine(AssertionError("must not run"))
    decision = LocalOcrRouter(
        vision=_Vision(_result("Vision text", engine="apple_vision")),
        rapid=rapid,
        tesseract=tesseract,
    ).recognize_page(_pdf(tmp_path / "book.pdf"), page_index=0)

    assert decision.outcome.engine == "apple_vision"
    assert decision.text == "Vision text"
    assert not rapid.called and not tesseract.called


def test_router_uses_rapid_after_low_quality_vision(tmp_path: Path) -> None:
    rapid = _ImageEngine(_result("Rapid text", engine="rapidocr"))
    decision = LocalOcrRouter(
        vision=_Vision(_result("\ufffd\ufffd", engine="apple_vision")),
        rapid=rapid,
        tesseract=_ImageEngine(AssertionError("must not run")),
    ).recognize_page(_pdf(tmp_path / "book.pdf"), page_index=0)

    assert decision.outcome.engine == "rapidocr"
    assert decision.text == "Rapid text"
    assert rapid.called


def test_router_returns_skipped_after_all_local_strategies_fail(tmp_path: Path) -> None:
    decision = LocalOcrRouter(
        vision=_Vision(RuntimeError("vision unavailable")),
        rapid=_ImageEngine(RuntimeError("rapid unavailable")),
        tesseract=_ImageEngine(RuntimeError("tesseract unavailable")),
    ).recognize_page(_pdf(tmp_path / "book.pdf"), page_index=0)

    assert decision.outcome.status == "skipped"
    assert decision.text == ""
    assert "vision unavailable" in (decision.outcome.detail or "")
