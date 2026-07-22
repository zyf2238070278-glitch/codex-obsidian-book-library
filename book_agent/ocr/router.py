from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from book_agent.ocr.models import OcrPageOutcome
from book_agent.ocr.quality import assess_page
from book_agent.ocr.rendering import RenderPlanner


class VisionPageEngine(Protocol):
    def recognize_page(self, pdf: Path, *, page_index: int) -> Any: ...


class ImagePageEngine(Protocol):
    def recognize_image(self, image: Path) -> Any: ...


@dataclass(frozen=True)
class OcrPageDecision:
    outcome: OcrPageOutcome
    text: str
    mean_confidence: float | None
    attempts: tuple[str, ...]


def _result_fields(result: Any) -> tuple[str, tuple[Any, ...], float | None]:
    ordered_text = getattr(result, "ordered_text", None)
    if not callable(ordered_text):
        raise ValueError("OCR engine result must provide ordered_text")
    text = ordered_text()
    lines = getattr(result, "lines", None)
    confidence = getattr(result, "mean_confidence", None)
    if type(text) is not str or type(lines) is not tuple:
        raise ValueError("OCR engine returned an invalid result")
    if confidence is not None and type(confidence) not in (int, float):
        raise ValueError("OCR engine returned an invalid mean confidence")
    return text, lines, None if confidence is None else float(confidence)


class LocalOcrRouter:
    """Bounded local fallback order: Apple Vision, RapidOCR, then Light OCR."""

    def __init__(
        self,
        *,
        vision: VisionPageEngine,
        rapid: ImagePageEngine,
        light: ImagePageEngine | None = None,
        renderer: RenderPlanner | None = None,
    ) -> None:
        self._vision = vision
        self._rapid = rapid
        self._light = light
        self._renderer = RenderPlanner() if renderer is None else renderer

    @staticmethod
    def _accepted(
        result: Any,
        *,
        engine: str,
        strategy: str,
        ink_ratio: float,
        attempts: tuple[str, ...],
    ) -> OcrPageDecision | None:
        text, lines, confidence = _result_fields(result)
        verdict = assess_page(text, lines, ink_ratio)
        if not verdict.accepted:
            return None
        outcome = verdict.outcome or OcrPageOutcome("recognized", engine, strategy)
        return OcrPageDecision(outcome, text.strip(), confidence, attempts)

    @staticmethod
    def _ink_ratio(samples: bytes) -> float:
        if not samples:
            return 0.0
        return sum(value < 245 for value in samples) / len(samples)

    def _render_image(self, pdf: Path, page_index: int) -> tuple[tempfile.TemporaryDirectory[str], Path, float]:
        rendered = self._renderer.render(pdf, page_index)
        directory = tempfile.TemporaryDirectory(prefix="book-ocr-page-")
        image = Path(directory.name) / "page.png"
        rendered.pixmap.save(image)
        return directory, image, self._ink_ratio(rendered.pixmap.samples)

    def recognize_page(self, pdf: Path, *, page_index: int) -> OcrPageDecision:
        attempts: list[str] = []
        errors: list[str] = []
        saw_nonempty_result = False
        try:
            attempts.append("standard:apple_vision")
            vision_result = self._vision.recognize_page(pdf, page_index=page_index)
            vision_text, _, _ = _result_fields(vision_result)
            saw_nonempty_result = bool(vision_text.strip())
            accepted = self._accepted(
                vision_result,
                engine="apple_vision",
                strategy="standard",
                ink_ratio=0.1,
                attempts=tuple(attempts),
            )
            if accepted is not None:
                return accepted
            errors.append("apple_vision: low quality")
        except Exception as exc:
            errors.append(f"apple_vision: {str(exc)[:160] or exc.__class__.__name__}")

        directory: tempfile.TemporaryDirectory[str] | None = None
        try:
            directory, image, ink_ratio = self._render_image(pdf, page_index)
            engines: tuple[tuple[str, ImagePageEngine, str], ...] = (
                (("rapidocr", self._rapid, "enhanced"),)
                if self._light is None
                else (
                    ("rapidocr", self._rapid, "enhanced"),
                    ("light_ocr", self._light, "enhanced"),
                )
            )
            terminal_empty = False
            for engine_name, engine, strategy in engines:
                try:
                    attempts.append(f"{strategy}:{engine_name}")
                    result = engine.recognize_image(image)
                    text, _, _ = _result_fields(result)
                    terminal_empty = not bool(text.strip())
                    saw_nonempty_result = saw_nonempty_result or not terminal_empty
                    accepted = self._accepted(
                        result,
                        engine=engine_name,
                        strategy=strategy,
                        ink_ratio=ink_ratio,
                        attempts=tuple(attempts),
                    )
                    if accepted is not None:
                        return accepted
                    errors.append(f"{engine_name}: low quality")
                except Exception as exc:
                    terminal_empty = False
                    errors.append(f"{engine_name}: {str(exc)[:160] or exc.__class__.__name__}")
            if terminal_empty and not saw_nonempty_result:
                verdict = assess_page("", (), ink_ratio, terminal=True)
                if verdict.accepted and verdict.outcome is not None:
                    return OcrPageDecision(
                        verdict.outcome,
                        "",
                        None,
                        tuple(attempts),
                    )
        except Exception as exc:
            errors.append(f"render: {str(exc)[:160] or exc.__class__.__name__}")
        finally:
            if directory is not None:
                directory.cleanup()
        detail = "; ".join(errors)[:500] or "all local OCR engines failed"
        return OcrPageDecision(
            OcrPageOutcome("skipped", None, "all_local_engines_failed", detail),
            "",
            None,
            tuple(attempts),
        )


__all__ = ["LocalOcrRouter", "OcrPageDecision"]
