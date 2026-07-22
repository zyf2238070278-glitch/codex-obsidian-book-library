from __future__ import annotations

from book_agent.ocr.quality import assess_page


def test_nonblank_image_with_empty_ocr_requires_fallback() -> None:
    verdict = assess_page(text="", lines=(), image_ink_ratio=0.08)

    assert verdict.accepted is False
    assert verdict.reason == "unexpected_empty_text"


def test_blank_image_with_empty_ocr_is_a_blank_outcome() -> None:
    verdict = assess_page(text="", lines=(), image_ink_ratio=0.0001)

    assert verdict.accepted is True
    assert verdict.outcome is not None
    assert verdict.outcome.status == "blank"


def test_control_character_heavy_text_is_rejected() -> None:
    verdict = assess_page(text="\x00\x01\ufffd\ufffd", lines=(), image_ink_ratio=0.1)

    assert verdict.accepted is False


def test_terminal_empty_visual_page_is_image_only() -> None:
    verdict = assess_page(
        text="",
        lines=(),
        image_ink_ratio=0.08,
        terminal=True,
    )

    assert verdict.accepted is True
    assert verdict.outcome is not None
    assert verdict.outcome.status == "image_only"
