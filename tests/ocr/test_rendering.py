from __future__ import annotations

from pathlib import Path

import fitz

from book_agent.ocr.rendering import (
    SAFE_PAGE_PIXELS,
    RenderPlanner,
    plan_render,
)


def _write_pdf(path: Path, *, width: float = 612, height: float = 792) -> Path:
    document = fitz.open()
    document.new_page(width=width, height=height)
    document.save(path)
    document.close()
    return path


def test_safe_scale_leaves_rounding_margin_at_pixel_cap() -> None:
    plan = plan_render(width_points=1983, height_points=2972)

    assert plan.dpi < 300
    assert plan.pixel_width * plan.pixel_height <= SAFE_PAGE_PIXELS


def test_renderer_exposes_dpi_fallback_ladder(tmp_path: Path) -> None:
    variants = list(
        RenderPlanner().variants_for(_write_pdf(tmp_path / "huge.pdf"))
    )

    assert [item.dpi for item in variants] == [300, 240, 180, 144]
