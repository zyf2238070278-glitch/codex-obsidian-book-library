from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import fitz


TARGET_DPI = 300
MAXIMUM_LONG_EDGE_PIXELS = 12_000
SAFE_PAGE_PIXELS = 19_600_000
DPI_LADDER = (300, 240, 180, 144)
_SAFETY_MARGIN = 0.99
_MAX_RENDER_ATTEMPTS = 4


class RenderError(RuntimeError):
    """A PDF page could not be safely rasterized by a local OCR engine."""


@dataclass(frozen=True)
class RenderPlan:
    dpi: int
    scale: float
    pixel_width: int
    pixel_height: int


@dataclass(frozen=True)
class RenderedPage:
    pixmap: fitz.Pixmap
    dpi: int
    strategy: str
    width: int
    height: int


def _validate_dimensions(width_points: float, height_points: float) -> None:
    if (
        not math.isfinite(width_points)
        or not math.isfinite(height_points)
        or width_points <= 0.0
        or height_points <= 0.0
    ):
        raise RenderError("PDF page has invalid dimensions")


def _pixel_size(width_points: float, height_points: float, scale: float) -> tuple[int, int]:
    return math.ceil(width_points * scale), math.ceil(height_points * scale)


def plan_render(
    *,
    width_points: float,
    height_points: float,
    dpi: int = TARGET_DPI,
) -> RenderPlan:
    """Calculate a raster scale with headroom for PyMuPDF integer rounding."""

    _validate_dimensions(width_points, height_points)
    if type(dpi) is not int or dpi <= 0:
        raise RenderError("dpi must be a positive integer")
    scale = dpi / 72.0
    desired_long_edge = max(width_points, height_points) * scale
    desired_pixels = width_points * height_points * scale * scale
    if not math.isfinite(desired_long_edge) or not math.isfinite(desired_pixels):
        raise RenderError("PDF page dimensions are too large")
    bound_scale = min(
        1.0,
        MAXIMUM_LONG_EDGE_PIXELS / desired_long_edge,
        math.sqrt(SAFE_PAGE_PIXELS / desired_pixels),
    )
    scale *= bound_scale
    # Keep the requested DPI for ordinary pages.  When a cap is active, leave
    # real headroom so MuPDF's integer rounding cannot land exactly on a limit.
    if bound_scale < 1.0:
        scale *= _SAFETY_MARGIN
    for _ in range(_MAX_RENDER_ATTEMPTS):
        width, height = _pixel_size(width_points, height_points, scale)
        if (
            width > 0
            and height > 0
            and max(width, height) <= MAXIMUM_LONG_EDGE_PIXELS
            and width * height <= SAFE_PAGE_PIXELS
        ):
            return RenderPlan(
                dpi=max(1, round(scale * 72)),
                scale=scale,
                pixel_width=width,
                pixel_height=height,
            )
        scale *= _SAFETY_MARGIN
    raise RenderError("PDF page render exceeded the safe pixel limits")


class RenderPlanner:
    """Render private grayscale page images at bounded DPI variants."""

    def variants_for(self, pdf: Path) -> tuple[RenderPlan, ...]:
        if not isinstance(pdf, Path) or not pdf.is_absolute():
            raise RenderError("PDF path must be absolute")
        document: fitz.Document | None = None
        try:
            document = fitz.open(pdf)
            if document.needs_pass and not document.authenticate(""):
                raise RenderError("PDF is encrypted and cannot be opened")
            if len(document) == 0:
                raise RenderError("PDF has no pages")
            page = document.load_page(0)
            return tuple(
                plan_render(
                    width_points=float(page.rect.width),
                    height_points=float(page.rect.height),
                    dpi=dpi,
                )
                for dpi in DPI_LADDER
            )
        except RenderError:
            raise
        except (fitz.FileDataError, RuntimeError, ValueError, OverflowError) as exc:
            raise RenderError(f"could not inspect PDF page: {exc}") from exc
        finally:
            if document is not None:
                document.close()

    def render(
        self,
        pdf: Path,
        page_index: int,
        *,
        dpi: int = TARGET_DPI,
    ) -> RenderedPage:
        if type(page_index) is not int or page_index < 0:
            raise RenderError("page_index must be a nonnegative native integer")
        if not isinstance(pdf, Path) or not pdf.is_absolute():
            raise RenderError("PDF path must be absolute")
        document: fitz.Document | None = None
        try:
            document = fitz.open(pdf)
            if document.needs_pass and not document.authenticate(""):
                raise RenderError("PDF is encrypted and cannot be opened")
            if page_index >= len(document):
                raise RenderError("page_index is outside the PDF page range")
            page = document.load_page(page_index)
            plan = plan_render(
                width_points=float(page.rect.width),
                height_points=float(page.rect.height),
                dpi=dpi,
            )
            for _ in range(_MAX_RENDER_ATTEMPTS):
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(plan.scale, plan.scale),
                    colorspace=fitz.csGRAY,
                    alpha=False,
                )
                if (
                    pixmap.width > 0
                    and pixmap.height > 0
                    and max(pixmap.width, pixmap.height) <= MAXIMUM_LONG_EDGE_PIXELS
                    and pixmap.width * pixmap.height <= SAFE_PAGE_PIXELS
                ):
                    pixmap.set_dpi(plan.dpi, plan.dpi)
                    return RenderedPage(
                        pixmap=pixmap,
                        dpi=plan.dpi,
                        strategy=f"dpi_{dpi}",
                        width=pixmap.width,
                        height=pixmap.height,
                    )
                plan = plan_render(
                    width_points=float(page.rect.width),
                    height_points=float(page.rect.height),
                    dpi=max(1, int(plan.dpi * _SAFETY_MARGIN)),
                )
            raise RenderError("PDF page render exceeded the safe pixel limits")
        except RenderError:
            raise
        except (fitz.FileDataError, RuntimeError, ValueError, OverflowError, MemoryError) as exc:
            raise RenderError(f"could not render PDF page: {exc}") from exc
        finally:
            if document is not None:
                document.close()


__all__ = [
    "DPI_LADDER",
    "MAXIMUM_LONG_EDGE_PIXELS",
    "RenderError",
    "RenderPlan",
    "RenderPlanner",
    "RenderedPage",
    "SAFE_PAGE_PIXELS",
    "TARGET_DPI",
    "plan_render",
]
