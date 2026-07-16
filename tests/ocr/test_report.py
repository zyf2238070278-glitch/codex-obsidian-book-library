from __future__ import annotations

from pathlib import Path

from book_agent.config import AppPaths
from book_agent.ocr.report import write_ocr_report


def test_report_lists_only_skipped_physical_pages(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    paths.ocr_reports.mkdir(parents=True)

    report = write_ocr_report(
        paths,
        book_id="a" * 24,
        title="测试书",
        skipped_pages=[
            {
                "page_number": 4,
                "page_label": "iv",
                "strategy": "all_local_engines_failed",
                "detail": "render failed",
            }
        ],
    )

    text = report.read_text(encoding="utf-8")
    assert report.parent == paths.ocr_reports
    assert "PDF 第 4 页" in text
    assert "render failed" in text
    assert "原文" not in text
