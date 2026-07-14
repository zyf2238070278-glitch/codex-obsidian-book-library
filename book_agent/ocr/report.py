from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from book_agent.config import AppPaths


def write_ocr_report(
    paths: AppPaths,
    *,
    book_id: str,
    title: str,
    skipped_pages: Iterable[Mapping[str, object]],
) -> Path:
    """Write bounded operational metadata, never OCR text, into Obsidian."""

    if not paths.ocr_reports.is_dir():
        paths.ocr_reports.mkdir(parents=True, exist_ok=True)
    lines = [f"# OCR 处理报告：{title}", "", f"书籍 ID：`{book_id}`", ""]
    rows = list(skipped_pages)
    if not rows:
        lines.extend(["## 结果", "", "没有跳过的 PDF 页面。", ""])
    else:
        lines.extend(["## 跳过页面", ""])
        for row in rows:
            page = row.get("page_number")
            label = row.get("page_label")
            strategy = row.get("strategy")
            detail = str(row.get("detail") or "未知本地 OCR 错误")[:500]
            position = f"PDF 第 {page} 页"
            if isinstance(label, str) and label.strip():
                position += f"（页码标签：{label.strip()}）"
            lines.extend(
                [f"- {position}：{detail}（策略：`{strategy}`）"]
            )
        lines.append("")
    report = paths.ocr_reports / f"{book_id}-OCR处理报告.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


__all__ = ["write_ocr_report"]
