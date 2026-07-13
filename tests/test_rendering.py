import json
from pathlib import Path

import pytest

import book_agent.rendering as rendering
from book_agent.models import ParsedBook, Passage


def _parsed() -> ParsedBook:
    return ParsedBook(
        title='周期："繁荣"与萧条',
        author="作者甲",
        source_format="pdf",
        units=(),
    )


def _passage(
    ordinal: int,
    text: str,
    *,
    section: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> Passage:
    passage_id = f"passage-{ordinal}"
    return Passage(
        passage_id=passage_id,
        book_id="book-1",
        ordinal=ordinal,
        text=text,
        section=section,
        page_start=page_start,
        page_end=page_end,
        page_label=None,
        markdown_path="书库/20-解析文本/book-1.md",
        anchor=passage_id,
        text_sha256=f"digest-{ordinal}",
    )


def test_render_writes_safe_frontmatter_source_and_unescaped_chinese(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "book.md"
    source_file = tmp_path / '原始："书".pdf'

    result = rendering.render_parsed_book(
        destination,
        "book-1",
        _parsed(),
        source_file,
        [_passage(0, "中文原文。")],
    )

    content = destination.read_text(encoding="utf-8")
    assert result == destination
    assert content.startswith("---\n")
    assert f"book_id: {json.dumps('book-1', ensure_ascii=False)}" in content
    assert f"title: {json.dumps(_parsed().title, ensure_ascii=False)}" in content
    assert 'source_format: "pdf"' in content
    assert f"source_file: {json.dumps(str(source_file), ensure_ascii=False)}" in content
    assert "source_type: original" in content
    assert f"# {_parsed().title}" in content
    assert "中文原文。" in content
    assert "\\u" not in content


def test_render_includes_locations_page_ranges_and_each_anchor_once(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "book.md"
    passages = [
        _passage(0, "第一页原文。", section="第一章", page_start=7, page_end=7),
        _passage(1, "跨页原文。", section="第二章", page_start=8, page_end=10),
        _passage(2, "无位置原文。"),
    ]

    rendering.render_parsed_book(
        destination, "book-1", _parsed(), "incoming/book.pdf", passages
    )

    content = destination.read_text(encoding="utf-8")
    assert "## 第一章 · PDF 页 7" in content
    assert "## 第二章 · PDF 页 8–10" in content
    assert "## 段落 3" in content
    anchor_offsets = []
    for passage in passages:
        marker = f"^{passage.anchor}"
        assert content.count(marker) == 1
        anchor_offsets.append(content.index(marker))
    assert anchor_offsets == sorted(anchor_offsets)
    assert content.index("第一页原文。") < content.index("跨页原文。")
    assert content.index("跨页原文。") < content.index("无位置原文。")


def test_render_atomically_replaces_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "book.md"
    destination.write_text("旧文件内容", encoding="utf-8")

    rendering.render_parsed_book(
        destination,
        "book-1",
        _parsed(),
        "source.pdf",
        [_passage(0, "完整的新文件。")],
    )

    content = destination.read_text(encoding="utf-8")
    assert content != "旧文件内容"
    assert "完整的新文件。" in content
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_replace_failure_preserves_old_file_and_cleans_unique_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "book.md"
    destination.write_text("不可破坏的旧文件", encoding="utf-8")

    def fail_replace(source: str | Path, target: str | Path) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(rendering.os, "replace", fail_replace)

    with pytest.raises(OSError, match="publish failed"):
        rendering.render_parsed_book(
            destination,
            "book-1",
            _parsed(),
            "source.pdf",
            [_passage(0, "不能发布的新文件。")],
        )

    assert destination.read_text(encoding="utf-8") == "不可破坏的旧文件"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
