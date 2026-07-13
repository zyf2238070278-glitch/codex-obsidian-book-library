from pathlib import Path

import pytest

from book_agent.parsers.base import DocumentParseError, NeedsOcrError
from book_agent.parsers.registry import parse_document
from book_agent.parsers.text import parse_markdown, parse_txt


def test_parse_error_types_are_value_errors() -> None:
    assert issubclass(DocumentParseError, ValueError)
    assert issubclass(NeedsOcrError, DocumentParseError)


def test_parse_markdown_uses_atx_headings_as_sections(tmp_path: Path) -> None:
    path = tmp_path / "投资.md"
    path.write_text(
        "# 周期\n\n库存上升。\n\n## 风险\n\n需求下降。",
        encoding="utf-8",
    )

    book = parse_markdown(path)

    assert book.title == "投资"
    assert book.author is None
    assert book.source_format == "md"
    assert [unit.section for unit in book.units] == ["周期", "风险"]
    assert [unit.text for unit in book.units] == ["库存上升。", "需求下降。"]
    assert all(
        (unit.page_start, unit.page_end, unit.page_label) == (None, None, None)
        for unit in book.units
    )


def test_parse_txt_preserves_paragraph_order_and_single_line_breaks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes("第一行\r\n第二行\r\n  \t\r\n第三段".encode())

    book = parse_txt(path, title="显式标题", author="作者甲")

    assert book.title == "显式标题"
    assert book.author == "作者甲"
    assert book.source_format == "txt"
    assert [unit.text for unit in book.units] == ["第一行\n第二行", "第三段"]
    assert [unit.section for unit in book.units] == [None, None]
    assert all(
        (unit.page_start, unit.page_end, unit.page_label) == (None, None, None)
        for unit in book.units
    )


@pytest.mark.parametrize("parser,suffix", [(parse_txt, ".txt"), (parse_markdown, ".md")])
def test_parsers_reject_whitespace_only_files(tmp_path: Path, parser, suffix: str) -> None:
    path = tmp_path / f"empty{suffix}"
    path.write_text(" \t\r\n\r\n", encoding="utf-8")

    with pytest.raises(DocumentParseError, match="empty"):
        parser(path)


@pytest.mark.parametrize("parser,suffix", [(parse_txt, ".txt"), (parse_markdown, ".md")])
def test_parsers_report_invalid_utf8(tmp_path: Path, parser, suffix: str) -> None:
    path = tmp_path / f"broken{suffix}"
    path.write_bytes(b"valid prefix\xff")

    with pytest.raises(DocumentParseError, match=r"broken\.(?:txt|md).*UTF-8"):
        parser(path)


def test_parse_markdown_rejects_a_document_with_only_headings(tmp_path: Path) -> None:
    path = tmp_path / "outline.md"
    path.write_text("# 第一章\n\n## 第二节\n", encoding="utf-8")

    with pytest.raises(DocumentParseError, match="outline.md"):
        parse_markdown(path)


def test_parse_markdown_rejects_only_empty_atx_heading_markers(tmp_path: Path) -> None:
    path = tmp_path / "empty-headings.md"
    path.write_text("#\n\n##   \n", encoding="utf-8")

    with pytest.raises(DocumentParseError, match="empty-headings.md"):
        parse_markdown(path)


def test_parse_document_routes_a_mixed_case_markdown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "Guide.MD"
    path.write_text("# Intro\r\n\r\nBody text.", encoding="utf-8")

    book = parse_document(path, title="Guide Title", author="A. Writer")

    assert book.title == "Guide Title"
    assert book.author == "A. Writer"
    assert book.source_format == "md"
    assert [(unit.section, unit.text) for unit in book.units] == [
        ("Intro", "Body text.")
    ]


def test_fenced_code_heading_text_is_content_and_closing_hashes_are_removed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "code.md"
    path.write_text(
        "# Examples ###\n\n"
        "Before.\n\n"
        "```python\n"
        "# not heading\n"
        "print('ok')\n"
        "```\n\n"
        "After.",
        encoding="utf-8",
    )

    book = parse_markdown(path)

    assert [(unit.section, unit.text) for unit in book.units] == [
        ("Examples", "Before."),
        ("Examples", "```python\n# not heading\nprint('ok')\n```"),
        ("Examples", "After."),
    ]


def test_an_over_indented_fence_marker_does_not_close_the_code_block(
    tmp_path: Path,
) -> None:
    path = tmp_path / "indented-fence.md"
    path.write_text(
        "# Outer\n\n"
        "```text\n"
        "line\n"
        "    ```\n"
        "# still code\n"
        "```\n\n"
        "Tail.",
        encoding="utf-8",
    )

    book = parse_markdown(path)

    assert [(unit.section, unit.text) for unit in book.units] == [
        ("Outer", "```text\nline\n    ```\n# still code\n```"),
        ("Outer", "Tail."),
    ]


def test_utf8_bom_is_ignored_for_markdown_content_and_empty_checks(
    tmp_path: Path,
) -> None:
    content_path = tmp_path / "bom.md"
    content_path.write_bytes(b"\xef\xbb\xbf# Intro\n\nBody.")
    empty_path = tmp_path / "bom-empty.md"
    empty_path.write_bytes(b"\xef\xbb\xbf \n")

    book = parse_markdown(content_path)

    assert [(unit.section, unit.text) for unit in book.units] == [
        ("Intro", "Body.")
    ]
    with pytest.raises(DocumentParseError, match="empty"):
        parse_markdown(empty_path)


def test_a_heading_change_flushes_the_current_markdown_paragraph(tmp_path: Path) -> None:
    path = tmp_path / "compact.md"
    path.write_text("# One\nfirst line\n## Two\nsecond line", encoding="utf-8")

    book = parse_markdown(path)

    assert [(unit.section, unit.text) for unit in book.units] == [
        ("One", "first line"),
        ("Two", "second line"),
    ]


def test_hash_tag_without_a_space_is_not_a_markdown_heading(tmp_path: Path) -> None:
    path = tmp_path / "tag.md"
    path.write_text("#tag\nstill body", encoding="utf-8")

    book = parse_markdown(path)

    assert [(unit.section, unit.text) for unit in book.units] == [
        (None, "#tag\nstill body")
    ]


@pytest.mark.parametrize("filename", ["book.docx", "README"])
def test_parse_document_rejects_unsupported_or_missing_extensions(
    tmp_path: Path, filename: str
) -> None:
    path = tmp_path / filename
    path.write_text("content", encoding="utf-8")

    with pytest.raises(DocumentParseError, match="unsupported|不支持"):
        parse_document(path)
