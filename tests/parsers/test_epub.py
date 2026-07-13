from pathlib import Path

from ebooklib import epub
import pytest

from book_agent.parsers.base import DocumentParseError
from book_agent.parsers.epub import parse_epub
from book_agent.parsers.registry import parse_document


def _write_epub(
    path: Path,
    chapters: list[tuple[str, str]],
    *,
    spine: list[str] | None = None,
    title: str | None = "Synthetic book",
    author: str | None = "Synthetic author",
) -> None:
    book = epub.EpubBook()
    book.set_identifier(f"urn:test:{path.stem}")
    book.set_language("zh")
    if title is not None:
        book.set_title(title)
    if author is not None:
        book.add_author(author)

    items = {}
    for filename, content in chapters:
        item = epub.EpubHtml(title=filename, file_name=filename, lang="zh")
        item.content = content
        book.add_item(item)
        items[filename] = item

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + [items[name] for name in (spine or items)]
    epub.write_epub(path, book)


def test_parse_document_routes_epub_and_follows_spine_order(tmp_path: Path) -> None:
    path = tmp_path / "测试 EPUB.epub"
    _write_epub(
        path,
        [
            ("chapter2.xhtml", "<h2>第二章</h2><p>库存下降。</p>"),
            ("chapter1.xhtml", "<h1>第一章</h1><p>需求增长。</p>"),
        ],
        spine=["chapter1.xhtml", "chapter2.xhtml"],
        title="测试 EPUB",
        author="作者甲",
    )

    book = parse_document(path)

    assert book.title == "测试 EPUB"
    assert book.author == "作者甲"
    assert book.source_format == "epub"
    assert [unit.section for unit in book.units] == ["第一章", "第二章"]
    assert [unit.text for unit in book.units] == ["需求增长。", "库存下降。"]
    assert all(
        (unit.page_start, unit.page_end, unit.page_label) == (None, None, None)
        for unit in book.units
    )


def test_explicit_empty_title_and_author_override_epub_metadata(tmp_path: Path) -> None:
    path = tmp_path / "metadata.epub"
    _write_epub(
        path,
        [("chapter.xhtml", "<h1>正文</h1><p>有效内容。</p>")],
        title="元数据标题",
        author="元数据作者",
    )

    book = parse_epub(path, title="", author="")

    assert book.title == ""
    assert book.author == ""


def test_blank_epub_metadata_falls_back_to_stem_and_none(tmp_path: Path) -> None:
    path = tmp_path / "fallback-name.epub"
    _write_epub(
        path,
        [("chapter.xhtml", "<p>有效内容。</p>")],
        title="   ",
        author="  ",
    )

    book = parse_epub(path)

    assert book.title == "fallback-name"
    assert book.author is None


def test_epub_body_extraction_removes_noise_and_preserves_dom_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dom-order.epub"
    _write_epub(
        path,
        [
            (
                "chapter.xhtml",
                """
                <h3>结构化内容</h3>
                <style>.hidden { display: none; }</style>
                <script>脚本噪声</script>
                <p>段落一 <span>重点</span></p>
                <nav><p>导航噪声</p></nav>
                <ul>
                  <li>before<ul><li>inner</li></ul>after</li>
                </ul>
                <p>段落二</p>
                """,
            )
        ],
    )

    book = parse_epub(path)

    assert len(book.units) == 1
    assert book.units[0].section == "结构化内容"
    assert book.units[0].text == (
        "段落一 重点\n\nbefore\n\ninner\n\nafter\n\n段落二"
    )
    assert "脚本噪声" not in book.units[0].text
    assert "导航噪声" not in book.units[0].text


def test_epub_skips_navigation_and_empty_spine_documents(tmp_path: Path) -> None:
    path = tmp_path / "skip-empty.epub"
    _write_epub(
        path,
        [
            ("empty.xhtml", "<h1>只有标题</h1><script>噪声</script>"),
            ("body.xhtml", "<h1>正文章</h1><p>唯一正文。</p>"),
        ],
        spine=["empty.xhtml", "body.xhtml"],
    )

    book = parse_epub(path)

    assert [(unit.section, unit.text) for unit in book.units] == [
        ("正文章", "唯一正文。")
    ]


def test_epub_without_body_text_is_a_document_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "empty-book.epub"
    _write_epub(
        path,
        [("empty.xhtml", "<h1>只有标题</h1><nav>目录</nav>")],
    )

    with pytest.raises(DocumentParseError, match="empty-book.epub") as error:
        parse_epub(path)

    assert error.value.__cause__ is None


@pytest.mark.parametrize("fixture", ["corrupt", "missing", "directory"])
def test_epub_wraps_read_errors_with_filename_and_cause(
    tmp_path: Path, fixture: str
) -> None:
    path = tmp_path / f"{fixture}.epub"
    if fixture == "corrupt":
        path.write_bytes(b"not an EPUB archive")
    elif fixture == "directory":
        path.mkdir()

    with pytest.raises(
        DocumentParseError,
        match=rf"(?i){path.name}.*(?:corrupt|damage|encrypt|DRM)",
    ) as error:
        parse_epub(path)

    assert error.value.__cause__ is not None
