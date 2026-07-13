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


def test_epub_extracts_common_body_containers_and_direct_body_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "common-containers.epub"
    _write_epub(
        path,
        [
            (
                "chapter.xhtml",
                """
                <h1>容器章节</h1>
                直属   正文
                <div>div <span>内容</span></div>
                <blockquote>引用   内容</blockquote>
                <table><tr><th>表头   文本</th><td>单元格 <strong>文本</strong></td></tr></table>
                <pre>预格式
                    内容</pre>
                <dl><dt>术语   名称</dt><dd>术语
                    定义</dd></dl>
                <figure><figcaption>图注   文本</figcaption></figure>
                """,
            )
        ],
    )

    book = parse_epub(path)

    assert len(book.units) == 1
    assert book.units[0].section == "容器章节"
    assert book.units[0].text == (
        "直属 正文\n\n"
        "div 内容\n\n"
        "引用 内容\n\n"
        "表头 文本\n\n"
        "单元格 文本\n\n"
        "预格式 内容\n\n"
        "术语 名称\n\n"
        "术语 定义\n\n"
        "图注 文本"
    )
    assert "容器章节" not in book.units[0].text


def test_epub_common_containers_preserve_order_without_nested_duplicates_or_noise(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested-containers.epub"
    _write_epub(
        path,
        [
            (
                "second.xhtml",
                "<h2>第二节</h2><div>末章   正文</div>",
            ),
            (
                "first.xhtml",
                """
                <h1>第一节</h1>
                <style>.hidden { display: none; } 样式噪声</style>
                <script>脚本噪声</script>
                <nav><div>导航噪声</div></nav>
                <div>
                  开头
                  <blockquote>
                    引用 <span>只出现一次</span>
                    <div>内部   内容</div>
                    结尾
                  </blockquote>
                  收尾
                </div>
                """,
            ),
        ],
        spine=["first.xhtml", "second.xhtml"],
    )

    book = parse_epub(path)

    assert [unit.section for unit in book.units] == ["第一节", "第二节"]
    assert [unit.text for unit in book.units] == [
        "开头\n\n引用 只出现一次\n\n内部 内容\n\n结尾\n\n收尾",
        "末章 正文",
    ]
    assert book.units[0].text.count("只出现一次") == 1
    assert "第一节" not in book.units[0].text
    assert "第二节" not in book.units[1].text
    assert "样式噪声" not in book.units[0].text
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
