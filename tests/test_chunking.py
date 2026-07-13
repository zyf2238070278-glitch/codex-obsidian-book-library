import hashlib
from pathlib import Path

import pytest

from book_agent.chunking import chunk_book
from book_agent.models import ParsedBook, SourceUnit


def _book(*units: SourceUnit, source_format: str = "txt") -> ParsedBook:
    return ParsedBook(
        title="测试书",
        author=None,
        source_format=source_format,
        units=tuple(units),
    )


def test_chunking_is_stable_and_derives_ids_from_final_text(tmp_path: Path) -> None:
    parsed = _book(
        SourceUnit(text="第一段内容。\n\n第二段内容。", section="第一章"),
        SourceUnit(text="第三段内容。", section="第二章"),
    )
    markdown_path = tmp_path / "测试书.md"

    first = chunk_book("book-42", parsed, markdown_path, target_chars=12, max_chars=18)
    second = chunk_book("book-42", parsed, markdown_path, target_chars=12, max_chars=18)

    assert first == second
    assert [passage.ordinal for passage in first] == list(range(len(first)))
    assert all(passage.book_id == "book-42" for passage in first)
    assert all(passage.markdown_path == str(markdown_path) for passage in first)
    for passage in first:
        digest = hashlib.sha256(passage.text.encode("utf-8")).hexdigest()
        expected_id = hashlib.sha256(
            f"book-42:{passage.ordinal}:{digest}".encode("utf-8")
        ).hexdigest()[:24]
        assert passage.text_sha256 == digest
        assert passage.passage_id == expected_id
        assert passage.anchor == expected_id


def test_ordinary_chunks_preserve_paragraph_order_and_stay_under_maximum() -> None:
    paragraphs = ["甲" * 7, "乙" * 7, "丙" * 7, "丁" * 7]
    parsed = _book(SourceUnit(text="\n\n".join(paragraphs)))

    passages = chunk_book("book", parsed, "book.md", target_chars=15, max_chars=18)

    assert passages
    assert all(0 < len(passage.text) <= 18 for passage in passages)
    assert "\n\n".join(passage.text for passage in passages) == "\n\n".join(
        paragraphs
    )
    assert all(
        left.text != right.text for left, right in zip(passages, passages[1:])
    )


def test_blank_line_separator_is_counted_exactly_at_maximum() -> None:
    parsed = _book(SourceUnit(text="甲乙丙丁\n\n戊己庚辛\n\n壬癸"))

    passages = chunk_book("book", parsed, "book.md", target_chars=10, max_chars=10)

    assert [passage.text for passage in passages] == ["甲乙丙丁\n\n戊己庚辛", "壬癸"]
    assert len(passages[0].text) == 10


@pytest.mark.parametrize(
    "text",
    [
        "这是第一句中文内容。这是第二句中文内容！这是第三句中文内容？最后一段没有句号",
        (
            "This is the first English sentence. This is the second English sentence! "
            "This final clause is deliberately long and has no nearby terminator"
        ),
    ],
)
def test_one_oversized_paragraph_is_split_without_losing_text(text: str) -> None:
    parsed = _book(SourceUnit(text=text, section="长段落"))

    passages = chunk_book("book", parsed, "book.md", target_chars=28, max_chars=36)

    assert len(passages) > 1
    assert all(0 < len(passage.text) <= 36 for passage in passages)
    assert "".join(passage.text for passage in passages) == text
    assert all(passage.section == "长段落" for passage in passages)


def test_long_paragraph_prefers_newline_or_sentence_boundaries() -> None:
    text = "甲" * 16 + "。" + "乙" * 15 + "\n" + "丙" * 30

    passages = chunk_book(
        "book", _book(SourceUnit(text=text)), "book.md", target_chars=20, max_chars=24
    )

    assert passages[0].text.endswith("。")
    assert any(passage.text.endswith("\n") for passage in passages)
    assert "".join(passage.text for passage in passages) == text


def test_section_change_forces_a_passage_boundary() -> None:
    parsed = _book(
        SourceUnit(text="第一章甲。\n\n第一章乙。", section="第一章"),
        SourceUnit(text="第二章甲。\n\n第二章乙。", section="第二章"),
    )

    passages = chunk_book("book", parsed, "book.md", target_chars=100, max_chars=120)

    assert [(passage.section, passage.text) for passage in passages] == [
        ("第一章", "第一章甲。\n\n第一章乙。"),
        ("第二章", "第二章甲。\n\n第二章乙。"),
    ]


def test_pdf_chunks_can_span_adjacent_pages_and_preserve_metadata_order() -> None:
    parsed = _book(
        SourceUnit(
            text="第三页内容。",
            section="共同章节",
            page_start=3,
            page_end=3,
            page_label="iii",
        ),
        SourceUnit(
            text="第四页内容。",
            section="共同章节",
            page_start=4,
            page_end=4,
            page_label="4",
        ),
        SourceUnit(
            text="第五页新章。",
            section="下一章",
            page_start=5,
            page_end=5,
            page_label="5",
        ),
        source_format="pdf",
    )

    passages = chunk_book("book", parsed, "book.md", target_chars=100, max_chars=120)

    assert [passage.text for passage in passages] == [
        "第三页内容。\n\n第四页内容。",
        "第五页新章。",
    ]
    assert (
        passages[0].page_start,
        passages[0].page_end,
        passages[0].page_label,
    ) == (3, 4, "iii")
    assert (
        passages[1].page_start,
        passages[1].page_end,
        passages[1].page_label,
    ) == (5, 5, "5")


@pytest.mark.parametrize(
    ("target_chars", "max_chars"),
    [(0, 10), (-1, 10), (11, 10)],
)
def test_invalid_chunking_configuration_is_rejected(
    target_chars: int, max_chars: int
) -> None:
    with pytest.raises(ValueError):
        chunk_book(
            "book",
            _book(SourceUnit(text="正文")),
            "book.md",
            target_chars=target_chars,
            max_chars=max_chars,
        )


def test_empty_body_returns_no_passages() -> None:
    parsed = _book(
        SourceUnit(text="  \n\n\t", section="空章"),
        SourceUnit(text="\r\n\r\n"),
    )

    assert chunk_book("book", parsed, "book.md") == []
