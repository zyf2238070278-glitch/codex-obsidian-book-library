from collections.abc import Sequence
import json

import pytest

from book_agent.config import MAX_EVIDENCE_TOKENS, MAX_FULL_PASSAGES
from book_agent.embeddings import NullEmbeddingProvider
from book_agent.models import Passage
from book_agent.retrieval import Retriever, estimate_tokens
from book_agent.storage import Database


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "books.sqlite3")
    database.initialize()
    return database


def _passage(
    passage_id: str,
    book_id: str,
    ordinal: int,
    text: str,
    *,
    section: str | None = "测试章节",
    page_start: int | None = None,
    page_end: int | None = None,
    page_label: str | None = None,
) -> Passage:
    return Passage(
        passage_id=passage_id,
        book_id=book_id,
        ordinal=ordinal,
        text=text,
        section=section,
        page_start=page_start,
        page_end=page_end,
        page_label=page_label,
        markdown_path=f"书库/20-解析文本/{book_id}.md",
        anchor=f"passage-{ordinal}",
        text_sha256=f"sha-{passage_id}",
    )


def _add_book(
    db: Database,
    book_id: str,
    passages: Sequence[Passage],
    *,
    status: str = "ready",
) -> None:
    db.create_book(
        book_id=book_id,
        title=f"Book {book_id}",
        author="Author",
        source_format="pdf",
        content_sha256=f"hash-{book_id}",
        original_path=f"/books/{book_id}.pdf",
    )
    db.replace_passages(book_id, passages)
    db.update_book_status(book_id, status)


def _retriever(db: Database) -> Retriever:
    return Retriever(db, NullEmbeddingProvider())


def test_expands_overlapping_neighbors_in_context_order_without_duplicates(
    db: Database,
) -> None:
    _add_book(
        db,
        "ordered",
        [
            _passage(f"p{number}", "ordered", number - 1, f"第 {number} 段。")
            for number in range(1, 9)
        ],
    )

    evidence = _retriever(db).get_passages(["p2", "p3"], neighbor_count=1)

    assert [item["passage_id"] for item in evidence] == ["p1", "p2", "p3", "p4"]
    assert len({item["passage_id"] for item in evidence}) == len(evidence)
    assert all(item["untrusted_content"] is True for item in evidence)
    assert len(evidence) <= MAX_FULL_PASSAGES
    assert sum(item["estimated_tokens"] for item in evidence) <= MAX_EVIDENCE_TOKENS


def test_duplicate_selected_ids_are_deduplicated_before_unknown_validation(
    db: Database,
) -> None:
    _add_book(
        db,
        "duplicates",
        [
            _passage("p1", "duplicates", 0, "前文"),
            _passage("p2", "duplicates", 1, "正文"),
            _passage("p3", "duplicates", 2, "后文"),
        ],
    )

    evidence = _retriever(db).get_passages(["p2", "p2"], neighbor_count=1)

    assert [item["passage_id"] for item in evidence] == ["p1", "p2", "p3"]


def test_rejects_empty_too_many_and_unknown_passage_ids(db: Database) -> None:
    _add_book(db, "known", [_passage("known", "known", 0, "正文")])
    retriever = _retriever(db)

    with pytest.raises(ValueError, match="至少|不能为空|passage"):
        retriever.get_passages([])
    with pytest.raises(ValueError, match="6|最多"):
        retriever.get_passages(["known"] * (MAX_FULL_PASSAGES + 1))
    with pytest.raises(ValueError, match="missing"):
        retriever.get_passages(["known", "known", "missing"])


@pytest.mark.parametrize("passage_ids", ["abc", b"abc", None])
def test_passage_ids_rejects_string_bytes_and_non_sequence_containers(
    db: Database, passage_ids: object
) -> None:
    with pytest.raises(ValueError, match="passage_ids|ID"):
        _retriever(db).get_passages(passage_ids)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_id", [1, None, "", " \t\n"])
def test_passage_ids_rejects_non_string_and_blank_elements(
    db: Database, invalid_id: object
) -> None:
    with pytest.raises(ValueError, match="passage_ids|ID"):
        _retriever(db).get_passages([invalid_id])  # type: ignore[list-item]


@pytest.mark.parametrize("passage_ids", [["known"], ("known",)])
def test_passage_ids_keeps_normal_list_and_tuple_inputs(
    db: Database, passage_ids: Sequence[str]
) -> None:
    _add_book(db, "known-inputs", [_passage("known", "known-inputs", 0, "正文")])

    evidence = _retriever(db).get_passages(passage_ids, neighbor_count=0)

    assert [item["passage_id"] for item in evidence] == ["known"]


@pytest.mark.parametrize("neighbor_count", [True, False, 1.0, -1, 2, None, "1"])
def test_neighbor_count_must_be_the_strict_integer_zero_or_one(
    db: Database, neighbor_count: object
) -> None:
    _add_book(db, "neighbor", [_passage("p1", "neighbor", 0, "正文")])

    with pytest.raises(ValueError, match="neighbor|邻"):
        _retriever(db).get_passages(["p1"], neighbor_count=neighbor_count)  # type: ignore[arg-type]


def test_normalized_duplicate_text_keeps_selected_citation_over_earlier_neighbor(
    db: Database,
) -> None:
    _add_book(
        db,
        "text-dedup",
        [
            _passage("neighbor-copy", "text-dedup", 0, "  相同\n正文\t内容  "),
            _passage("selected", "text-dedup", 1, "相同正文内容"),
            _passage("following", "text-dedup", 2, "不同的后文"),
        ],
    )

    evidence = _retriever(db).get_passages(["selected"], neighbor_count=1)

    assert [item["passage_id"] for item in evidence] == ["selected", "following"]


def test_duplicate_selected_text_keeps_the_first_selected_citation(
    db: Database,
) -> None:
    _add_book(
        db,
        "selected-dedup",
        [
            _passage("first", "selected-dedup", 0, "同一 正文"),
            _passage("second", "selected-dedup", 1, "同一\n正文"),
        ],
    )

    evidence = _retriever(db).get_passages(["first", "second"], neighbor_count=0)

    assert [item["passage_id"] for item in evidence] == ["first"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("abcd", 1),
        ("abcde", 2),
        ("ab \n", 1),
        ("中文。", 3),
        ("日本語", 3),
        ("한글", 2),
        ("，。", 2),
        ("😀🚀", 4),
        ("中abcd😀", 4),
    ],
)
def test_estimate_tokens_is_conservative_for_ascii_cjk_and_emoji(
    text: str, expected: int
) -> None:
    assert estimate_tokens(text) == expected


@pytest.mark.parametrize(
    ("fitting_text", "overflowing_text"),
    [
        ("界" * 7999, "界" * 8000),
        ("a" * 31996, "a" * 32000),
    ],
    ids=["cjk", "english"],
)
def test_long_neighbors_stop_at_the_hard_token_cap(
    db: Database, fitting_text: str, overflowing_text: str
) -> None:
    _add_book(
        db,
        "long-neighbors",
        [
            _passage("fits", "long-neighbors", 0, fitting_text),
            _passage("selected", "long-neighbors", 1, "中"),
            _passage("overflows", "long-neighbors", 2, overflowing_text),
        ],
    )

    evidence = _retriever(db).get_passages(["selected"], neighbor_count=1)

    assert [item["passage_id"] for item in evidence] == ["fits", "selected"]
    assert sum(item["estimated_tokens"] for item in evidence) == MAX_EVIDENCE_TOKENS


def test_large_neighbor_cannot_displace_explicitly_selected_passages(
    db: Database,
) -> None:
    _add_book(
        db,
        "selected-first",
        [
            _passage("before", "selected-first", 0, "前文"),
            _passage("selected-1", "selected-first", 1, "甲"),
            _passage("huge-neighbor", "selected-first", 2, "界" * 7999),
            _passage("selected-2", "selected-first", 3, "乙"),
            _passage("after", "selected-first", 4, "后文"),
        ],
    )

    evidence = _retriever(db).get_passages(
        ["selected-1", "selected-2"], neighbor_count=1
    )

    returned_ids = [item["passage_id"] for item in evidence]
    assert "selected-1" in returned_ids
    assert "selected-2" in returned_ids
    assert "huge-neighbor" not in returned_ids
    assert sum(item["estimated_tokens"] for item in evidence) <= MAX_EVIDENCE_TOKENS


@pytest.mark.parametrize(
    "texts",
    [
        ["界" * (MAX_EVIDENCE_TOKENS + 1)],
        ["甲" * 4001, "乙" * 4001],
    ],
    ids=["one-passage", "selected-total"],
)
def test_selected_evidence_over_budget_requires_multiple_calls(
    db: Database, texts: list[str]
) -> None:
    _add_book(
        db,
        "over-budget",
        [
            _passage(f"selected-{index}", "over-budget", index, text)
            for index, text in enumerate(texts)
        ],
    )

    with pytest.raises(ValueError, match="8000|拆|多次"):
        _retriever(db).get_passages(
            [f"selected-{index}" for index in range(len(texts))], neighbor_count=0
        )


def test_evidence_fields_locations_page_ranges_and_obsidian_links(
    db: Database,
) -> None:
    _add_book(
        db,
        "locations",
        [
            _passage(
                "single-page",
                "locations",
                0,
                "单页正文",
                section="第一章",
                page_start=7,
                page_end=7,
                page_label="7",
            ),
            _passage(
                "page-range",
                "locations",
                1,
                "跨页正文",
                section="第二章",
                page_start=8,
                page_end=10,
                page_label="8–10",
            ),
            _passage(
                "no-location",
                "locations",
                2,
                "无位置正文",
                section=None,
            ),
        ],
    )

    evidence = _retriever(db).get_passages(
        ["single-page", "page-range", "no-location"], neighbor_count=0
    )

    assert [item["location"] for item in evidence] == [
        "第一章 · PDF 页 7",
        "第二章 · PDF 页 8–10",
        "no-location",
    ]
    assert evidence[0]["obsidian_link"] == (
        "[[书库/20-解析文本/locations.md#^passage-0]]"
    )
    assert evidence[1]["page_label"] == "8–10"
    assert set(evidence[0]) == {
        "passage_id",
        "book_id",
        "title",
        "text",
        "section",
        "page_start",
        "page_end",
        "page_label",
        "location",
        "obsidian_link",
        "untrusted_content",
        "estimated_tokens",
    }
    assert "ordinal" not in evidence[0]
    assert "embedding" not in evidence[0]
    json.dumps(evidence, ensure_ascii=False)


@pytest.mark.parametrize("status", ["processing", "failed"])
def test_nonsearchable_passage_id_is_reported_as_unknown(
    db: Database, status: str
) -> None:
    _add_book(
        db,
        f"book-{status}",
        [_passage(f"passage-{status}", f"book-{status}", 0, "不可检索正文")],
        status=status,
    )

    with pytest.raises(ValueError, match=f"passage-{status}"):
        _retriever(db).get_passages([f"passage-{status}"])


def test_selected_passage_with_missing_ordinal_is_not_silently_dropped(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_book(db, "ordinal-race", [_passage("selected", "ordinal-race", 0, "正文")])
    monkeypatch.setattr(db, "get_ordinal", lambda passage_id: None)

    with pytest.raises(ValueError, match="selected"):
        _retriever(db).get_passages(["selected"])


def test_selected_passage_missing_from_neighbor_query_is_not_silently_dropped(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_book(db, "neighbor-race", [_passage("selected", "neighbor-race", 0, "正文")])
    monkeypatch.setattr(db, "get_neighbors", lambda book_id, ordinal, distance: [])

    with pytest.raises(ValueError, match="selected"):
        _retriever(db).get_passages(["selected"])


def test_selected_books_keep_group_order_across_books(db: Database) -> None:
    _add_book(
        db,
        "book-a",
        [
            _passage("a1", "book-a", 0, "A 前文"),
            _passage("a2", "book-a", 1, "A 后文"),
        ],
    )
    _add_book(
        db,
        "book-b",
        [
            _passage("b1", "book-b", 0, "B 前文"),
            _passage("b2", "book-b", 1, "B 后文"),
        ],
    )

    evidence = _retriever(db).get_passages(["b2", "a1"], neighbor_count=1)

    assert [item["passage_id"] for item in evidence] == ["b1", "b2", "a1", "a2"]
