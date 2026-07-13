import sqlite3
from pathlib import Path

import pytest

from book_agent.models import Passage
from book_agent.storage import Database


def _book(db: Database, book_id: str = "book-1", content_hash: str = "hash-1") -> None:
    db.create_book(
        book_id=book_id,
        title=f"Book {book_id}",
        author="Author",
        source_format="txt",
        content_sha256=content_hash,
        original_path=f"/books/{book_id}.txt",
        status="processing",
    )


def _passage(
    passage_id: str = "passage-1",
    *,
    book_id: str = "book-1",
    ordinal: int = 0,
    text: str = "芯片行业会反复经历缺货和库存过剩。",
    section: str | None = "第一章",
    page_start: int | None = 12,
    embedding: bytes | None = None,
) -> Passage:
    return Passage(
        passage_id=passage_id,
        book_id=book_id,
        ordinal=ordinal,
        text=text,
        section=section,
        page_start=page_start,
        page_end=page_start,
        page_label=str(page_start) if page_start is not None else None,
        markdown_path=f"书库/20-解析文本/{book_id}.md",
        anchor=f"passage-{ordinal}",
        text_sha256=f"sha-{passage_id}-{ordinal}",
        embedding=embedding,
    )


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "state" / "books.sqlite3")
    database.initialize()
    return database


def test_initialize_rejects_database_leaf_symlink_without_external_write(
    tmp_path: Path,
) -> None:
    database_directory = tmp_path / "project" / "data"
    database_directory.mkdir(parents=True)
    external_database = tmp_path / "outside.sqlite3"
    database_path = database_directory / "library.sqlite3"
    database_path.symlink_to(external_database)

    with pytest.raises(ValueError, match="symlink|symbolic|安全"):
        Database(database_path).initialize()

    assert database_path.is_symlink()
    assert not external_database.exists()


def test_initialize_rejects_symlinked_database_parent_without_external_write(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external_directory = tmp_path / "outside-data"
    external_directory.mkdir()
    (project / "data").symlink_to(external_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|symbolic|安全"):
        Database(project / "data" / "library.sqlite3").initialize()

    assert list(external_directory.iterdir()) == []


def test_keyword_search_falls_back_to_substring_for_two_chinese_characters(db: Database) -> None:
    _book(db)
    db.replace_passages("book-1", [_passage()])
    db.update_book_status("book-1", "keyword_only")

    hits = db.keyword_search("芯片", 5)

    assert [hit.passage_id for hit in hits] == ["passage-1"]
    assert hits[0].page_start == 12
    assert hits[0].score == 1.0
    assert db.get_book("book-1")["status"] == "keyword_only"


def test_replace_rejects_a_passage_for_another_book_without_writing(db: Database) -> None:
    _book(db)

    with pytest.raises(ValueError, match="book_id"):
        db.replace_passages("book-1", [_passage(book_id="book-2")])

    assert db.count_passages("book-1") == 0


def test_replace_rolls_back_passages_and_fts_when_a_later_insert_fails(db: Database) -> None:
    _book(db)
    db.replace_passages("book-1", [_passage(text="原有内容讨论芯片行业。")])
    db.update_book_status("book-1", "keyword_only")
    duplicate_id = _passage("duplicate", ordinal=1, text="新内容第一段。")
    conflicting = _passage("duplicate", ordinal=2, text="新内容第二段。")

    with pytest.raises(sqlite3.IntegrityError):
        db.replace_passages("book-1", [duplicate_id, conflicting])

    assert db.count_passages("book-1") == 1
    assert [hit.passage_id for hit in db.keyword_search("原有内容", 5)] == ["passage-1"]
    assert db.keyword_search("新内容", 5) == []


def test_fts_trigram_finds_a_three_plus_character_chinese_substring(db: Database) -> None:
    _book(db)
    db.replace_passages("book-1", [_passage(text="周期中的芯片行业经常出现波动。")])
    db.update_book_status("book-1", "keyword_only")

    hits = db.keyword_search("芯片行业", 5)

    assert [hit.passage_id for hit in hits] == ["passage-1"]


def test_whitespace_terms_all_must_match_text_and_exact_phrase_ranks_first(
    db: Database,
) -> None:
    _book(db)
    db.replace_passages(
        "book-1",
        [
            _passage(
                "separated-reversed",
                ordinal=0,
                text="库存过剩会压低价格；多年后半导体企业才削减产能。",
            ),
            _passage(
                "exact-phrase",
                ordinal=1,
                text="报告把半导体 库存过剩列为首要风险。",
            ),
            _passage("first-term-only", ordinal=2, text="半导体行业正在扩产。"),
            _passage("second-term-only", ordinal=3, text="库存过剩会压低价格。"),
        ],
    )
    db.update_book_status("book-1", "keyword_only")

    first = db.keyword_search("半导体 库存过剩", 20)
    second = db.keyword_search("半导体 库存过剩", 20)

    assert [hit.passage_id for hit in first] == [
        "exact-phrase",
        "separated-reversed",
    ]
    assert second == first
    assert [hit.passage_id for hit in db.keyword_search("半导体 库存过剩", 1)] == [
        "exact-phrase"
    ]


def test_whitespace_search_combines_short_cjk_like_and_long_fts_terms(
    db: Database,
) -> None:
    _book(db)
    db.replace_passages(
        "book-1",
        [
            _passage("both", ordinal=0, text="芯片价格最终因库存过剩而下跌。"),
            _passage("long-only", ordinal=1, text="库存过剩也影响其他行业。"),
        ],
    )
    db.update_book_status("book-1", "keyword_only")

    assert [
        hit.passage_id for hit in db.keyword_search("芯片 库存过剩", 20)
    ] == ["both"]


def test_keyword_search_merges_title_author_and_section_with_text_hits(
    db: Database,
) -> None:
    _book(db, "ready-book", "ready-hash")
    db.update_book_metadata("ready-book", "中国芯片产业周期", "李明研究组")
    db.replace_passages(
        "ready-book",
        [
            _passage(
                "text-and-title",
                book_id="ready-book",
                ordinal=0,
                text="芯片产业周期会影响企业利润。",
                section="供给冲击",
            ),
            _passage(
                "title-only",
                book_id="ready-book",
                ordinal=1,
                text="这一段没有查询词。",
                section="需求侧库存周期",
            ),
        ],
    )
    db.update_book_status("ready-book", "ready")

    _book(db, "keyword-book", "keyword-hash")
    db.update_book_metadata("keyword-book", "产业观察", "王芳研究组")
    db.replace_passages(
        "keyword-book",
        [
            _passage(
                "author-hit",
                book_id="keyword-book",
                text="作者访谈摘录。",
                section="市场回顾",
            )
        ],
    )
    db.update_book_status("keyword-book", "keyword_only")

    assert [
        hit.passage_id for hit in db.keyword_search("芯片 周期", 20)
    ] == ["text-and-title", "title-only"]
    assert [hit.passage_id for hit in db.keyword_search("李明", 20)] == [
        "text-and-title",
        "title-only",
    ]
    assert [
        hit.passage_id for hit in db.keyword_search("需求 库存周期", 20)
    ] == ["title-only"]
    assert [hit.passage_id for hit in db.keyword_search("王芳研究组", 20)] == [
        "author-hit"
    ]


def test_metadata_search_respects_all_terms_status_filters_limits_and_dedup(
    db: Database,
) -> None:
    for book_id, status in (
        ("visible", "keyword_only"),
        ("partial", "ready"),
        ("hidden", "failed"),
    ):
        _book(db, book_id, f"hash-{book_id}")
        title = "资本 市场周期" if book_id != "partial" else "资本观察"
        db.update_book_metadata(book_id, title, "研究作者")
        db.replace_passages(
            book_id,
            [
                _passage(
                    f"{book_id}-0",
                    book_id=book_id,
                    ordinal=0,
                    text="资本 市场周期会影响估值。" if book_id == "visible" else "无关原文。",
                    section="资本 市场周期" if book_id == "visible" else "普通章节",
                ),
                _passage(
                    f"{book_id}-1",
                    book_id=book_id,
                    ordinal=1,
                    text="第二段无关原文。",
                    section="普通章节",
                ),
            ],
        )
        db.update_book_status(book_id, status)

    first = db.keyword_search("资本 市场周期", 20)
    second = db.keyword_search("资本 市场周期", 20)

    assert [hit.passage_id for hit in first] == ["visible-0", "visible-1"]
    assert second == first
    assert [
        hit.passage_id
        for hit in db.keyword_search("资本 市场周期", 20, ["visible"])
    ] == ["visible-0", "visible-1"]
    assert db.keyword_search("资本 市场周期", 20, ["partial"]) == []
    assert db.keyword_search("资本 市场周期", 20, []) == []
    assert [hit.passage_id for hit in db.keyword_search("资本 市场周期", 1)] == [
        "visible-0"
    ]


def test_fts_operator_characters_are_literal_and_cannot_broaden_results(
    db: Database,
) -> None:
    _book(db)
    db.replace_passages(
        "book-1",
        [
            _passage("literal", ordinal=0, text='代号 foo"bar 与 baz*qux 分开放置。'),
            _passage("one-term", ordinal=1, text='这里只出现 foo"bar。'),
            _passage("unrelated", ordinal=2, text="普通内容。"),
        ],
    )
    db.update_book_status("book-1", "keyword_only")

    assert [
        hit.passage_id for hit in db.keyword_search('foo"bar baz*qux', 20)
    ] == ["literal"]
    assert db.keyword_search('missing" OR *', 20) == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [("%", "percent"), ("_", "underscore"), ("\\", "backslash")],
)
def test_short_like_queries_treat_wildcards_and_escape_as_literals(
    db: Database, query: str, expected: str
) -> None:
    _book(db)
    db.replace_passages(
        "book-1",
        [
            _passage("percent", ordinal=0, text="增长达到 100%"),
            _passage("underscore", ordinal=1, text="字段名是 foo_bar"),
            _passage("backslash", ordinal=2, text=r"目录是 C:\books"),
            _passage("plain", ordinal=3, text="这里没有特殊符号"),
        ],
    )
    db.update_book_status("book-1", "keyword_only")

    assert [hit.passage_id for hit in db.keyword_search(query, 20)] == [expected]


def test_keyword_search_filters_book_ids_for_like_and_fts(db: Database) -> None:
    _book(db, "book-1", "hash-1")
    _book(db, "book-2", "hash-2")
    db.replace_passages("book-1", [_passage(book_id="book-1", text="芯片行业甲")])
    db.replace_passages(
        "book-2",
        [_passage("passage-2", book_id="book-2", text="芯片行业乙")],
    )
    db.update_book_status("book-1", "keyword_only")
    db.update_book_status("book-2", "keyword_only")

    assert [h.book_id for h in db.keyword_search("芯片", 5, ["book-2"])] == ["book-2"]
    assert [h.book_id for h in db.keyword_search("芯片行业", 5, ["book-1"])] == ["book-1"]
    assert db.keyword_search("芯片", 5, []) == []


def test_get_passages_preserves_requested_order(db: Database) -> None:
    _book(db)
    db.replace_passages(
        "book-1",
        [
            _passage("passage-1", ordinal=0, text="甲段"),
            _passage("passage-2", ordinal=1, text="乙段"),
            _passage("passage-3", ordinal=2, text="丙段"),
        ],
    )
    db.update_book_status("book-1", "keyword_only")

    hits = db.get_passages(["passage-3", "missing", "passage-1"])

    assert [hit.passage_id for hit in hits] == ["passage-3", "passage-1"]


def test_embedding_roundtrip_neighbors_and_book_helpers(db: Database) -> None:
    _book(db)
    db.replace_passages(
        "book-1",
        [
            _passage("passage-1", ordinal=0, text="甲段"),
            _passage("passage-2", ordinal=1, text="乙段"),
            _passage("passage-3", ordinal=2, text="丙段"),
        ],
    )
    db.set_embeddings({"passage-1": b"one", "passage-3": b"three"})
    db.update_book_status("book-1", "ready")

    embedded = list(db.iter_embeddings(["book-1"]))

    assert [(hit.passage_id, value) for hit, value in embedded] == [
        ("passage-1", b"one"),
        ("passage-3", b"three"),
    ]
    assert [hit.passage_id for hit in db.get_neighbors("book-1", 1, 1)] == [
        "passage-1",
        "passage-2",
        "passage-3",
    ]
    assert db.get_ordinal("passage-2") == 1
    assert db.get_ordinal("missing") is None
    assert db.find_book_by_hash("hash-1")["book_id"] == "book-1"
    assert db.status_counts() == {"ready": 1}


@pytest.mark.parametrize("status", ["processing", "failed", "needs_ocr"])
def test_nonsearchable_book_statuses_hide_even_existing_passages_and_embeddings(
    db: Database, status: str
) -> None:
    _book(db)
    db.replace_passages("book-1", [_passage(embedding=b"unsafe-vector")])
    if status != "processing":
        db.update_book_status("book-1", status)

    assert db.count_passages("book-1") == 1
    assert db.keyword_search("芯片", 5) == []
    assert db.keyword_search("芯片行业", 5) == []
    assert list(db.iter_embeddings()) == []
    assert db.get_passages(["passage-1"]) == []
    assert db.get_neighbors("book-1", 0, 1) == []
    assert db.get_ordinal("passage-1") is None


def test_book_updates_listing_and_connection_configuration(db: Database) -> None:
    _book(db)
    db.update_book_metadata("book-1", title="Updated", author=None)
    db.update_book_status(
        "book-1",
        "failed",
        error="parse failed",
        parsed_path="/parsed/book-1.md",
    )

    book = db.get_book("book-1")
    assert book["title"] == "Updated"
    assert book["author"] is None
    assert book["error"] == "parse failed"
    assert book["parsed_path"] == "/parsed/book-1.md"
    assert [row["book_id"] for row in db.list_books("failed")] == ["book-1"]
    with db.connect() as connection:
        assert isinstance(connection.execute("SELECT 1").fetchone(), sqlite3.Row)
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_book_queries_return_plain_json_serializable_dicts(db: Database) -> None:
    _book(db)

    by_id = db.get_book("book-1")
    by_hash = db.find_book_by_hash("hash-1")
    listed = db.list_books()

    assert type(by_id) is dict
    assert type(by_hash) is dict
    assert [type(book) for book in listed] == [dict]


def test_internal_database_operations_explicitly_close_connections(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_connection = db.connect()

    class TrackedConnection:
        def __init__(self) -> None:
            self.closed = False

        def __enter__(self):
            raw_connection.__enter__()
            return self

        def __exit__(self, *args):
            return raw_connection.__exit__(*args)

        def execute(self, *args, **kwargs):
            return raw_connection.execute(*args, **kwargs)

        def close(self) -> None:
            self.closed = True
            raw_connection.close()

    tracked = TrackedConnection()
    monkeypatch.setattr(db, "connect", lambda: tracked)

    db.get_book("missing")

    assert tracked.closed
