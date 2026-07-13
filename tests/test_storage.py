import sqlite3

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
    page_start: int | None = 12,
    embedding: bytes | None = None,
) -> Passage:
    return Passage(
        passage_id=passage_id,
        book_id=book_id,
        ordinal=ordinal,
        text=text,
        section="第一章",
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

    hits = db.keyword_search("芯片行业", 5)

    assert [hit.passage_id for hit in hits] == ["passage-1"]


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

    assert [hit.passage_id for hit in db.keyword_search(query, 20)] == [expected]


def test_keyword_search_filters_book_ids_for_like_and_fts(db: Database) -> None:
    _book(db, "book-1", "hash-1")
    _book(db, "book-2", "hash-2")
    db.replace_passages("book-1", [_passage(book_id="book-1", text="芯片行业甲")])
    db.replace_passages(
        "book-2",
        [_passage("passage-2", book_id="book-2", text="芯片行业乙")],
    )

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
    assert db.status_counts() == {"processing": 1}


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
