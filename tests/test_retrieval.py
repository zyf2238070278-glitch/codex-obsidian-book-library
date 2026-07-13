from collections.abc import Sequence

import numpy as np
import pytest

from book_agent.config import MAX_PREVIEWS
from book_agent.embeddings import NullEmbeddingProvider, encode_vector
from book_agent.models import Passage
from book_agent.retrieval import Retriever
from book_agent.storage import Database
from fakes import DeterministicEmbeddingProvider


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
    embedding: bytes | None = None,
) -> Passage:
    return Passage(
        passage_id=passage_id,
        book_id=book_id,
        ordinal=ordinal,
        text=text,
        section="测试章节",
        page_start=ordinal + 1,
        page_end=ordinal + 1,
        page_label=str(ordinal + 1),
        markdown_path=f"书库/20-解析文本/{book_id}.md",
        anchor=f"passage-{ordinal}",
        text_sha256=f"sha-{passage_id}",
        embedding=embedding,
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
        source_format="txt",
        content_sha256=f"hash-{book_id}",
        original_path=f"/books/{book_id}.txt",
    )
    db.replace_passages(book_id, passages)
    db.update_book_status(book_id, status)


def _vector(values: Sequence[float]) -> bytes:
    return encode_vector(np.asarray(values, dtype=np.float32))


def test_semantic_paraphrase_finds_the_intended_passage_without_keyword_overlap(
    db: Database,
) -> None:
    query = "行业为什么会周期性缺货"
    _add_book(
        db,
        "industry",
        [
            _passage(
                "cycle",
                "industry",
                0,
                "晶圆扩产耗时多年，需求突然上升时供给无法及时响应。",
                _vector([1.0, 0.0]),
            ),
            _passage(
                "marketing",
                "industry",
                1,
                "品牌投放通常会提升消费者认知。",
                _vector([0.0, 1.0]),
            ),
        ],
    )
    provider = DeterministicEmbeddingProvider({query: [1.0, 0.0]})

    assert db.keyword_search(query, 20) == []
    hybrid = Retriever(db, provider).search(query, mode="auto")
    quote_fallback = Retriever(db, provider).search(query, mode="quote")

    assert [hit.passage_id for hit in hybrid] == ["cycle", "marketing"]
    assert hybrid[0].score == pytest.approx(1 / 61)
    assert [hit.passage_id for hit in quote_fallback] == ["cycle", "marketing"]
    assert quote_fallback[0].score == pytest.approx(1.0)


def test_quote_mode_returns_exact_keyword_order_and_scores_without_encoding(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_book(
        db,
        "quotes",
        [
            _passage("exact-1", "quotes", 0, "企业会设置安全库存来缓冲波动。"),
            _passage("exact-2", "quotes", 1, "安全库存也会增加持有成本。"),
        ],
        status="keyword_only",
    )
    expected = db.keyword_search("安全库存", 20)
    calls: list[tuple[str, int, Sequence[str] | None]] = []
    real_keyword_search = db.keyword_search

    def recording_keyword_search(query, limit, book_ids=None):
        calls.append((query, limit, book_ids))
        return real_keyword_search(query, limit, book_ids)

    monkeypatch.setattr(db, "keyword_search", recording_keyword_search)
    provider = DeterministicEmbeddingProvider({})

    actual = Retriever(db, provider).search("安全库存", mode="quote")

    assert actual == expected
    assert calls == [("安全库存", 20, None)]
    assert provider.query_calls == []


def test_rrf_ties_are_deterministic_by_passage_id(db: Database) -> None:
    query = "供需周期"
    _add_book(
        db,
        "rrf",
        [
            _passage(
                "p-a", "rrf", 0, "供需周期会改变行业价格。", _vector([0.8, 0.6])
            ),
            _passage(
                "p-b", "rrf", 1, "供需周期会改变行业价格。", _vector([1.0, 0.0])
            ),
        ],
    )
    provider = DeterministicEmbeddingProvider({query: [1.0, 0.0]})

    first = Retriever(db, provider).search(query, mode="compare")
    second = Retriever(db, provider).search(query, mode="compare")

    assert [hit.passage_id for hit in first] == ["p-a", "p-b"]
    assert first == second
    assert first[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert first[1].score == pytest.approx(first[0].score)


class _UnavailableProvider:
    available = False

    def embed_query(self, text: str) -> np.ndarray:
        raise AssertionError("an unavailable provider must not be called")


class _RuntimeFailingProvider:
    available = True

    def embed_query(self, text: str) -> np.ndarray:
        raise RuntimeError("local model failed")


class _AvailabilityFailingProvider:
    @property
    def available(self) -> bool:
        raise RuntimeError("availability check failed")

    def embed_query(self, text: str) -> np.ndarray:
        raise AssertionError("availability failure must skip encoding")


@pytest.mark.parametrize(
    "provider",
    [_UnavailableProvider(), _RuntimeFailingProvider(), _AvailabilityFailingProvider()],
)
def test_semantic_failures_safely_degrade_to_keyword_results(
    db: Database, provider
) -> None:
    _add_book(
        db,
        "fallback",
        [_passage("keyword", "fallback", 0, "库存周期影响企业利润。")],
        status="keyword_only",
    )

    hits = Retriever(db, provider).search("库存周期", mode="explain")

    assert [hit.passage_id for hit in hits] == ["keyword"]


def test_keyboard_interrupt_from_embedding_is_not_swallowed(db: Database) -> None:
    _add_book(
        db,
        "interrupt",
        [_passage("keyword", "interrupt", 0, "库存周期影响企业利润。")],
        status="keyword_only",
    )

    class InterruptingProvider:
        available = True

        def embed_query(self, text: str) -> np.ndarray:
            raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        Retriever(db, InterruptingProvider()).search("库存周期", mode="auto")


def test_invalid_stored_embeddings_are_skipped_without_hiding_valid_hits(
    db: Database,
) -> None:
    query = "供应为何突然紧张"
    _add_book(
        db,
        "vectors",
        [
            _passage("valid", "vectors", 0, "扩产很慢而需求快速增长。", _vector([1, 0])),
            _passage("empty", "vectors", 1, "空向量。", b""),
            _passage("corrupt", "vectors", 2, "损坏向量。", b"bad"),
            _passage("nan", "vectors", 3, "非数向量。", _vector([np.nan, 0])),
            _passage("wrong", "vectors", 4, "维度不符。", _vector([1, 0, 0])),
            _passage("zero", "vectors", 5, "零向量。", _vector([0, 0])),
        ],
    )
    provider = DeterministicEmbeddingProvider({query: [1.0, 0.0]})

    hits = Retriever(db, provider).search(query, mode="quote")

    assert [hit.passage_id for hit in hits] == ["valid"]


@pytest.mark.parametrize(
    "query_vector",
    [
        np.empty(0, dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        np.array([np.nan, 0], dtype=np.float32),
        np.array([np.inf, 0], dtype=np.float32),
        np.array([0, 0], dtype=np.float32),
    ],
    ids=["empty", "two-dimensional", "nan", "infinite", "zero-norm"],
)
def test_invalid_query_vectors_disable_only_semantic_search(
    db: Database, query_vector: np.ndarray
) -> None:
    query = "没有关键词重叠"
    _add_book(
        db,
        "query-vector",
        [_passage("valid", "query-vector", 0, "另一种表达。", _vector([1, 0]))],
    )
    provider = DeterministicEmbeddingProvider({query: query_vector})

    assert Retriever(db, provider).search(query, mode="quote") == []


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_empty_queries_are_rejected(db: Database, query: str) -> None:
    with pytest.raises(ValueError, match="query|查询"):
        Retriever(db, NullEmbeddingProvider()).search(query)


@pytest.mark.parametrize("mode", ["semantic", "AUTO", ""])
def test_unknown_modes_are_rejected(db: Database, mode: str) -> None:
    with pytest.raises(ValueError, match="mode|模式"):
        Retriever(db, NullEmbeddingProvider()).search("库存", mode=mode)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["auto", "quote", "explain", "compare"])
def test_valid_modes_limit_clamping_and_book_filters(
    db: Database, mode: str
) -> None:
    _add_book(
        db,
        "book-1",
        [
            _passage(f"one-{ordinal:02d}", "book-1", ordinal, "库存周期影响利润。")
            for ordinal in range(MAX_PREVIEWS + 2)
        ],
        status="keyword_only",
    )
    _add_book(
        db,
        "book-2",
        [_passage("two", "book-2", 0, "库存周期影响利润。")],
        status="keyword_only",
    )
    retriever = Retriever(db, NullEmbeddingProvider())

    upper = retriever.search("库存周期", mode=mode, book_ids=["book-1"], limit=999)
    lower = retriever.search("库存周期", mode=mode, book_ids=["book-1"], limit=0)
    filtered = retriever.search("库存周期", mode=mode, book_ids=["book-2"])

    assert len(upper) == MAX_PREVIEWS
    assert len(lower) == 1
    assert [hit.book_id for hit in filtered] == ["book-2"]
    assert retriever.search("库存周期", mode=mode, book_ids=[]) == []
