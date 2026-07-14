import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import book_agent.indexing as indexing_module
from book_agent.config import AppPaths
from book_agent.embeddings import NullEmbeddingProvider, decode_vector
from book_agent.indexing import BookIndexer, IndexResult
from book_agent.models import ParsedBook, SourceUnit
from book_agent.storage import Database


class _ReadyEmbeddingProvider:
    available = True

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [
                [ordinal, ordinal + 0.5, ordinal + 1.0]
                for ordinal, _ in enumerate(texts)
            ],
            dtype=np.float64,
        )


class _FailingEmbeddingProvider(_ReadyEmbeddingProvider):
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("模型暂时不可用")


class _WrongCountEmbeddingProvider(_ReadyEmbeddingProvider):
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.empty((0, 3), dtype=np.float32)


class _InvalidEmbeddingProvider(_ReadyEmbeddingProvider):
    def __init__(self, case: str) -> None:
        self.case = case

    def embed_passages(self, texts: list[str]):
        count = len(texts)
        if self.case == "empty":
            return np.empty((count, 0), dtype=np.float32)
        if self.case == "nan":
            vectors = np.ones((count, 3), dtype=np.float32)
            vectors[0, 0] = np.nan
            return vectors
        if self.case == "inf":
            vectors = np.ones((count, 3), dtype=np.float32)
            vectors[0, 0] = np.inf
            return vectors
        if self.case == "two-dimensional-row":
            return [np.ones((1, 3), dtype=np.float32) for _ in texts]
        if self.case == "mixed-dimensions":
            return [
                np.ones(2 + ordinal, dtype=np.float32)
                for ordinal, _ in enumerate(texts)
            ]
        raise AssertionError(f"unknown case: {self.case}")


class _InterruptingEmbeddingProvider(_ReadyEmbeddingProvider):
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        raise KeyboardInterrupt("operator cancelled embedding")


@pytest.fixture
def app(tmp_path: Path) -> tuple[AppPaths, Database, _ReadyEmbeddingProvider]:
    paths = AppPaths.from_root(tmp_path / "app")
    database = Database(paths.database)
    database.initialize()
    paths.originals.mkdir(parents=True)
    original = paths.originals / "scan.pdf"
    original.write_bytes(b"source")
    database.create_book(
        book_id="a" * 24,
        title="导入前标题",
        author=None,
        source_format="pdf",
        content_sha256="f" * 64,
        original_path=str(original),
    )
    return paths, database, _ReadyEmbeddingProvider()


def _register_book(
    paths: AppPaths,
    database: Database,
    *,
    book_id: str,
    source_format: str = "pdf",
) -> Path:
    paths.originals.mkdir(parents=True, exist_ok=True)
    original = paths.originals / f"{book_id}.{source_format}"
    original.write_bytes(b"source")
    database.create_book(
        book_id=book_id,
        title="导入前标题",
        author=None,
        source_format=source_format,
        content_sha256=book_id.ljust(64, "f"),
        original_path=str(original),
    )
    return original


def _two_passage_book() -> ParsedBook:
    return ParsedBook(
        title="多段测试",
        author=None,
        source_format="txt",
        units=(
            SourceUnit("第一段" + "甲" * 1600),
            SourceUnit("第二段" + "乙" * 1600),
        ),
    )


def test_indexer_publishes_markdown_passages_and_ready_status(
    app: tuple[AppPaths, Database, _ReadyEmbeddingProvider],
) -> None:
    paths, database, provider = app
    parsed = ParsedBook(
        title="扫描测试",
        author="作者",
        source_format="pdf",
        units=(
            SourceUnit(
                "第一页正文足够长。",
                page_start=1,
                page_end=1,
            ),
        ),
    )

    result = BookIndexer(paths, database, provider).index_parsed_book(
        book_id="a" * 24,
        parsed=parsed,
        original_path=paths.originals / "scan.pdf",
    )

    assert result.status == "ready"
    assert database.count_passages("a" * 24) == 1
    assert result.parsed_path is not None
    assert "PDF 页 1" in Path(result.parsed_path).read_text(encoding="utf-8")
    book = database.get_book("a" * 24)
    assert book is not None
    assert book["title"] == "扫描测试"
    assert book["author"] == "作者"
    assert book["parsed_path"] == result.parsed_path
    assert result.parsed_path == str(
        (paths.parsed / ("a" * 24) / "正文.md").absolute()
    )
    hits = database.keyword_search("第一页正文", 5)
    assert len(hits) == 1
    assert hits[0].page_start == hits[0].page_end == 1
    assert hits[0].markdown_path == f"书库/20-解析文本/{'a' * 24}/正文.md"
    embedded = list(database.iter_embeddings(["a" * 24]))
    assert len(embedded) == result.passage_count == 1
    np.testing.assert_array_equal(
        decode_vector(embedded[0][1]),
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )


def test_index_result_is_frozen_json_safe_and_strictly_native() -> None:
    result = IndexResult(
        status="keyword_only",
        parsed_path="/vault/正文.md",
        passage_count=2,
        error="语义模型未启用",
        message="可使用关键词检索",
    )

    serialized = result.to_dict()

    assert type(serialized) is dict
    assert json.loads(json.dumps(serialized, ensure_ascii=False)) == serialized
    with pytest.raises(FrozenInstanceError):
        result.status = "ready"  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        {
            "status": True,
            "parsed_path": None,
            "passage_count": 0,
            "error": None,
            "message": "message",
        },
        {
            "status": "ready",
            "parsed_path": Path("正文.md"),
            "passage_count": 0,
            "error": None,
            "message": "message",
        },
        {
            "status": "ready",
            "parsed_path": None,
            "passage_count": True,
            "error": None,
            "message": "message",
        },
        {
            "status": "ready",
            "parsed_path": None,
            "passage_count": -1,
            "error": None,
            "message": "message",
        },
        {
            "status": "ready",
            "parsed_path": None,
            "passage_count": 0,
            "error": RuntimeError("not JSON safe"),
            "message": "message",
        },
        {
            "status": "ready",
            "parsed_path": None,
            "passage_count": 0,
            "error": None,
            "message": 3,
        },
    ],
)
def test_index_result_rejects_non_native_or_invalid_values(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        IndexResult(**values)  # type: ignore[arg-type]


def test_indexer_fails_when_parsing_produced_no_passages(
    app: tuple[AppPaths, Database, _ReadyEmbeddingProvider],
) -> None:
    paths, database, provider = app
    parsed = ParsedBook(
        title="空书",
        author=None,
        source_format="pdf",
        units=(SourceUnit("   \n\n  ", page_start=1, page_end=1),),
    )

    result = BookIndexer(paths, database, provider).index_parsed_book(
        book_id="a" * 24,
        parsed=parsed,
        original_path=paths.originals / "scan.pdf",
    )

    expected = "导入失败：解析完成，但没有生成可检索段落。"
    assert result == IndexResult(
        status="failed",
        parsed_path=None,
        passage_count=0,
        error=expected,
        message=expected,
    )
    assert database.get_book("a" * 24)["status"] == "failed"
    assert database.count_passages("a" * 24) == 0


def test_indexer_degrades_to_keyword_only_when_provider_is_unavailable(
    app: tuple[AppPaths, Database, _ReadyEmbeddingProvider],
) -> None:
    paths, database, _ = app
    parsed = ParsedBook(
        title="仅关键词",
        author=None,
        source_format="pdf",
        units=(SourceUnit("关键词仍然可检索", page_start=2, page_end=2),),
    )

    result = BookIndexer(
        paths,
        database,
        NullEmbeddingProvider(),
    ).index_parsed_book(
        book_id="a" * 24,
        parsed=parsed,
        original_path=paths.originals / "scan.pdf",
    )

    message = (
        "导入完成；语义模型未启用，当前可使用关键词检索，"
        "稍后启用模型即可恢复语义索引。"
    )
    assert result.status == "keyword_only"
    assert result.error == result.message == message
    assert database.get_book("a" * 24)["error"] == message
    assert database.keyword_search("关键词仍然可检索", 5)


@pytest.mark.parametrize(
    ("provider", "detail"),
    [
        (
            _WrongCountEmbeddingProvider(),
            "语义向量数量不匹配：应有 1 个，实际得到 0 个。",
        ),
        (_FailingEmbeddingProvider(), "模型暂时不可用"),
    ],
)
def test_embedding_failures_degrade_with_existing_message_and_guidance(
    app: tuple[AppPaths, Database, _ReadyEmbeddingProvider],
    provider: _ReadyEmbeddingProvider,
    detail: str,
) -> None:
    paths, database, _ = app
    parsed = ParsedBook(
        title="语义降级",
        author=None,
        source_format="pdf",
        units=(SourceUnit("语义失败仍可搜索", page_start=3, page_end=3),),
    )

    result = BookIndexer(paths, database, provider).index_parsed_book(
        book_id="a" * 24,
        parsed=parsed,
        original_path=paths.originals / "scan.pdf",
    )

    expected = f"语义索引失败，可稍后恢复：{detail}"
    assert result.status == "keyword_only"
    assert result.error == result.message == expected
    assert database.get_book("a" * 24)["error"] == expected
    assert database.keyword_search("语义失败仍可搜索", 5)
    assert list(database.iter_embeddings(["a" * 24])) == []


@pytest.mark.parametrize(
    ("case", "detail"),
    [
        ("empty", "第 1 个语义向量不能为空。"),
        ("nan", "第 1 个语义向量必须全部是有限数值。"),
        ("inf", "第 1 个语义向量必须全部是有限数值。"),
        ("two-dimensional-row", "第 1 个语义向量必须是一维数组。"),
        (
            "mixed-dimensions",
            "所有语义向量必须维度一致：期望 2，第 2 个为 3。",
        ),
    ],
)
def test_invalid_vectors_degrade_before_writing_any_embedding(
    app: tuple[AppPaths, Database, _ReadyEmbeddingProvider],
    case: str,
    detail: str,
) -> None:
    paths, database, _ = app
    original = _register_book(
        paths,
        database,
        book_id="b" * 24,
        source_format="txt",
    )

    result = BookIndexer(
        paths,
        database,
        _InvalidEmbeddingProvider(case),
    ).index_parsed_book(
        book_id="b" * 24,
        parsed=_two_passage_book(),
        original_path=original,
    )

    expected = f"语义索引失败，可稍后恢复：{detail}"
    assert result.passage_count >= 2
    assert result.status == "keyword_only"
    assert result.error == result.message == expected
    assert database.keyword_search("第一段", 5)
    assert list(database.iter_embeddings(["b" * 24])) == []


@pytest.mark.parametrize("failure_point", ["render", "replace"])
def test_pipeline_failures_preserve_existing_nonsearchable_behavior(
    app: tuple[AppPaths, Database, _ReadyEmbeddingProvider],
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    paths, database, provider = app

    def fail(*args, **kwargs) -> None:
        raise OSError(f"{failure_point} injected failure")

    if failure_point == "render":
        monkeypatch.setattr(indexing_module, "render_parsed_book", fail)
    else:
        monkeypatch.setattr(database, "replace_passages", fail)

    result = BookIndexer(paths, database, provider).index_parsed_book(
        book_id="a" * 24,
        parsed=ParsedBook(
            title="流水线失败",
            author=None,
            source_format="pdf",
            units=(SourceUnit("不可搜索的内容", page_start=1, page_end=1),),
        ),
        original_path=paths.originals / "scan.pdf",
    )

    assert result.status == "failed"
    expected = f"导入失败：{failure_point} injected failure"
    assert result.error == result.message == expected
    assert database.get_book("a" * 24)["status"] == "failed"
    assert database.count_passages("a" * 24) == 0
    assert database.keyword_search("不可搜索的内容", 5) == []


def test_final_status_failure_falls_back_to_failed_and_hides_passages(
    app: tuple[AppPaths, Database, _ReadyEmbeddingProvider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database, _ = app
    real_update = database.update_book_status
    attempts = 0

    def fail_once(*args, **kwargs) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("first status write failed")
        real_update(*args, **kwargs)

    monkeypatch.setattr(database, "update_book_status", fail_once)

    result = BookIndexer(
        paths,
        database,
        NullEmbeddingProvider(),
    ).index_parsed_book(
        book_id="a" * 24,
        parsed=ParsedBook(
            title="状态失败",
            author=None,
            source_format="pdf",
            units=(SourceUnit("状态失败证据", page_start=1, page_end=1),),
        ),
        original_path=paths.originals / "scan.pdf",
    )

    assert attempts == 2
    assert result.status == "failed"
    assert result.error is not None and "语义模型未启用" in result.error
    assert "状态写入失败：first status write failed" in result.message
    assert database.count_passages("a" * 24) == result.passage_count == 1
    assert database.keyword_search("状态失败证据", 5) == []


def test_interruption_marks_failed_hides_passages_and_reraises(
    app: tuple[AppPaths, Database, _ReadyEmbeddingProvider],
) -> None:
    paths, database, _ = app

    with pytest.raises(KeyboardInterrupt, match="operator cancelled embedding"):
        BookIndexer(
            paths,
            database,
            _InterruptingEmbeddingProvider(),
        ).index_parsed_book(
            book_id="a" * 24,
            parsed=ParsedBook(
                title="中断",
                author=None,
                source_format="pdf",
                units=(SourceUnit("中断前已提交", page_start=1, page_end=1),),
            ),
            original_path=paths.originals / "scan.pdf",
        )

    book = database.get_book("a" * 24)
    assert book is not None
    assert book["status"] == "failed"
    assert book["error"] == "导入被中断：operator cancelled embedding"
    assert database.count_passages("a" * 24) == 1
    assert database.keyword_search("中断前已提交", 5) == []


def test_indexer_does_not_copy_or_parse_the_original(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "app")
    database = Database(paths.database)
    database.initialize()
    paths.vault.mkdir(parents=True)
    external_original = tmp_path / "external-scan.pdf"
    external_original.write_bytes(b"not a parseable PDF")
    database.create_book(
        book_id="c" * 24,
        title="待索引",
        author=None,
        source_format="pdf",
        content_sha256="c" * 64,
        original_path=str(external_original),
    )

    result = BookIndexer(
        paths,
        database,
        NullEmbeddingProvider(),
    ).index_parsed_book(
        book_id="c" * 24,
        parsed=ParsedBook(
            title="已在外部解析",
            author=None,
            source_format="pdf",
            units=(
                SourceUnit("调用方提供的解析结果", page_start=1, page_end=1),
            ),
        ),
        original_path=external_original,
    )

    assert result.status == "keyword_only"
    assert not paths.originals.exists()
    assert external_original.read_bytes() == b"not a parseable PDF"
