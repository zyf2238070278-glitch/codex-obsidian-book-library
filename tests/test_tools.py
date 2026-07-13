import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest

from book_agent.config import MAX_PREVIEWS
from book_agent.embeddings import NullEmbeddingProvider
from book_agent.models import SearchHit
from book_agent.tools import LibraryTools, build_tools


def _json_round_trip(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _write_chinese_book(path: Path) -> Path:
    path.write_text(
        "库存周期会同时影响企业利润与产业需求。" + "甲" * 700
        + "\n\n第二段讨论风险、机会与现金流。" + "乙" * 500,
        encoding="utf-8",
    )
    return path


@pytest.fixture
def library(tmp_path: Path) -> LibraryTools:
    return build_tools(
        tmp_path / "library-root",
        embedding_provider=NullEmbeddingProvider(),
    )


def test_real_txt_workflow_is_json_safe_and_preserves_content_boundaries(
    library: LibraryTools,
    tmp_path: Path,
) -> None:
    source = _write_chinese_book(tmp_path / "中文书籍.txt")

    imported = library.import_book(str(source), author="研究员")
    listed = library.list_books(status="keyword_only")
    status = library.library_status(imported["book_id"])
    searched = library.search_books("库存周期", mode="quote", limit=100)

    assert imported["ok"] is True
    assert imported["status"] == "keyword_only"
    assert listed["ok"] is True
    assert listed["count"] == 1
    assert listed["books"][0]["book_id"] == imported["book_id"]
    assert "text" not in listed["books"][0]
    assert "embedding" not in listed["books"][0]
    assert status["ok"] is True
    assert Path(status["database"]).is_absolute()
    assert status["embedding_available"] is False
    assert status["embedding_provider"] == "NullEmbeddingProvider"
    assert status["counts"]["books"] == 1
    assert status["counts"]["passages"] >= 1
    assert status["counts"]["by_status"] == {"keyword_only": 1}
    assert status["book"]["book_id"] == imported["book_id"]
    assert searched["ok"] is True
    assert 1 <= searched["count"] <= MAX_PREVIEWS

    result = searched["results"][0]
    assert set(result) == {
        "passage_id",
        "book_id",
        "title",
        "preview",
        "preview_truncated",
        "section",
        "page_start",
        "page_end",
        "page_label",
        "location",
        "score",
        "obsidian_link",
        "untrusted_content",
    }
    assert "text" not in result
    assert "embedding" not in result
    assert len(result["preview"]) <= 320
    assert result["preview_truncated"] is True
    assert result["untrusted_content"] is True
    assert math.isfinite(result["score"])

    evidence = library.get_passages([result["passage_id"]], neighbor_count=0)
    saved = library.save_reading_note(
        "周期研读",
        "这是基于证据的分析。",
        [result["passage_id"]],
    )
    duplicate = library.import_book(str(source))

    assert evidence["ok"] is True
    assert evidence["evidence"][0]["text"].startswith("库存周期")
    assert evidence["evidence"][0]["untrusted_content"] is True
    assert saved["ok"] is True
    assert Path(saved["path"]).is_file()
    assert saved["wiki_link"].startswith("[[书库/30-AI读书笔记/")
    assert duplicate["ok"] is True
    assert duplicate["status"] == "duplicate"
    assert library.list_books()["count"] == 1

    for payload in (
        imported,
        listed,
        status,
        searched,
        evidence,
        saved,
        duplicate,
    ):
        assert _json_round_trip(payload) == payload


def test_import_facade_names_codex_attachment_path_explicitly(
    library: LibraryTools,
    tmp_path: Path,
) -> None:
    source = _write_chinese_book(tmp_path / "Codex附件.txt")

    result = library.import_book(file_path=str(source))
    signature = inspect.signature(LibraryTools.import_book)

    assert result["ok"] is True
    assert "file_path" in signature.parameters
    assert "source" not in signature.parameters
    assert _json_round_trip(result) == result


def test_library_status_reports_actionable_issues_without_book_text(
    library: LibraryTools,
) -> None:
    for status in ("processing", "keyword_only", "needs_ocr", "failed"):
        library.database.create_book(
            book_id=f"book-{status}",
            title=f"书-{status}",
            author=None,
            source_format="pdf",
            content_sha256=f"hash-{status}",
            original_path=f"/books/{status}.pdf",
            status=status,
            error=f"error-{status}" if status != "processing" else None,
        )

    report = library.library_status()

    assert report["ok"] is True
    assert report["counts"]["books"] == 4
    assert {issue["status"] for issue in report["issues"]} == {
        "processing",
        "keyword_only",
        "needs_ocr",
        "failed",
    }
    for issue in report["issues"]:
        assert set(issue) == {"book_id", "title", "status", "error", "action"}
        assert issue["action"].strip()
        assert "text" not in issue
        assert "embedding" not in issue
    for book in report["books"]:
        assert "text" not in book
        assert "embedding" not in book

    keyword_issue = next(
        issue for issue in report["issues"] if issue["status"] == "keyword_only"
    )
    assert "关键词检索" in keyword_issue["action"]
    assert "error" in keyword_issue["action"]
    assert "模型状态" in keyword_issue["action"]


@pytest.mark.parametrize(
    "error",
    [
        "导入完成；语义模型未启用，当前可使用关键词检索。",
        "语义模型缓存缺失，当前可使用关键词检索。",
    ],
)
def test_keyword_only_model_unavailable_action_orders_model_recovery_steps(
    library: LibraryTools,
    error: str,
) -> None:
    library.database.create_book(
        book_id="book-model-unavailable",
        title="模型未就绪",
        author=None,
        source_format="txt",
        content_sha256="hash-model-unavailable",
        original_path="/books/model-unavailable.txt",
        status="keyword_only",
        error=error,
    )

    action = library.library_status("book-model-unavailable")["issues"][0]["action"]

    assert "关键词检索" in action
    download = action.index("下载模型")
    reload_process = action.index("重新加载 Codex/MCP")
    reimport = action.index("重新导入")
    assert download < reload_process < reimport


@pytest.mark.parametrize(
    "error",
    [
        "语义索引失败，可稍后恢复：语义向量数量不匹配。",
        "语义索引失败，可稍后恢复：模型运行时暂时不可用。",
        "语义索引失败，可稍后恢复：数据库写入失败。",
    ],
)
def test_keyword_only_index_failure_action_uses_recorded_error_not_download(
    library: LibraryTools,
    error: str,
) -> None:
    library.database.create_book(
        book_id="book-index-failure",
        title="索引失败",
        author=None,
        source_format="txt",
        content_sha256="hash-index-failure",
        original_path="/books/index-failure.txt",
        status="keyword_only",
        error=error,
    )

    action = library.library_status("book-index-failure")["issues"][0]["action"]

    assert "关键词检索" in action
    inspect_error = action.index("查看 error")
    repair = action.index("修复")
    reimport = action.index("重新导入")
    assert inspect_error < repair < reimport
    assert "下载模型" not in action


def test_keyword_only_legacy_record_without_error_gets_conservative_action(
    library: LibraryTools,
) -> None:
    library.database.create_book(
        book_id="book-legacy-keyword-only",
        title="旧记录",
        author=None,
        source_format="txt",
        content_sha256="hash-legacy-keyword-only",
        original_path="/books/legacy.txt",
        status="keyword_only",
        error=None,
    )

    action = library.library_status("book-legacy-keyword-only")["issues"][0][
        "action"
    ]

    assert "关键词检索" in action
    assert action.index("检查 error") < action.index("模型状态")
    assert "下载模型" not in action


def test_search_caps_results_and_normalizes_non_finite_scores(
    library: LibraryTools,
) -> None:
    hits = [
        SearchHit(
            passage_id=f"passage-{index}",
            book_id="book-1",
            title="测试书",
            text="原文" * 300,
            section="章节",
            page_start=index,
            page_end=index,
            page_label=str(index),
            markdown_path="书库/20-解析文本/book-1/正文.md",
            anchor=f"passage-{index}",
            score=float("inf") if index == 0 else float(index),
        )
        for index in range(MAX_PREVIEWS + 5)
    ]

    class ManyHitsRetriever:
        def search(self, *args: object, **kwargs: object) -> list[SearchHit]:
            return hits

    wrapped = LibraryTools(
        paths=library.paths,
        database=library.database,
        importer=library.importer,
        retriever=ManyHitsRetriever(),
        notes=library.notes,
        embedding_provider=library.embedding_provider,
    )

    result = wrapped.search_books("原文", limit=100)

    assert result["ok"] is True
    assert result["count"] == MAX_PREVIEWS
    assert all(math.isfinite(hit["score"]) for hit in result["results"])
    assert all(len(hit["preview"]) <= 320 for hit in result["results"])
    assert all("text" not in hit for hit in result["results"])


@pytest.mark.parametrize(
    "invoke",
    [
        lambda tools, missing: tools.import_book(str(missing)),
        lambda tools, missing: tools.library_status("missing-book"),
        lambda tools, missing: tools.search_books("库存", mode="unsupported"),
        lambda tools, missing: tools.search_books("库存", limit="bad"),
        lambda tools, missing: tools.search_books("库存", book_ids="book-1"),
        lambda tools, missing: tools.get_passages(["missing"], neighbor_count=0),
        lambda tools, missing: tools.get_passages(["missing"], neighbor_count=2),
        lambda tools, missing: tools.get_passages("missing", neighbor_count=0),
        lambda tools, missing: tools.save_reading_note(
            "未知证据", "正文", ["missing"]
        ),
    ],
)
def test_invalid_tool_calls_return_readable_json_errors(
    library: LibraryTools,
    tmp_path: Path,
    invoke: Any,
) -> None:
    result = invoke(library, tmp_path / "missing.txt")

    assert result["ok"] is False
    assert isinstance(result["error"], str) and result["error"].strip()
    assert isinstance(result["error_type"], str) and result["error_type"].strip()
    assert _json_round_trip(result) == result
    assert "Traceback" not in result["error"]


def test_regular_dependency_exceptions_are_wrapped_but_interrupts_propagate(
    library: LibraryTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(status: str | None = None) -> list[dict[str, object]]:
        raise KeyError("injected failure")

    monkeypatch.setattr(library.database, "list_books", fail)
    wrapped = library.list_books()

    assert wrapped == {
        "ok": False,
        "error": "'injected failure'",
        "error_type": "KeyError",
    }

    def interrupt(status: str | None = None) -> list[dict[str, object]]:
        raise KeyboardInterrupt("operator cancelled")

    monkeypatch.setattr(library.database, "list_books", interrupt)
    with pytest.raises(KeyboardInterrupt, match="operator cancelled"):
        library.list_books()


def test_provider_availability_failures_are_wrapped(
    library: LibraryTools,
) -> None:
    class BrokenProvider:
        @property
        def available(self) -> bool:
            raise RuntimeError("provider probe failed")

    wrapped = LibraryTools(
        paths=library.paths,
        database=library.database,
        importer=library.importer,
        retriever=library.retriever,
        notes=library.notes,
        embedding_provider=BrokenProvider(),
    )

    result = wrapped.library_status()

    assert result == {
        "ok": False,
        "error": "provider probe failed",
        "error_type": "RuntimeError",
    }


def test_build_tools_uses_one_explicit_provider_and_empty_cache_stays_offline(
    tmp_path: Path,
) -> None:
    explicit = NullEmbeddingProvider()
    injected = build_tools(tmp_path / "injected", embedding_provider=explicit)
    offline = build_tools(tmp_path / "offline")

    assert injected.embedding_provider is explicit
    assert injected.importer.embedding_provider is explicit
    assert injected.retriever.embedding_provider is explicit
    assert isinstance(offline.embedding_provider, NullEmbeddingProvider)
    assert offline.paths.database.is_file()
    assert offline.paths.notes.is_dir()
