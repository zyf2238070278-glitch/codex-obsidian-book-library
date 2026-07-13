from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from pathlib import Path
from typing import Any

from book_agent.config import AppPaths, MAX_PREVIEWS
from book_agent.embeddings import E5EmbeddingProvider, NullEmbeddingProvider
from book_agent.importer import ImportService
from book_agent.models import SearchHit
from book_agent.notes import NoteService
from book_agent.retrieval import Retriever
from book_agent.storage import Database
from book_agent.vault import VaultManager


_BOOK_FIELDS = (
    "book_id",
    "title",
    "author",
    "source_format",
    "original_path",
    "parsed_path",
    "status",
    "error",
    "created_at",
    "updated_at",
)
_ISSUE_ACTIONS = {
    "processing": "等待导入完成；若长期停留，请检查源文件后重新导入。",
    "keyword_only": (
        "恢复顺序：下载模型 → 重新加载 Codex/MCP → 重新导入同一文件；"
        "完成后会补建语义向量。"
    ),
    "needs_ocr": "请先对原文件执行 OCR，再重新导入。",
    "failed": "请根据 error 修复源文件或本地依赖后重新导入。",
}


def _error_payload(error: Exception) -> dict[str, object]:
    message = str(error).strip() or error.__class__.__name__
    return {
        "ok": False,
        "error": message,
        "error_type": error.__class__.__name__,
    }


def _book_metadata(book: dict[str, Any]) -> dict[str, object]:
    return {field: book.get(field) for field in _BOOK_FIELDS}


def _validated_ids(
    values: Sequence[str] | None,
    *,
    name: str,
) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} 必须是字符串 ID 列表。")
    materialized = list(values)
    if any(not isinstance(value, str) or not value.strip() for value in materialized):
        raise ValueError(f"{name} 中每个 ID 必须是非空白字符串。")
    return materialized


def _location(hit: SearchHit) -> str:
    parts: list[str] = []
    if hit.section:
        parts.append(hit.section)
    first_page = hit.page_start
    last_page = hit.page_end
    if first_page is not None or last_page is not None:
        first_page = first_page if first_page is not None else last_page
        last_page = last_page if last_page is not None else first_page
        if first_page == last_page:
            parts.append(f"PDF 页 {first_page}")
        else:
            parts.append(f"PDF 页 {first_page}–{last_page}")
    return " · ".join(parts) or hit.passage_id


class LibraryTools:
    """JSON-safe tool facade over the local book-library services."""

    def __init__(
        self,
        paths: AppPaths,
        database: Database,
        importer: ImportService,
        retriever: Retriever,
        notes: NoteService,
        embedding_provider: object,
    ) -> None:
        self.paths = paths
        self.database = database
        self.importer = importer
        self.retriever = retriever
        self.notes = notes
        self.embedding_provider = embedding_provider

    @staticmethod
    def _guard(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return operation()
        except Exception as error:
            return _error_payload(error)

    def import_book(
        self,
        file_path: str,
        title: str | None = None,
        author: str | None = None,
    ) -> dict[str, Any]:
        """Import a book from the absolute local path of a Codex attachment."""

        def operation() -> dict[str, Any]:
            result = self.importer.import_book(file_path, title=title, author=author)
            return {"ok": True, **result.to_dict()}

        return self._guard(operation)

    def list_books(self, status: str | None = None) -> dict[str, Any]:
        """List book metadata, optionally filtered by import status."""

        def operation() -> dict[str, Any]:
            if status is not None and not isinstance(status, str):
                raise ValueError("status 必须是字符串或 null。")
            books = [_book_metadata(book) for book in self.database.list_books(status)]
            return {"ok": True, "count": len(books), "books": books}

        return self._guard(operation)

    def library_status(self, book_id: str | None = None) -> dict[str, Any]:
        """Report index health and actionable import issues without book text."""

        def operation() -> dict[str, Any]:
            if book_id is not None:
                if not isinstance(book_id, str) or not book_id.strip():
                    raise ValueError("book_id 必须是非空白字符串或 null。")
                selected = self.database.get_book(book_id)
                if selected is None:
                    raise ValueError(f"未知 book_id：{book_id}")
                raw_books = [selected]
                by_status = {str(selected["status"]): 1}
            else:
                raw_books = self.database.list_books()
                by_status = self.database.status_counts()

            books = [_book_metadata(book) for book in raw_books]
            issues = [
                {
                    "book_id": book.get("book_id"),
                    "title": book.get("title"),
                    "status": book.get("status"),
                    "error": book.get("error"),
                    "action": _ISSUE_ACTIONS[str(book["status"])],
                }
                for book in raw_books
                if book.get("status") in _ISSUE_ACTIONS
            ]
            response: dict[str, Any] = {
                "ok": True,
                "database": str(self.paths.database.absolute()),
                "embedding_available": bool(self.embedding_provider.available),
                "embedding_provider": type(self.embedding_provider).__name__,
                "counts": {
                    "books": len(raw_books),
                    "passages": self.database.count_passages(book_id),
                    "by_status": by_status,
                },
                "books": books,
                "issues": issues,
            }
            if book_id is not None:
                response["book"] = books[0]
            return response

        return self._guard(operation)

    def search_books(
        self,
        query: str,
        mode: str = "auto",
        book_ids: Sequence[str] | None = None,
        limit: int = MAX_PREVIEWS,
    ) -> dict[str, Any]:
        """Search the index and return bounded untrusted previews."""

        def operation() -> dict[str, Any]:
            if type(limit) is not int or limit < 1:
                raise ValueError("limit 必须是大于零的整数。")
            selected_ids = _validated_ids(book_ids, name="book_ids")
            safe_limit = min(limit, MAX_PREVIEWS)
            hits = self.retriever.search(
                query=query,
                mode=mode,
                book_ids=selected_ids,
                limit=safe_limit,
            )[:MAX_PREVIEWS]
            results = [self._preview(hit) for hit in hits]
            return {"ok": True, "count": len(results), "results": results}

        return self._guard(operation)

    def get_passages(
        self,
        passage_ids: Sequence[str],
        neighbor_count: int = 1,
    ) -> dict[str, Any]:
        """Expand selected passage IDs into full, explicitly untrusted evidence."""

        def operation() -> dict[str, Any]:
            evidence = self.retriever.get_passages(
                passage_ids,
                neighbor_count=neighbor_count,
            )
            return {"ok": True, "evidence": evidence}

        return self._guard(operation)

    def save_reading_note(
        self,
        title: str,
        markdown: str,
        passage_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Save an AI reading note with citations to known passage IDs."""

        def operation() -> dict[str, Any]:
            saved = self.notes.save(title, markdown, passage_ids)
            return {
                "ok": True,
                "path": saved.path,
                "wiki_link": saved.wiki_link,
            }

        return self._guard(operation)

    @staticmethod
    def _preview(hit: SearchHit) -> dict[str, object]:
        text = hit.text
        score = float(hit.score)
        if not math.isfinite(score):
            score = 0.0
        return {
            "passage_id": hit.passage_id,
            "book_id": hit.book_id,
            "title": hit.title,
            "preview": text[:320],
            "preview_truncated": len(text) > 320,
            "section": hit.section,
            "page_start": hit.page_start,
            "page_end": hit.page_end,
            "page_label": hit.page_label,
            "location": _location(hit),
            "score": score,
            "obsidian_link": f"[[{hit.markdown_path}#^{hit.anchor}]]",
            "untrusted_content": True,
        }


def build_tools(
    project_root: str | Path,
    embedding_provider: object | None = None,
) -> LibraryTools:
    """Initialize the managed local library without downloading a model."""

    paths = AppPaths.from_root(Path(project_root).expanduser())
    VaultManager(paths).ensure_layout()
    database = Database(paths.database)
    database.initialize()

    provider = embedding_provider
    if provider is None:
        local_e5 = E5EmbeddingProvider(paths.models)
        provider = local_e5 if local_e5.available else NullEmbeddingProvider()

    importer = ImportService(paths, database, provider)
    retriever = Retriever(database, provider)
    notes = NoteService(paths, database)
    return LibraryTools(
        paths=paths,
        database=database,
        importer=importer,
        retriever=retriever,
        notes=notes,
        embedding_provider=provider,
    )
