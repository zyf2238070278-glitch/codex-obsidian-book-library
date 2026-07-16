from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

from book_agent.config import AppPaths, MAX_PREVIEWS
from book_agent.embeddings import E5EmbeddingProvider, NullEmbeddingProvider
from book_agent.importer import ImportService
from book_agent.indexing import BookIndexer
from book_agent.models import SearchHit
from book_agent.notes import NoteService
from book_agent.ocr.models import OcrJobSummary
from book_agent.ocr.service import (
    DEFAULT_PENDING_LIMIT,
    DEFAULT_STATUS_LIMIT,
    MAXIMUM_RESULT_LIMIT,
    OcrService,
)
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
    "needs_ocr": "请明确说“开始 OCR 这本书”后再进行本机识别。",
    "failed": "请根据 error 修复源文件或本地依赖后重新导入。",
}
_ISSUE_STATUSES = {*_ISSUE_ACTIONS, "keyword_only"}
_OCR_BOOK_ID = re.compile(r"[0-9a-f]{24}\Z")


def _error_payload(error: Exception) -> dict[str, object]:
    message = str(error).strip() or error.__class__.__name__
    return {
        "ok": False,
        "error": message,
        "error_type": error.__class__.__name__,
    }


def _normalize_ocr_value(value: object) -> object:
    if isinstance(value, OcrJobSummary) or is_dataclass(value):
        payload: dict[str, object] = asdict(value)
        percent = getattr(value, "percent_complete", None)
        if percent is not None:
            payload["percent_complete"] = percent
        return _normalize_ocr_value(payload)
    if isinstance(value, dict):
        blocked = {"text", "ocr_text", "page_text", "content", "lines"}
        return {
            str(key): _normalize_ocr_value(item)
            for key, item in value.items()
            if str(key).lower() not in blocked
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_ocr_value(item) for item in value]
    return value


def _ocr_payload(value: object) -> dict[str, Any]:
    """Convert an OCR service result to bounded, JSON-safe metadata only."""

    normalized = _normalize_ocr_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("OCR service returned an unsupported result")
    payload: dict[str, Any] = normalized
    # OCR status intentionally never exposes page text.  Validate the complete
    # response before returning so a bad provider cannot leak non-JSON values.
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload


def _validate_ocr_book_id(book_id: object) -> str:
    if type(book_id) is not str or _OCR_BOOK_ID.fullmatch(book_id) is None:
        raise ValueError("book_id 必须是 24 位小写十六进制字符串。")
    return book_id


def _validate_ocr_limit(value: object, *, name: str) -> int:
    if type(value) is not int or not 1 <= value <= MAXIMUM_RESULT_LIMIT:
        raise ValueError(
            f"{name} 必须是 1 到 {MAXIMUM_RESULT_LIMIT} 之间的整数。"
        )
    return value


def _validate_ocr_offset(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("offset 必须是非负整数。")
    return value


def _book_metadata(book: dict[str, Any]) -> dict[str, object]:
    return {field: book.get(field) for field in _BOOK_FIELDS}


def _keyword_only_action(error: object) -> str:
    detail = error.strip() if isinstance(error, str) else ""
    availability = "关键词检索仍可用。"
    if "语义模型未启用" in detail or "缓存缺失" in detail:
        return (
            f"{availability}恢复顺序：下载模型 → 重新加载 Codex/MCP → "
            "重新导入同一文件；完成后会补建语义向量。"
        )
    if "语义索引失败" in detail:
        return (
            f"{availability}请先查看 error，按 error 修复向量生成、模型运行或"
            "数据库问题，再重新导入同一文件。"
        )
    return (
        f"{availability}请先检查 error 与模型状态，确认具体原因后再按状态建议"
        "修复和重新导入。"
    )


def _issue_action(status: object, error: object) -> str:
    normalized_status = str(status)
    if normalized_status == "keyword_only":
        return _keyword_only_action(error)
    return _ISSUE_ACTIONS[normalized_status]


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
        indexer: BookIndexer | None = None,
        ocr_service: OcrService | Any | None = None,
    ) -> None:
        self.paths = paths
        self.database = database
        self.importer = importer
        self.retriever = retriever
        self.notes = notes
        self.embedding_provider = embedding_provider
        self.indexer = indexer
        self.ocr_service = ocr_service

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
                    "action": _issue_action(book.get("status"), book.get("error")),
                }
                for book in raw_books
                if book.get("status") in _ISSUE_STATUSES
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

    def start_ocr(self, book_id: str) -> dict[str, Any]:
        """Explicitly queue or resume OCR for one managed PDF."""

        def operation() -> dict[str, Any]:
            if self.ocr_service is None:
                raise RuntimeError("OCR 服务未配置。")
            validated_id = _validate_ocr_book_id(book_id)
            return {"ok": True, **_ocr_payload(self.ocr_service.start_ocr(validated_id))}

        return self._guard(operation)

    def start_pending_ocr(
        self,
        limit: int = DEFAULT_PENDING_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Queue a bounded page of eligible OCR books."""

        def operation() -> dict[str, Any]:
            if self.ocr_service is None:
                raise RuntimeError("OCR 服务未配置。")
            safe_limit = _validate_ocr_limit(limit, name="limit")
            safe_offset = _validate_ocr_offset(offset)
            return {
                "ok": True,
                **_ocr_payload(
                    self.ocr_service.start_pending_ocr(
                        limit=safe_limit,
                        offset=safe_offset,
                    )
                ),
            }

        return self._guard(operation)

    def ocr_status(
        self,
        book_id: str | None = None,
        limit: int = DEFAULT_STATUS_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return bounded OCR metadata and progress, never page text."""

        def operation() -> dict[str, Any]:
            if self.ocr_service is None:
                raise RuntimeError("OCR 服务未配置。")
            safe_limit = _validate_ocr_limit(limit, name="limit")
            safe_offset = _validate_ocr_offset(offset)
            if book_id is not None:
                validated_id = _validate_ocr_book_id(book_id)
                result = self.ocr_service.status(validated_id)
            else:
                result = self.ocr_service.status(limit=safe_limit, offset=safe_offset)
            return {"ok": True, **_ocr_payload(result)}

        return self._guard(operation)

    def pause_ocr(self, book_id: str) -> dict[str, Any]:
        """Pause one OCR job at a safe page boundary."""

        def operation() -> dict[str, Any]:
            if self.ocr_service is None:
                raise RuntimeError("OCR 服务未配置。")
            validated_id = _validate_ocr_book_id(book_id)
            return {"ok": True, **_ocr_payload(self.ocr_service.pause(validated_id))}

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
    *,
    vault_root: str | Path | None = None,
) -> LibraryTools:
    """Initialize the managed local library without downloading a model."""

    explicit_vault: Path | None = None
    explicit_vault_identity: tuple[int, int] | None = None
    if vault_root is not None:
        try:
            explicit_vault = Path(
                os.path.abspath(os.fspath(Path(vault_root).expanduser()))
            )
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError(
                f"Explicit Obsidian vault path is invalid: {vault_root}"
            ) from exc
        try:
            vault_info = os.lstat(explicit_vault)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Explicit Obsidian vault does not exist: {explicit_vault}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Explicit Obsidian vault is unavailable: {explicit_vault}"
            ) from exc
        if stat.S_ISLNK(vault_info.st_mode):
            raise ValueError(
                f"Explicit Obsidian vault must not be a symlink: {explicit_vault}"
            )
        if not stat.S_ISDIR(vault_info.st_mode):
            raise ValueError(
                f"Explicit Obsidian vault must be a directory: {explicit_vault}"
            )
        explicit_vault_identity = (vault_info.st_dev, vault_info.st_ino)

    paths = AppPaths.from_root(
        Path(project_root).expanduser(),
        vault_root=explicit_vault,
    )
    VaultManager(
        paths,
        vault_root_identity=explicit_vault_identity,
    ).ensure_layout()
    database = Database(paths.database, root=paths.root)
    database.initialize()

    provider = embedding_provider
    if provider is None:
        local_e5 = E5EmbeddingProvider(paths.models)
        provider = local_e5 if local_e5.available else NullEmbeddingProvider()

    indexer = BookIndexer(
        paths,
        database,
        provider,
        vault_root_identity=explicit_vault_identity,
    )
    importer = ImportService(
        paths,
        database,
        provider,
        vault_root_identity=explicit_vault_identity,
        indexer=indexer,
    )
    retriever = Retriever(database, provider)
    notes = NoteService(
        paths,
        database,
        vault_root_identity=explicit_vault_identity,
    )
    ocr_service = OcrService(paths, database)
    return LibraryTools(
        paths=paths,
        database=database,
        importer=importer,
        retriever=retriever,
        notes=notes,
        embedding_provider=provider,
        indexer=indexer,
        ocr_service=ocr_service,
    )
