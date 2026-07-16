from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from book_agent.chunking import chunk_book
from book_agent.config import AppPaths
from book_agent.embeddings import encode_vector
from book_agent.models import ParsedBook, Passage
from book_agent.rendering import render_parsed_book
from book_agent.storage import Database


_INDEX_STATUSES = frozenset(
    {"processing", "ready", "keyword_only", "needs_ocr", "failed"}
)
_BOOK_ID_LENGTH = 24
_LOWER_HEX_DIGITS = frozenset("0123456789abcdef")
_INVALID_BOOK_ID_MESSAGE = (
    "索引失败：book_id 必须是 24 位小写十六进制字符串。"
)
_UNSAFE_DESTINATION_MESSAGE = "索引失败：解析文本目标路径不安全。"


def _error_detail(error: BaseException) -> str:
    return str(error).strip() or error.__class__.__name__


@dataclass(frozen=True)
class IndexResult:
    status: str
    parsed_path: str | None
    passage_count: int
    error: str | None
    message: str

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in _INDEX_STATUSES:
            raise ValueError("status is not a supported indexing state")
        if self.parsed_path is not None and (
            type(self.parsed_path) is not str or not self.parsed_path.strip()
        ):
            raise ValueError("parsed_path must be a nonblank string or None")
        if type(self.passage_count) is not int or self.passage_count < 0:
            raise ValueError("passage_count must be a nonnegative integer")
        if self.error is not None and (
            type(self.error) is not str or not self.error.strip()
        ):
            raise ValueError("error must be a nonblank string or None")
        if type(self.message) is not str or not self.message.strip():
            raise ValueError("message must be a nonblank string")

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "status": self.status,
            "parsed_path": self.parsed_path,
            "passage_count": self.passage_count,
            "error": self.error,
            "message": self.message,
        }


class BookIndexer:
    def __init__(
        self,
        paths: AppPaths,
        database: Database,
        embedding_provider: Any,
        *,
        vault_root_identity: tuple[int, int] | None = None,
    ) -> None:
        self.paths = paths
        self.database = database
        self.embedding_provider = embedding_provider
        self.vault_root_identity = vault_root_identity

    def index_parsed_book(
        self,
        *,
        book_id: str,
        parsed: ParsedBook,
        original_path: str | Path,
    ) -> IndexResult:
        if not self._valid_book_id(book_id):
            return self._diagnostic_failure(_INVALID_BOOK_ID_MESSAGE)
        try:
            existing = self.database.get_book(book_id)
        except Exception as error:
            detail = _error_detail(error)
            message = f"导入失败：{detail}"
            return self._finalize(
                book_id=book_id,
                status="failed",
                parsed_path=None,
                passage_count=0,
                error=message,
                message=message,
            )
        except BaseException as interruption:
            self._record_interruption(
                book_id=book_id,
                interruption=interruption,
                parsed_path=None,
            )
            raise
        if existing is None:
            return self._diagnostic_failure(
                f"索引失败：找不到书籍记录：{book_id}"
            )
        destination = self._parsed_destination(book_id)
        if destination is None:
            return self._diagnostic_failure(_UNSAFE_DESTINATION_MESSAGE)

        parsed_path: str | None = None
        passage_count = 0
        try:
            self.database.update_book_metadata(book_id, parsed.title, parsed.author)
            markdown_path = destination.relative_to(self.paths.vault).as_posix()
            passages = chunk_book(book_id, parsed, markdown_path)
            if not passages:
                raise ValueError("解析完成，但没有生成可检索段落。")
            render_parsed_book(
                destination,
                book_id,
                parsed,
                original_path,
                passages,
                managed_root=self.paths.vault,
                expected_root_identity=self.vault_root_identity,
            )
            parsed_path = str(destination.absolute())
            self.database.replace_passages(book_id, passages)
            passage_count = len(passages)

            status, error, message = self._build_semantic_index(passages)
            return self._finalize(
                book_id=book_id,
                status=status,
                parsed_path=parsed_path,
                passage_count=passage_count,
                error=error,
                message=message,
            )
        except Exception as error:
            detail = _error_detail(error)
            message = f"导入失败：{detail}"
            return self._finalize(
                book_id=book_id,
                status="failed",
                parsed_path=parsed_path,
                passage_count=passage_count,
                error=message,
                message=message,
            )
        except BaseException as interruption:
            self._record_interruption(
                book_id=book_id,
                interruption=interruption,
                parsed_path=parsed_path,
            )
            raise

    @staticmethod
    def _valid_book_id(book_id: object) -> bool:
        return (
            type(book_id) is str
            and len(book_id) == _BOOK_ID_LENGTH
            and all(character in _LOWER_HEX_DIGITS for character in book_id)
        )

    @staticmethod
    def _diagnostic_failure(message: str) -> IndexResult:
        return IndexResult(
            status="failed",
            parsed_path=None,
            passage_count=0,
            error=message,
            message=message,
        )

    def _record_interruption(
        self,
        *,
        book_id: str,
        interruption: BaseException,
        parsed_path: str | None,
    ) -> None:
        detail = _error_detail(interruption)
        try:
            self.database.update_book_status(
                book_id,
                "failed",
                error=f"导入被中断：{detail}",
                parsed_path=parsed_path,
            )
        except BaseException as status_error:
            interruption.add_note(
                "记录导入中断状态时失败：" + _error_detail(status_error)
            )

    def _parsed_destination(self, book_id: str) -> Path | None:
        try:
            parsed_root = Path(
                os.path.abspath(os.fspath(self.paths.parsed.expanduser()))
            )
            expected_parent = Path(
                os.path.abspath(os.fspath(parsed_root / book_id))
            )
            destination = Path(
                os.path.abspath(os.fspath(expected_parent / "正文.md"))
            )
            expected_parent.relative_to(parsed_root)
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

        if (
            expected_parent.parent != parsed_root
            or destination.parent != expected_parent
        ):
            return None
        return destination

    def _build_semantic_index(
        self,
        passages: list[Passage],
    ) -> tuple[str, str | None, str]:
        try:
            if not self.embedding_provider.available:
                message = (
                    "导入完成；语义模型未启用，当前可使用关键词检索，"
                    "稍后启用模型即可恢复语义索引。"
                )
                return "keyword_only", message, message

            vectors = list(
                self.embedding_provider.embed_passages(
                    [passage.text for passage in passages]
                )
            )
            if len(vectors) != len(passages):
                raise ValueError(
                    "语义向量数量不匹配："
                    f"应有 {len(passages)} 个，实际得到 {len(vectors)} 个。"
                )
            normalized_vectors = self._validated_vectors(vectors)
            encoded = {
                passage.passage_id: encode_vector(vector)
                for passage, vector in zip(
                    passages,
                    normalized_vectors,
                    strict=True,
                )
            }
            self.database.set_embeddings(encoded)
        except Exception as error:
            detail = _error_detail(error)
            message = f"语义索引失败，可稍后恢复：{detail}"
            return "keyword_only", message, message

        return "ready", None, "导入完成，关键词与语义索引均已就绪。"

    @staticmethod
    def _validated_vectors(vectors: list[Any]) -> list[np.ndarray]:
        normalized_vectors: list[np.ndarray] = []
        expected_dimension: int | None = None
        for ordinal, vector in enumerate(vectors, start=1):
            try:
                normalized = np.asarray(vector, dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"第 {ordinal} 个语义向量无法转换为 float32。"
                ) from error
            if normalized.ndim != 1:
                raise ValueError(f"第 {ordinal} 个语义向量必须是一维数组。")
            if normalized.size == 0:
                raise ValueError(f"第 {ordinal} 个语义向量不能为空。")
            if not np.all(np.isfinite(normalized)):
                raise ValueError(
                    f"第 {ordinal} 个语义向量必须全部是有限数值。"
                )
            if expected_dimension is None:
                expected_dimension = int(normalized.size)
            elif normalized.size != expected_dimension:
                raise ValueError(
                    "所有语义向量必须维度一致："
                    f"期望 {expected_dimension}，"
                    f"第 {ordinal} 个为 {normalized.size}。"
                )
            normalized_vectors.append(np.ascontiguousarray(normalized))
        return normalized_vectors

    def _finalize(
        self,
        *,
        book_id: str,
        status: str,
        parsed_path: str | None,
        passage_count: int,
        error: str | None,
        message: str,
    ) -> IndexResult:
        result_status = status
        result_error = error
        result_message = message
        try:
            self.database.update_book_status(
                book_id,
                status,
                error=error,
                parsed_path=parsed_path,
            )
        except Exception as update_error:
            detail = _error_detail(update_error)
            result_status = "failed"
            result_message = f"{message} 状态写入失败：{detail}"
            result_error = result_message
            try:
                self.database.update_book_status(
                    book_id,
                    "failed",
                    error=result_message,
                    parsed_path=parsed_path,
                )
            except Exception as retry_error:
                result_message += (
                    f"；失败状态也无法写入：{_error_detail(retry_error)}"
                )
                result_error = result_message
                try:
                    persisted = self.database.get_book(book_id)
                except Exception as read_error:
                    result_message += (
                        f"；也无法读取实际状态：{_error_detail(read_error)}"
                    )
                    result_error = result_message
                else:
                    if persisted is not None:
                        result_status = str(persisted["status"])

        return IndexResult(
            status=result_status,
            parsed_path=parsed_path,
            passage_count=passage_count,
            error=result_error,
            message=result_message,
        )
