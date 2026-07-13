from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from book_agent.chunking import chunk_book
from book_agent.config import AppPaths
from book_agent.embeddings import encode_vector
from book_agent.parsers import (
    SUPPORTED_EXTENSIONS,
    NeedsOcrError,
    parse_document,
)
from book_agent.rendering import render_parsed_book
from book_agent.storage import Database
from book_agent.vault import VaultManager


_HASH_BLOCK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ImportResult:
    book_id: str
    status: str
    source_format: str
    original_path: str
    parsed_path: str | None
    passage_count: int
    message: str

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "book_id": self.book_id,
            "status": self.status,
            "source_format": self.source_format,
            "original_path": self.original_path,
            "parsed_path": self.parsed_path,
            "passage_count": self.passage_count,
            "message": self.message,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _error_detail(error: BaseException) -> str:
    return str(error).strip() or error.__class__.__name__


class ImportService:
    def __init__(
        self,
        paths: AppPaths,
        database: Database,
        embedding_provider: Any,
    ) -> None:
        self.paths = paths
        self.database = database
        self.embedding_provider = embedding_provider
        self.vault = VaultManager(paths)

    def import_book(
        self,
        source: str | Path,
        title: str | None = None,
        author: str | None = None,
    ) -> ImportResult:
        source_path, suffix = self._validate_source(source)
        source_format = suffix.removeprefix(".")
        try:
            content_hash = sha256_file(source_path)
        except OSError as error:
            raise ValueError(f"无法读取待导入文件：{source_path}") from error
        book_id = content_hash[:24]

        duplicate = self.database.find_book_by_hash(content_hash)
        if duplicate is not None:
            return self._duplicate_result(duplicate)

        self.vault.ensure_layout()
        original = self.vault.import_original(source_path)
        try:
            original_path = str(original.absolute())
            copied_hash = sha256_file(original)
            if copied_hash != content_hash:
                content_hash = copied_hash
                book_id = content_hash[:24]
                duplicate = self.database.find_book_by_hash(content_hash)
                if duplicate is not None:
                    duplicate_result = self._duplicate_result(duplicate)
                    self._remove_imported_original(original)
                    return duplicate_result
            initial_title = source_path.stem if title is None else title
        except BaseException as ownership_error:
            self._cleanup_unregistered_original(
                original,
                ownership_error,
                "注册书籍前失败后无法清理刚复制的原书",
            )
            raise
        try:
            self.database.create_book(
                book_id=book_id,
                title=initial_title,
                author=author,
                source_format=source_format,
                content_sha256=content_hash,
                original_path=original_path,
                status="processing",
            )
        except BaseException as create_error:
            self._cleanup_unregistered_original(
                original,
                create_error,
                "创建书籍记录失败后无法清理刚复制的原书",
            )
            raise

        parsed_path: str | None = None
        passage_count = 0
        try:
            parsed = parse_document(original, title=title, author=author)
            self.database.update_book_metadata(book_id, parsed.title, parsed.author)
            destination = self.paths.parsed / book_id / "正文.md"
            markdown_path = destination.relative_to(self.paths.vault).as_posix()
            passages = chunk_book(book_id, parsed, markdown_path)
            if not passages:
                raise ValueError("解析完成，但没有生成可检索段落。")
            render_parsed_book(destination, book_id, parsed, original, passages)
            parsed_path = str(destination.absolute())
            self.database.replace_passages(book_id, passages)
            passage_count = len(passages)

            status, error, message = self._build_semantic_index(passages)
            return self._finalize(
                book_id=book_id,
                status=status,
                source_format=source_format,
                original_path=original_path,
                parsed_path=parsed_path,
                passage_count=passage_count,
                error=error,
                message=message,
            )
        except NeedsOcrError as error:
            detail = _error_detail(error)
            return self._finalize(
                book_id=book_id,
                status="needs_ocr",
                source_format=source_format,
                original_path=original_path,
                parsed_path=None,
                passage_count=0,
                error=f"需要 OCR：{detail}",
                message="原书已保存，但该 PDF 需要 OCR 后才能建立检索索引。",
            )
        except Exception as error:
            detail = _error_detail(error)
            return self._finalize(
                book_id=book_id,
                status="failed",
                source_format=source_format,
                original_path=original_path,
                parsed_path=parsed_path,
                passage_count=passage_count,
                error=f"导入失败：{detail}",
                message=f"导入失败：{detail}",
            )
        except BaseException as interruption:
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
            raise

    @staticmethod
    def _validate_source(source: str | Path) -> tuple[Path, str]:
        try:
            expanded = Path(source).expanduser()
            source_path = Path(os.path.abspath(os.fspath(expanded)))
            source_info = source_path.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"待导入文件不存在：{source}") from error
        except (OSError, RuntimeError, TypeError) as error:
            raise ValueError(f"无法检查待导入文件：{source}") from error

        if stat.S_ISLNK(source_info.st_mode):
            raise ValueError(f"安全起见，不允许导入符号链接：{source_path}")
        if not stat.S_ISREG(source_info.st_mode):
            raise ValueError(f"待导入路径必须是普通文件：{source_path}")

        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            displayed = suffix or "<无扩展名>"
            raise ValueError(f"不支持的书籍格式：{displayed}")
        return source_path, suffix

    def _duplicate_result(self, duplicate: dict[str, Any]) -> ImportResult:
        duplicate_id = str(duplicate["book_id"])
        return ImportResult(
            book_id=duplicate_id,
            status="duplicate",
            source_format=str(duplicate["source_format"]),
            original_path=str(duplicate["original_path"]),
            parsed_path=duplicate["parsed_path"],
            passage_count=self.database.count_passages(duplicate_id),
            message="这本书已经导入，无需重复复制。",
        )

    def _remove_imported_original(self, original: Path) -> None:
        original = original.absolute()
        originals = self.paths.originals.absolute()
        if original.parent != originals:
            raise RuntimeError(f"拒绝删除书库目录外的文件：{original}")
        original_info = original.lstat()
        if stat.S_ISLNK(original_info.st_mode) or not stat.S_ISREG(
            original_info.st_mode
        ):
            raise RuntimeError(f"拒绝删除非普通原书文件：{original}")
        original.unlink()

    def _cleanup_unregistered_original(
        self,
        original: Path,
        failure: BaseException,
        context: str,
    ) -> None:
        try:
            self._remove_imported_original(original)
        except BaseException as cleanup_error:
            failure.add_note(f"{context}：{_error_detail(cleanup_error)}")

    def _build_semantic_index(
        self, passages: list[Any]
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
                    passages, normalized_vectors, strict=True
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
                raise ValueError(f"第 {ordinal} 个语义向量必须全部是有限数值。")
            if expected_dimension is None:
                expected_dimension = int(normalized.size)
            elif normalized.size != expected_dimension:
                raise ValueError(
                    "所有语义向量必须维度一致："
                    f"期望 {expected_dimension}，第 {ordinal} 个为 {normalized.size}。"
                )
            normalized_vectors.append(np.ascontiguousarray(normalized))
        return normalized_vectors

    def _finalize(
        self,
        *,
        book_id: str,
        status: str,
        source_format: str,
        original_path: str,
        parsed_path: str | None,
        passage_count: int,
        error: str | None,
        message: str,
    ) -> ImportResult:
        result_status = status
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
            try:
                self.database.update_book_status(
                    book_id,
                    "failed",
                    error=result_message,
                    parsed_path=parsed_path,
                )
            except Exception as retry_error:
                result_message += f"；失败状态也无法写入：{_error_detail(retry_error)}"
                try:
                    persisted = self.database.get_book(book_id)
                except Exception as read_error:
                    result_message += (
                        f"；也无法读取实际状态：{_error_detail(read_error)}"
                    )
                else:
                    if persisted is not None:
                        result_status = str(persisted["status"])

        return ImportResult(
            book_id=book_id,
            status=result_status,
            source_format=source_format,
            original_path=original_path,
            parsed_path=parsed_path,
            passage_count=passage_count,
            message=result_message,
        )
