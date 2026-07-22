from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from book_agent.catalog import CatalogService
from book_agent.config import AppPaths
from book_agent.indexing import BookIndexer
from book_agent.parsers import (
    SUPPORTED_EXTENSIONS,
    NeedsOcrError,
    parse_document,
)
from book_agent.storage import Database
from book_agent.vault import ImportedOriginal, VaultManager


_HASH_BLOCK_SIZE = 1024 * 1024
_MAX_HASH_DRIFT_ATTEMPTS = 3
_BOOK_ID_LENGTH = 24
_LOWER_HEX_DIGITS = frozenset("0123456789abcdef")


class _ContentHashDrift(Exception):
    """Signal that copied bytes no longer match the hash whose lock is held."""


def _required_open_flag(name: str) -> int:
    flag = getattr(os, name, None)
    if not isinstance(flag, int) or flag == 0:
        raise RuntimeError(f"当前平台缺少安全打开导入锁所需的 {name} 支持。")
    return flag


def _secure_directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | _required_open_flag("O_DIRECTORY")
        | _required_open_flag("O_NOFOLLOW")
    )


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


def sha256_file(path: str | Path | int) -> str:
    digest = hashlib.sha256()
    if isinstance(path, int):
        stream_context = os.fdopen(os.dup(path), "rb")
    else:
        stream_context = Path(path).open("rb")
    with stream_context as stream:
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
        *,
        vault_root_identity: tuple[int, int] | None = None,
        indexer: BookIndexer | None = None,
        catalog: CatalogService | None = None,
    ) -> None:
        self.paths = paths
        self.database = database
        self.embedding_provider = embedding_provider
        self.vault_root_identity = vault_root_identity
        self.vault = VaultManager(
            paths,
            vault_root_identity=vault_root_identity,
        )
        self.indexer = (
            indexer
            if indexer is not None
            else BookIndexer(
                paths,
                database,
                embedding_provider,
                vault_root_identity=vault_root_identity,
            )
        )
        self.catalog = catalog if catalog is not None else CatalogService(paths, database)

    def import_book(
        self,
        source: str | Path,
        title: str | None = None,
        author: str | None = None,
    ) -> ImportResult:
        source_path, suffix = self._validate_source(source)
        try:
            content_hash = sha256_file(source_path)
        except OSError as error:
            raise ValueError(f"无法读取待导入文件：{source_path}") from error

        self.vault.ensure_layout()
        for attempt in range(_MAX_HASH_DRIFT_ATTEMPTS):
            book_id = content_hash[:24]
            try:
                with self._book_lock(book_id):
                    result = self._import_locked(
                        source_path=source_path,
                        suffix=suffix,
                        content_hash=content_hash,
                        book_id=book_id,
                        title=title,
                        author=author,
                    )
                return self._sync_catalog(result)
            except _ContentHashDrift:
                if attempt + 1 == _MAX_HASH_DRIFT_ATTEMPTS:
                    break
                try:
                    content_hash = sha256_file(source_path)
                except OSError as error:
                    raise ValueError(
                        f"无法重新读取发生变化的待导入文件：{source_path}"
                    ) from error

        raise ValueError(
            "待导入文件在复制期间持续发生变化，"
            f"已停止导入（共尝试 {_MAX_HASH_DRIFT_ATTEMPTS} 次）：{source_path}"
        )

    def _sync_catalog(self, result: ImportResult) -> ImportResult:
        try:
            book = self.database.get_book(result.book_id)
            if book is not None:
                self.catalog.sync_book(
                    book,
                    preview=CatalogService._preview(book.get("parsed_path")),
                )
                self.catalog.write_base()
        except (OSError, UnicodeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return replace(
                result,
                message=f"{result.message} 书目同步失败：{detail[:240]}",
            )
        return result

    def _import_locked(
        self,
        *,
        source_path: Path,
        suffix: str,
        content_hash: str,
        book_id: str,
        title: str | None,
        author: str | None,
    ) -> ImportResult:
        source_format = suffix.removeprefix(".")
        duplicate = self.database.find_book_by_hash(content_hash)
        if duplicate is not None:
            return self._existing_result(
                duplicate,
                source_path=source_path,
                content_hash=content_hash,
                title=title,
                author=author,
            )

        copied_path = self.vault.import_original(source_path)
        imported_original = self.vault._inspect_original(
            copied_path,
            hasher=sha256_file,
            remove_on_hash_error=True,
        )
        original = imported_original.path
        try:
            original_path = str(original.absolute())
            copied_hash = imported_original.content_sha256
            if copied_hash != content_hash:
                raise _ContentHashDrift
            initial_title = source_path.stem if title is None else title
        except _ContentHashDrift:
            self._remove_imported_original(imported_original)
            raise
        except BaseException as ownership_error:
            self._cleanup_unregistered_original(
                imported_original,
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
                imported_original,
                create_error,
                "创建书籍记录失败后无法清理刚复制的原书",
            )
            raise

        return self._process_registered_book(
            book_id=book_id,
            source_format=source_format,
            original=imported_original,
            original_path=original_path,
            title=title,
            author=author,
        )

    def _existing_result(
        self,
        existing: dict[str, Any],
        *,
        source_path: Path,
        content_hash: str,
        title: str | None,
        author: str | None,
    ) -> ImportResult:
        status = str(existing["status"])
        requires_relocation = self._requires_vault_relocation(existing)
        if (
            not requires_relocation
            and status == "keyword_only"
            and not self.embedding_provider.available
        ):
            return self._duplicate_result(existing)
        if (
            not requires_relocation
            and status not in {"keyword_only", "failed", "processing"}
        ):
            return self._duplicate_result(existing)

        book_id = str(existing["book_id"])
        original, original_path = self._recovery_original(
            existing,
            supplied_source=source_path,
            content_hash=content_hash,
        )
        self.database.update_book_status(book_id, "processing")
        recovery_title = str(existing["title"]) if title is None else title
        recovery_author = existing.get("author") if author is None else author
        return self._process_registered_book(
            book_id=book_id,
            source_format=str(existing["source_format"]),
            original=original,
            original_path=original_path,
            title=recovery_title,
            author=recovery_author,
        )

    def _requires_vault_relocation(self, existing: dict[str, Any]) -> bool:
        try:
            book_id = str(existing["book_id"])
            original_path = Path(
                os.path.abspath(
                    os.fspath(Path(str(existing["original_path"])).expanduser())
                )
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            return True
        if original_path.parent != self.paths.originals.absolute():
            return True

        parsed_value = existing.get("parsed_path")
        if parsed_value is None:
            return str(existing.get("status")) in {"ready", "keyword_only"}
        try:
            parsed_path = Path(
                os.path.abspath(
                    os.fspath(Path(str(parsed_value)).expanduser())
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return True
        expected_parsed = (self.paths.parsed / book_id / "正文.md").absolute()
        return parsed_path != expected_parsed

    def _recovery_original(
        self,
        existing: dict[str, Any],
        *,
        supplied_source: Path,
        content_hash: str,
    ) -> tuple[ImportedOriginal, str]:
        configured = Path(str(existing["original_path"]))
        try:
            original = self.vault._inspect_original(
                configured,
                hasher=sha256_file,
            )
            if original.path.parent != self.paths.originals.absolute():
                raise ValueError("原书不在受管书库目录中。")
            if original.content_sha256 != content_hash:
                raise ValueError("受管原书内容与数据库哈希不一致。")
        except (OSError, ValueError):
            restored_path = self.vault.import_original(supplied_source)
            restored = self.vault._inspect_original(
                restored_path,
                hasher=sha256_file,
            )
            try:
                if restored.content_sha256 != content_hash:
                    raise ValueError("待恢复文件在导入期间发生变化。")
                restored_path = str(restored.path.absolute())
                self.database.update_book_original_path(
                    str(existing["book_id"]), restored_path
                )
            except BaseException as restore_error:
                self._cleanup_unregistered_original(
                    restored,
                    restore_error,
                    "恢复原书失败后无法清理刚复制的文件",
                )
                raise
            return restored, restored_path
        return original, str(original.path.absolute())

    def _process_registered_book(
        self,
        *,
        book_id: str,
        source_format: str,
        original: ImportedOriginal,
        original_path: str,
        title: str | None,
        author: str | None,
    ) -> ImportResult:
        parsed_path: str | None = None
        passage_count = 0
        try:
            self.vault._validate_original_identity(original.path, original.identity)
            try:
                parsed = parse_document(original.path, title=title, author=author)
            finally:
                self.vault._validate_original_identity(
                    original.path,
                    original.identity,
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
                message=(
                    "原书已保存，但该 PDF 没有可提取文字。请明确说“开始 OCR 这本书”"
                    "后再进行本机识别。"
                ),
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

        indexed = self.indexer.index_parsed_book(
            book_id=book_id,
            parsed=parsed,
            original_path=original.path,
        )
        return ImportResult(
            book_id=book_id,
            status=indexed.status,
            source_format=source_format,
            original_path=original_path,
            parsed_path=indexed.parsed_path,
            passage_count=indexed.passage_count,
            message=indexed.message,
        )

    @contextmanager
    def _book_lock(self, book_id: str) -> Iterator[None]:
        if (
            len(book_id) != _BOOK_ID_LENGTH
            or any(character not in _LOWER_HEX_DIGITS for character in book_id)
        ):
            raise ValueError("书籍导入锁标识必须是 24 位小写十六进制字符。")

        flags = (
            os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | _required_open_flag("O_NOFOLLOW")
        )
        lock_directory_fd = self._open_lock_directory()
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(
                    f"{book_id}.lock",
                    flags,
                    0o600,
                    dir_fd=lock_directory_fd,
                )
            except OSError as error:
                raise ValueError(f"无法安全创建书籍导入锁：{book_id}") from error
            lock_info = os.fstat(descriptor)
            if not stat.S_ISREG(lock_info.st_mode):
                raise ValueError(f"书籍导入锁不是普通文件：{book_id}")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                if descriptor is not None:
                    os.close(descriptor)
            finally:
                os.close(lock_directory_fd)

    def _open_lock_directory(self) -> int:
        try:
            root = self.paths.root.expanduser()
            lock_directory = (
                self.paths.database.parent / ".import-locks"
            ).expanduser()
            if not root.is_absolute() or not lock_directory.is_absolute():
                raise ValueError
            relative = lock_directory.relative_to(root)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError("导入锁目录必须位于项目根目录内。") from error

        components = relative.parts
        if not components or any(
            not component or component in {os.curdir, os.pardir}
            for component in components
        ):
            raise ValueError("导入锁目录包含不安全的相对路径组件。")

        directory_flags = _secure_directory_open_flags()
        try:
            current_fd = os.open(root, directory_flags)
        except OSError as error:
            raise ValueError(f"无法安全打开项目根目录：{root}") from error

        current_path = root
        try:
            for component in components:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                except OSError as error:
                    raise ValueError(
                        f"无法安全创建导入锁目录：{current_path / component}"
                    ) from error

                try:
                    next_fd = os.open(
                        component,
                        directory_flags,
                        dir_fd=current_fd,
                    )
                except OSError as error:
                    raise ValueError(
                        f"导入锁目录不安全：{current_path / component}"
                    ) from error

                previous_fd = current_fd
                current_fd = next_fd
                os.close(previous_fd)
                current_path /= component
            return current_fd
        except BaseException:
            os.close(current_fd)
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

    def _remove_imported_original(self, original: ImportedOriginal) -> None:
        self.vault._remove_original(original.path, original.identity)

    def _cleanup_unregistered_original(
        self,
        original: ImportedOriginal,
        failure: BaseException,
        context: str,
    ) -> None:
        try:
            self._remove_imported_original(original)
        except BaseException as cleanup_error:
            failure.add_note(f"{context}：{_error_detail(cleanup_error)}")

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
