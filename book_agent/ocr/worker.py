"""Resumable, single-process OCR queue worker.

The worker deliberately keeps OCR page text in the database checkpoint only.  It
does not print OCR output, which makes it safe to run detached from Codex.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import fitz

from book_agent.catalog import CatalogService
from book_agent.config import AppPaths
from book_agent.indexing import BookIndexer
from book_agent.models import ParsedBook, SourceUnit
from book_agent.ocr.models import OcrPageOutcome
from book_agent.ocr.report import write_ocr_report
from book_agent.ocr.router import OcrPageDecision
from book_agent.storage import Database


class VisionOcrEngine(Protocol):
    def recognize_page(self, pdf: Path, *, page_index: int) -> Any: ...


_HASH_BLOCK = 1024 * 1024
_LEASE_SECONDS = 120
_MAX_RETRIES = 2
_BOOK_ID = re.compile(r"[0-9a-f]{24}")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _error(error: BaseException) -> str:
    # Keep diagnostics useful while never writing a page's OCR response to a
    # worker log.  The database error is metadata, not searchable page text.
    detail = str(error).strip()
    if not detail:
        return error.__class__.__name__
    return f"{error.__class__.__name__}: {detail[:500]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _result_text(result: Any) -> tuple[str, float | None]:
    """Normalize VisionPageResult and small test doubles without weakening types."""
    if isinstance(result, str):
        return result.strip(), None
    text_method = getattr(result, "ordered_text", None)
    if callable(text_method):
        text = text_method()
        if type(text) is not str:
            raise ValueError("Vision OCR ordered_text must return a native string")
        confidence = getattr(result, "mean_confidence", None)
        return text.strip(), _confidence(confidence)
    if isinstance(result, dict):
        if "text" not in result:
            raise ValueError("Vision OCR result must contain text")
        text = result["text"]
        if type(text) is not str:
            raise ValueError("Vision OCR text must be a native string")
        confidence = result.get("mean_confidence", result.get("confidence"))
        return text.strip(), _confidence(confidence)
    raise ValueError("Vision OCR returned an unsupported page result")


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Vision OCR confidence must be a finite number from 0 to 1")
    return float(value)


class OcrWorker:
    """Claim and process exactly one book at a time, resuming page checkpoints."""

    def __init__(
        self,
        paths: AppPaths,
        database: Database,
        engine: VisionOcrEngine,
        indexer: BookIndexer,
        *,
        worker_id: str | None = None,
        catalog: CatalogService | Any | None = None,
        lease_seconds: int = _LEASE_SECONDS,
        clock: Callable[[], datetime] = _now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(paths) is not AppPaths or not isinstance(database, Database):
            raise ValueError("paths and database are required")
        if not callable(getattr(engine, "recognize_page", None)):
            raise ValueError("engine must provide recognize_page")
        if not callable(getattr(indexer, "index_parsed_book", None)):
            raise ValueError("indexer must provide index_parsed_book")
        if not isinstance(worker_id, str) or not worker_id.strip():
            worker_id = f"ocr-{uuid.uuid4().hex}"
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.paths = paths
        self.database = database
        self.engine = engine
        self.indexer = indexer
        self.catalog = catalog if catalog is not None else CatalogService(paths, database)
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.monotonic = monotonic

    def run_once(self) -> bool:
        """Process one claimed/claimable job.  Return False when the queue is empty."""
        now = self.clock()
        job = next(
            (
                row
                for row in self.database.list_ocr_jobs()
                if row.get("status") == "running"
                and row.get("worker_id") == self.worker_id
                and self._lease_live(row.get("lease_expires_at"), now)
            ),
            None,
        )
        if job is None:
            job = self.database.claim_next_ocr_job(
                self.worker_id, self.lease_seconds, now=now
            )
        if job is None:
            return False
        try:
            self._process_job(job)
        except (KeyboardInterrupt, SystemExit):
            # Leave a resumable running lease when possible; after expiry a new
            # worker will reclaim it.  Never swallow process-level interruption.
            raise
        except Exception as exc:
            self._book_failure(job, _error(exc), int(job.get("current_page") or 1))
        return True

    def run_until_empty(self) -> int:
        count = 0
        while self.run_once():
            count += 1
        return count

    def _process_job(self, claimed: dict[str, Any]) -> None:
        book_id_value = claimed.get("book_id")
        if type(book_id_value) is not str or _BOOK_ID.fullmatch(book_id_value) is None:
            raise ValueError("OCR job book_id must be 24 lowercase hexadecimal characters")
        book_id = book_id_value
        total_value = claimed.get("total_pages")
        if type(total_value) is not int or isinstance(total_value, bool) or total_value <= 0:
            raise ValueError("OCR job total_pages must be a positive native integer")
        book = self.database.get_book(book_id)
        if book is None:
            raise ValueError(f"unknown book {book_id}")
        original, document = self._validate_original(book, total_value)
        original_identity = self._path_identity(original)
        try:
            pages = {int(row["page_number"]): row for row in self.database.list_ocr_pages(book_id)}
            total = total_value
            for physical_page in range(1, total + 1):
                if physical_page in pages:
                    continue
                # Validate again at the page boundary so a replaced original can
                # never produce a checkpoint for the new bytes.
                current_book = self.database.get_book(book_id)
                if current_book is None or (
                    current_book.get("original_path") != book.get("original_path")
                    or current_book.get("content_sha256") != book.get("content_sha256")
                ):
                    raise ValueError("managed original metadata changed")
                checked = self._validate_original(
                    book, total, expected_path=original, expected_identity=original_identity
                )
                if checked is not None:
                    checked[1].close()
                page_label = self._page_label(document, physical_page)
                last_error: BaseException | None = None
                for attempt in range(_MAX_RETRIES + 1):
                    started = self.monotonic()
                    try:
                        result = self.engine.recognize_page(
                            original, page_index=physical_page - 1
                        )
                        if isinstance(result, OcrPageDecision):
                            outcome = result.outcome
                            text = result.text
                            confidence = result.mean_confidence
                        else:
                            text, confidence = _result_text(result)
                            outcome = (
                                OcrPageOutcome("recognized", "apple_vision", "legacy")
                                if text
                                else OcrPageOutcome("blank", None, "legacy_empty")
                            )
                        elapsed = max(0, int(round((self.monotonic() - started) * 1000)))
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:
                        last_error = exc
                        if attempt >= _MAX_RETRIES:
                            self.database.fail_ocr_job(
                                book_id,
                                self.worker_id,
                                f"physical page {physical_page}: {_error(exc)}",
                                physical_page,
                                now=self.clock(),
                            )
                            return
                        continue
                    # The engine may have taken long enough for the managed PDF
                    # to be replaced or rewritten.  Verify bytes and identity
                    # after recognition and before writing this page.
                    checked = self._validate_original(
                        book, total, expected_path=original, expected_identity=original_identity
                    )
                    if checked is not None:
                        checked[1].close()
                    digest = (
                        hashlib.sha256(text.encode("utf-8")).hexdigest()
                        if outcome.status == "recognized"
                        else None
                    )
                    # Persistence and lease renewal are deliberately outside the
                    # engine retry block: a saved page must never be OCR'd again
                    # merely because renewing its lease failed.
                    self.database.save_ocr_page_result(
                        book_id,
                        self.worker_id,
                        physical_page,
                        page_label,
                        outcome,
                        text if outcome.status == "recognized" else None,
                        digest,
                        confidence,
                        elapsed,
                        now=self.clock(),
                    )
                    break
                else:  # pragma: no cover - defensive; loop always breaks/returns
                    raise last_error or RuntimeError("OCR page failed")
                self.database.renew_ocr_lease(
                    book_id,
                    self.worker_id,
                    self.lease_seconds,
                    now=self.clock(),
                )
                current = self.database.get_ocr_job(book_id)
                if current and current.get("pause_requested"):
                    self.database.pause_ocr_job(book_id, self.worker_id, now=self.clock())
                    return

            # Close the final page-level TOCTOU window before indexing.
            checked = self._validate_original(
                book, total, expected_path=original, expected_identity=original_identity
            )
            if checked is not None:
                checked[1].close()
            # Only non-empty pages become searchable source units.  Empty pages
            # remain checkpoints and still count toward completion.
            checkpoints = self.database.list_ocr_pages(book_id)
            skipped_pages = self.database.list_skipped_ocr_pages(book_id)
            if skipped_pages:
                write_ocr_report(
                    self.paths,
                    book_id=book_id,
                    title=str(book.get("title") or original.stem),
                    skipped_pages=skipped_pages,
                )
            units = tuple(
                SourceUnit(
                    text=str(row["text"]),
                    page_start=int(row["page_number"]),
                    page_end=int(row["page_number"]),
                    page_label=row.get("page_label"),
                )
                for row in checkpoints
                if str(row.get("text") or "").strip()
            )
            parsed = ParsedBook(
                title=str(book.get("title") or original.stem),
                author=book.get("author"),
                source_format="pdf",
                units=units,
            )
            indexed = self._index_with_heartbeat(
                book_id, parsed, original
            )
            status = str(getattr(indexed, "status", "failed"))
            passages = int(getattr(indexed, "passage_count", 0))
            if status not in {"ready", "keyword_only"} or passages <= 0:
                self.database.update_book_status(
                    book_id, "needs_ocr", error="OCR 完成但没有生成可检索段落"
                )
                self.database.fail_ocr_job(
                    book_id,
                    self.worker_id,
                    "OCR 完成但没有生成可检索段落",
                    total,
                    now=self.clock(),
                )
                return
            self.database.complete_ocr_job(book_id, self.worker_id, now=self.clock())
            self.database.delete_ocr_page_checkpoints(book_id)
            if self.catalog is not None:
                refreshed = self.database.get_book(book_id)
                if refreshed is not None:
                    try:
                        self.catalog.sync_book(refreshed)
                    except (OSError, UnicodeError, ValueError):
                        pass
        finally:
            document.close()

    @staticmethod
    def _page_label(document: fitz.Document, page: int) -> str | None:
        try:
            label = document[page - 1].get_label()
        except Exception:
            return None
        value = str(label).strip() if label is not None else ""
        return value or None

    def _validate_original(
        self,
        book: dict[str, Any],
        total_pages: int,
        *,
        expected_path: Path | None = None,
        expected_identity: tuple[int, int] | None = None,
    ) -> tuple[Path, fitz.Document] | None:
        raw = book.get("original_path")
        path = Path(str(raw)).absolute() if raw else Path("/")
        if expected_path is not None and path != expected_path:
            raise ValueError("managed original path changed")
        try:
            root = self.paths.originals.absolute()
            # Match the importer/vault contract: originals are flat, and nested
            # or escaped paths are never managed documents.
            book_id = book.get("book_id")
            if type(book_id) is not str or _BOOK_ID.fullmatch(book_id) is None:
                raise ValueError("book record has an invalid book_id")
            if path.parent != root:
                raise ValueError("managed original must be directly beneath originals")
            if path.name in ("", ".", "..") or path.suffix.lower() != ".pdf":
                raise ValueError("managed original must be a PDF filename")
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("managed original is not a regular file")
            if metadata.st_nlink != 1:
                raise ValueError("managed original must not have hard-link aliases")
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            if expected_identity is not None and identity != expected_identity:
                raise ValueError("managed original file identity changed")
            digest = _sha256(path)
            after_hash = os.lstat(path)
            if (
                stat.S_ISLNK(after_hash.st_mode)
                or not stat.S_ISREG(after_hash.st_mode)
                or after_hash.st_nlink != 1
                or (int(after_hash.st_dev), int(after_hash.st_ino)) != identity
            ):
                raise ValueError("managed original changed while being hashed")
            if digest != str(book.get("content_sha256")):
                raise ValueError("managed original hash changed")
            document = fitz.open(path)
            if document.needs_pass and not document.authenticate(""):
                document.close()
                raise ValueError("managed original is encrypted")
            if len(document) != total_pages:
                document.close()
                raise ValueError("managed original page count changed")
            for index in range(total_pages):
                document[index]
            return path, document
        except BaseException:
            raise

    @staticmethod
    def _path_identity(path: Path) -> tuple[int, int]:
        metadata = os.lstat(path)
        return int(metadata.st_dev), int(metadata.st_ino)

    @staticmethod
    def _lease_live(value: object, now: datetime) -> bool:
        if not isinstance(value, str):
            return False
        try:
            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            expires = datetime.fromisoformat(normalized)
            return expires.tzinfo is not None and expires.astimezone(timezone.utc) > now.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return False

    def _book_failure(self, job: dict[str, Any], error: str, page: int) -> None:
        book_id = str(job["book_id"])
        try:
            self.database.update_book_status(book_id, "needs_ocr", error=error)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass
        try:
            current = self.database.get_ocr_job(book_id)
            if current and current.get("status") == "running":
                self.database.fail_ocr_job(
                    book_id, self.worker_id, error, max(1, page), now=self.clock()
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass

    def _index_with_heartbeat(self, book_id: str, parsed: ParsedBook, original: Path) -> Any:
        stop = threading.Event()
        heartbeat_error: list[BaseException] = []
        interval = max(0.05, min(self.lease_seconds / 3.0, 10.0))

        def heartbeat() -> None:
            while not stop.wait(interval):
                try:
                    self.database.renew_ocr_lease(
                        book_id, self.worker_id, self.lease_seconds, now=self.clock()
                    )
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        heartbeat_error.append(exc)
                    else:
                        heartbeat_error.append(exc)
                    return

        self.database.renew_ocr_lease(
            book_id, self.worker_id, self.lease_seconds, now=self.clock()
        )
        thread = threading.Thread(target=heartbeat, name="ocr-lease-heartbeat", daemon=True)
        thread.start()
        try:
            result = self.indexer.index_parsed_book(
                book_id=book_id, parsed=parsed, original_path=original
            )
            if heartbeat_error:
                raise heartbeat_error[0]
            return result
        finally:
            stop.set()
            thread.join(timeout=max(1.0, interval + 0.5))
