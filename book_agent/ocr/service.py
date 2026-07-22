from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import fitz

from book_agent.config import AppPaths
from book_agent.ocr.models import OCR_SCHEMA_VERSION, OcrJobSummary
from book_agent.storage import Database
from book_agent.vault import _managed_directory_beneath


DEFAULT_LANGUAGES = ("zh-Hans", "en-US")
DEFAULT_PENDING_LIMIT = 25
DEFAULT_STATUS_LIMIT = 20
MAXIMUM_RESULT_LIMIT = 100
RECENT_DURATION_LIMIT = 20
MINIMUM_ESTIMATE_SAMPLES = 5
WORKER_STARTUP_GRACE_SECONDS = 30
SAFE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

_BOOK_ID_PATTERN = re.compile(r"[0-9a-f]{24}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LOCK_NAME = "worker-launch.lock"
_MARKER_NAME = "worker.json"
_LOG_NAME = "worker.log"
_MAXIMUM_MARKER_BYTES = 4096
_HASH_BLOCK_BYTES = 1024 * 1024
_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")

# Darwin's fdesc filesystem rejects a directory descriptor as Popen(cwd=...)
# with ENOTDIR.  This isolated, constant bootstrap binds cwd with fchdir, verifies
# the pinned identity, marks the descriptor close-on-next-exec, then execs only
# the fixed worker module with the already allowlisted environment.
_BOUND_WORKER_BOOTSTRAP = """\
import fcntl, os, sys
if len(sys.argv) != 5:
    raise SystemExit(64)
fd_text, device_text, inode_text, python = sys.argv[1:]
if not all(value.isascii() and value.isdigit() for value in (fd_text, device_text, inode_text)):
    raise SystemExit(64)
fd, expected_device, expected_inode = map(int, (fd_text, device_text, inode_text))
if not os.path.isabs(python) or python != sys.executable:
    raise SystemExit(64)
os.fchdir(fd)
current = os.stat('.')
if (current.st_dev, current.st_ino) != (expected_device, expected_inode):
    raise SystemExit(74)
fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.fcntl(fd, fcntl.F_GETFD) | fcntl.FD_CLOEXEC)
allowed = ('PATH', 'LANG', 'LC_ALL', 'BOOK_LIBRARY_ROOT', 'BOOK_LIBRARY_OBSIDIAN_VAULT', 'BOOK_LIBRARY_LIGHT_OCR_NODE')
environment = {key: os.environ[key] for key in allowed if key in os.environ}
if set(environment) != set(allowed):
    raise SystemExit(78)
os.execve(python, [python, '-m', 'book_agent.ocr_worker'], environment)
"""


class _Process(Protocol):
    pid: int


class _PopenFactory(Protocol):
    def __call__(self, argv: list[str], **kwargs: Any) -> _Process: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_book_id(book_id: object) -> str:
    if type(book_id) is not str or _BOOK_ID_PATTERN.fullmatch(book_id) is None:
        raise ValueError("book_id must be exactly 24 lowercase hexadecimal characters")
    return book_id


def _validate_limit(value: object, *, default_name: str = "limit") -> int:
    if type(value) is not int or not 1 <= value <= MAXIMUM_RESULT_LIMIT:
        raise ValueError(
            f"{default_name} must be an integer from 1 to {MAXIMUM_RESULT_LIMIT}"
        )
    return value


def _validate_offset(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("offset must be a nonnegative integer")
    return value


def _timestamp(value: datetime) -> str:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now_factory must return a timezone-aware datetime")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_timestamp(value: object, label: str) -> datetime:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonblank timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include timezone information")
    return parsed.astimezone(timezone.utc)


def _summary_payload(summary: OcrJobSummary) -> dict[str, object]:
    payload: dict[str, object] = asdict(summary)
    payload["percent_complete"] = summary.percent_complete
    return payload


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class OcrService:
    """Queue and inspect explicit OCR work without recognizing any page inline."""

    def __init__(
        self,
        paths: AppPaths,
        database: Database,
        *,
        catalog: Any | None = None,
        popen_factory: _PopenFactory | None = None,
        now_factory: Callable[[], datetime] = _utc_now,
        pid_probe: Callable[[int], bool] = _pid_exists,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if type(paths) is not AppPaths:
            raise ValueError("paths must be an AppPaths value")
        if not isinstance(database, Database):
            raise ValueError("database must be a Database")
        if not callable(popen_factory if popen_factory is not None else subprocess.Popen):
            raise ValueError("popen_factory must be callable")
        if not callable(now_factory) or not callable(pid_probe):
            raise ValueError("now_factory and pid_probe must be callable")
        if token_factory is not None and not callable(token_factory):
            raise ValueError("token_factory must be callable")
        if type(sys.executable) is not str or not os.path.isabs(sys.executable):
            raise RuntimeError("OCR worker launch requires an absolute Python executable")
        self.paths = paths
        self.database = database
        self.catalog = catalog
        self._popen = subprocess.Popen if popen_factory is None else popen_factory
        self._now_factory = now_factory
        self._pid_probe = pid_probe
        self._token_factory = (
            (lambda: secrets.token_hex(16)) if token_factory is None else token_factory
        )
        self._root_identity = self._directory_identity(paths.root, "project root")

    def start_ocr(self, book_id: str) -> OcrJobSummary:
        """Explicitly queue or resume one book, then ensure a detached worker exists."""

        validated_id = _validate_book_id(book_id)
        book = self.database.get_book(validated_id)
        if book is None:
            raise ValueError(f"Unknown book_id: {validated_id}")

        existing = self.database.get_ocr_job(validated_id)
        if existing is not None and existing.get("status") == "completed":
            return self._summary_for(validated_id)
        if existing is not None and existing.get("status") in ("queued", "running"):
            self._validate_existing_job(book, existing)
            status_value = str(existing["status"])
            if status_value == "running":
                status_value = self._requeue_stale_running(validated_id)
            if status_value == "queued":
                self._sync_catalog(validated_id)
                self._ensure_worker_started()
            return self._summary_for(validated_id)

        self._validate_eligible_book(book)
        total_pages = self._validated_pdf_page_count(book)
        self.database.queue_ocr_job(
            validated_id,
            total_pages,
            DEFAULT_LANGUAGES,
            schema_version=OCR_SCHEMA_VERSION,
        )
        self._sync_catalog(validated_id)
        self._ensure_worker_started()
        return self._summary_for(validated_id)

    def start_pending_ocr(
        self,
        *,
        limit: int = DEFAULT_PENDING_LIMIT,
        offset: int = 0,
    ) -> dict[str, object]:
        """Queue one bounded page of eligible books in deterministic import order."""

        safe_limit = _validate_limit(limit)
        safe_offset = _validate_offset(offset)
        rows = self._pending_books(safe_limit + 1, safe_offset)
        selected = rows[:safe_limit]
        jobs: list[dict[str, object]] = []
        errors: list[dict[str, str]] = []
        should_ensure_worker = False
        for book in selected:
            book_id = _validate_book_id(book.get("book_id"))
            try:
                existing = self.database.get_ocr_job(book_id)
                if existing is not None and existing.get("status") in (
                    "queued",
                    "running",
                ):
                    self._validate_existing_job(book, existing)
                    if existing.get("status") == "running":
                        self._requeue_stale_running(book_id)
                        existing = self.database.get_ocr_job(book_id)
                if existing is None or existing.get("status") not in (
                    "queued",
                    "running",
                    "completed",
                ):
                    self._validate_eligible_book(book)
                    total_pages = self._validated_pdf_page_count(book)
                    self.database.queue_ocr_job(
                        book_id,
                        total_pages,
                        DEFAULT_LANGUAGES,
                        schema_version=OCR_SCHEMA_VERSION,
                    )
                summary = self._summary_for(book_id)
                self._sync_catalog(book_id)
                jobs.append(_summary_payload(summary))
                should_ensure_worker = should_ensure_worker or summary.status == "queued"
            except Exception as exc:
                errors.append(
                    {
                        "book_id": book_id,
                        "error": str(exc).strip() or exc.__class__.__name__,
                        "error_type": exc.__class__.__name__,
                    }
                )

        if should_ensure_worker:
            self._ensure_worker_started()
        has_more = len(rows) > safe_limit
        result: dict[str, object] = {
            "count": len(jobs),
            "queued_count": sum(job["status"] == "queued" for job in jobs),
            "jobs": jobs,
            "error_count": len(errors),
            "errors": errors,
            "offset": safe_offset,
            "limit": safe_limit,
            "has_more": has_more,
            "next_offset": safe_offset + safe_limit if has_more else None,
        }
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result

    def status(
        self,
        book_id: str | None = None,
        *,
        limit: int = DEFAULT_STATUS_LIMIT,
        offset: int = 0,
    ) -> OcrJobSummary | dict[str, object]:
        """Return metadata-only status for one job or a bounded queue page."""

        safe_limit = _validate_limit(limit)
        safe_offset = _validate_offset(offset)
        if book_id is not None:
            validated_id = _validate_book_id(book_id)
            return self._summary_for(validated_id)

        rows = self._job_rows(safe_limit + 1, safe_offset)
        selected = rows[:safe_limit]
        jobs = [_summary_payload(self._summary_from_row(row)) for row in selected]
        total = self._job_count()
        result: dict[str, object] = {
            "count": len(jobs),
            "total": total,
            "jobs": jobs,
            "offset": safe_offset,
            "limit": safe_limit,
            "has_more": len(rows) > safe_limit,
            "next_offset": (
                safe_offset + safe_limit if len(rows) > safe_limit else None
            ),
        }
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result

    def pause(self, book_id: str) -> OcrJobSummary:
        """Pause queued work immediately or request a running page-boundary pause."""

        validated_id = _validate_book_id(book_id)
        now_text = _timestamp(self._now())
        connection = self.database.connect()
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                job = connection.execute(
                    "SELECT status FROM ocr_jobs WHERE book_id=?", (validated_id,)
                ).fetchone()
                if job is None:
                    raise ValueError(f"Unknown OCR job book_id: {validated_id}")
                status = job["status"]
                if status == "queued":
                    connection.execute(
                        """
                        UPDATE ocr_jobs
                        SET status='paused', pause_requested=0, worker_id=NULL,
                            lease_expires_at=NULL, updated_at=?, finished_at=NULL
                        WHERE book_id=? AND status='queued'
                        """,
                        (now_text, validated_id),
                    )
                elif status == "running":
                    connection.execute(
                        """
                        UPDATE ocr_jobs
                        SET pause_requested=1, updated_at=?
                        WHERE book_id=? AND status='running'
                        """,
                        (now_text, validated_id),
                    )
                elif status not in ("paused", "failed", "completed"):
                    raise ValueError(f"Unsupported OCR job status: {status}")
        finally:
            connection.close()
        self._sync_catalog(validated_id)
        return self._summary_for(validated_id)

    def _sync_catalog(self, book_id: str) -> None:
        if self.catalog is None:
            return
        book = self.database.get_book(book_id)
        if book is None:
            return
        try:
            self.catalog.sync_book(book)
        except (KeyboardInterrupt, SystemExit):
            raise
        except (OSError, UnicodeError, ValueError):
            pass

    def _validate_eligible_book(self, book: Mapping[str, object]) -> None:
        source_format = book.get("source_format")
        status_value = book.get("status")
        if source_format != "pdf":
            raise ValueError("OCR is available only for managed PDF books")
        if status_value == "ready" or status_value == "keyword_only":
            raise ValueError("Book is already searchable and does not need OCR")
        if status_value != "needs_ocr":
            raise ValueError(f"Book status {status_value} is not eligible for OCR")

    def _validate_existing_job(
        self,
        book: Mapping[str, object],
        job: Mapping[str, object],
    ) -> None:
        """Revalidate mutable filesystem evidence before every explicit restart."""

        self._validate_eligible_book(book)
        current_page_count = self._validated_pdf_page_count(book)
        stored_page_count = job.get("total_pages")
        if type(stored_page_count) is not int or stored_page_count <= 0:
            raise ValueError("Existing OCR job page count is invalid")
        if current_page_count != stored_page_count:
            raise ValueError(
                "Managed PDF page count no longer matches the existing OCR job"
            )

    def _validated_pdf_page_count(self, book: Mapping[str, object]) -> int:
        raw_path = book.get("original_path")
        if type(raw_path) is not str or not raw_path:
            raise ValueError("Book original_path must be a nonblank absolute path")
        try:
            original = Path(os.path.abspath(os.fspath(Path(raw_path).expanduser())))
            originals = Path(
                os.path.abspath(os.fspath(self.paths.originals.expanduser()))
            )
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError("Book original_path is invalid") from exc
        if not Path(raw_path).expanduser().is_absolute() or original.parent != originals:
            raise ValueError("Book original_path must be directly inside managed originals")
        if original.suffix.lower() != ".pdf":
            raise ValueError("Managed OCR original must have a PDF filename")

        expected_hash = book.get("content_sha256")
        if type(expected_hash) is not str or _SHA256_PATTERN.fullmatch(expected_hash) is None:
            raise ValueError("Book content hash is invalid")

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if type(nofollow) is not int or nofollow == 0:
            raise RuntimeError("This platform lacks O_NOFOLLOW for managed PDF validation")
        flags |= nofollow

        with _managed_directory_beneath(
            self.paths.vault,
            self.paths.originals,
            "managed originals",
            create=False,
            root_label="vault root",
        ) as (_, originals_fd):
            try:
                descriptor = os.open(original.name, flags, dir_fd=originals_fd)
            except OSError as exc:
                raise ValueError(
                    "Managed PDF cannot be opened safely; symlinks are not allowed"
                ) from exc
            document: fitz.Document | None = None
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise ValueError(
                        "Managed PDF must be a regular non-symlink file without aliases"
                    )
                digest_before = self._hash_descriptor(descriptor)
                if digest_before != expected_hash:
                    raise ValueError("Managed PDF content hash no longer matches the book record")
                try:
                    document = fitz.open(f"/dev/fd/{descriptor}")
                except (fitz.FileDataError, RuntimeError, ValueError, OSError) as exc:
                    raise ValueError("Managed PDF is invalid or damaged") from exc
                if document.needs_pass:
                    raise ValueError("Managed PDF is encrypted and requires a password")
                page_count = document.page_count
                if type(page_count) is not int or page_count <= 0:
                    raise ValueError("Managed PDF has zero pages")
                document.close()
                document = None
                digest_after = self._hash_descriptor(descriptor)
                final_descriptor_info = os.fstat(descriptor)
                if digest_after != expected_hash or digest_after != digest_before:
                    raise ValueError("Managed PDF content hash changed during validation")
                if self._file_snapshot(final_descriptor_info) != self._file_snapshot(
                    before
                ):
                    raise ValueError("Managed PDF changed during validation")
                after = os.stat(
                    original.name,
                    dir_fd=originals_fd,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISLNK(after.st_mode)
                    or not stat.S_ISREG(after.st_mode)
                    or self._file_snapshot(after) != self._file_snapshot(before)
                ):
                    raise ValueError("Managed PDF identity changed during validation")
                return page_count
            finally:
                if document is not None:
                    document.close()
                os.close(descriptor)

    @staticmethod
    def _hash_descriptor(descriptor: int) -> str:
        digest = hashlib.sha256()
        offset = 0
        while block := os.pread(descriptor, _HASH_BLOCK_BYTES, offset):
            digest.update(block)
            offset += len(block)
        return digest.hexdigest()

    @staticmethod
    def _file_snapshot(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_uid,
            info.st_gid,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def _summary_for(self, book_id: str) -> OcrJobSummary:
        rows = self._select(
            """
            SELECT j.*, b.title
            FROM ocr_jobs AS j
            JOIN books AS b ON b.book_id=j.book_id
            WHERE j.book_id=?
            """,
            (book_id,),
        )
        if not rows:
            raise ValueError(f"Unknown OCR job book_id: {book_id}")
        return self._summary_from_row(rows[0])

    def _summary_from_row(self, row: Mapping[str, object]) -> OcrJobSummary:
        book_id = _validate_book_id(row.get("book_id"))
        status_value = row.get("status")
        if type(status_value) is not str:
            raise ValueError("OCR job status must be a native string")
        total_pages = row.get("total_pages")
        completed_pages = row.get("completed_pages")
        current_page = row.get("current_page")
        title = row.get("title")
        if type(title) is not str:
            raise ValueError("OCR job title must be a native string")
        queue_position = self._queue_position(row) if status_value == "queued" else None
        estimate = self._remaining_estimate(book_id, total_pages, completed_pages)
        return OcrJobSummary(
            book_id=book_id,
            title=title,
            status=status_value,  # type: ignore[arg-type]
            total_pages=total_pages,  # type: ignore[arg-type]
            completed_pages=completed_pages,  # type: ignore[arg-type]
            current_page=current_page,  # type: ignore[arg-type]
            queue_position=queue_position,
            updated_at=row.get("updated_at"),  # type: ignore[arg-type]
            error=row.get("error"),  # type: ignore[arg-type]
            estimated_remaining_seconds=estimate,
        )

    def _remaining_estimate(
        self,
        book_id: str,
        total_pages: object,
        completed_pages: object,
    ) -> int | None:
        if type(total_pages) is not int or type(completed_pages) is not int:
            raise ValueError("OCR page counts must be native integers")
        rows = self._select(
            """
            SELECT duration_ms
            FROM ocr_pages
            WHERE book_id=?
            ORDER BY completed_at DESC, page_number DESC
            LIMIT ?
            """,
            (book_id, RECENT_DURATION_LIMIT),
        )
        durations = [row.get("duration_ms") for row in rows]
        for duration in durations:
            if type(duration) is not int or duration < 0:
                raise ValueError("OCR duration checkpoints must be nonnegative integers")
        if len(durations) < MINIMUM_ESTIMATE_SAMPLES:
            return None
        remaining = total_pages - completed_pages
        if remaining < 0:
            raise ValueError("OCR completed_pages exceeds total_pages")
        average_ms = sum(durations) / len(durations)
        return max(0, int(round(average_ms * remaining / 1000.0)))

    def _queue_position(self, row: Mapping[str, object]) -> int:
        created_at = row.get("created_at")
        book_id = row.get("book_id")
        if type(created_at) is not str or type(book_id) is not str:
            raise ValueError("Queued OCR job ordering fields are invalid")
        rows = self._select(
            """
            SELECT COUNT(*) AS position
            FROM ocr_jobs
            WHERE status='queued'
              AND (created_at < ? OR (created_at = ? AND book_id <= ?))
            """,
            (created_at, created_at, book_id),
        )
        position = rows[0]["position"]
        if type(position) is not int or position <= 0:
            raise ValueError("Queued OCR job position is invalid")
        return position

    def _pending_books(self, limit: int, offset: int) -> list[dict[str, Any]]:
        return self._select(
            """
            SELECT * FROM books
            WHERE status='needs_ocr' AND source_format='pdf'
            ORDER BY created_at, book_id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

    def _job_rows(self, limit: int, offset: int) -> list[dict[str, Any]]:
        return self._select(
            """
            SELECT j.*, b.title
            FROM ocr_jobs AS j
            JOIN books AS b ON b.book_id=j.book_id
            ORDER BY j.created_at, j.book_id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

    def _job_count(self) -> int:
        value = self._select("SELECT COUNT(*) AS count FROM ocr_jobs", ())[0]["count"]
        if type(value) is not int or value < 0:
            raise ValueError("OCR job count is invalid")
        return value

    def _has_queued_job(self) -> bool:
        return bool(
            self._select("SELECT 1 AS found FROM ocr_jobs WHERE status='queued' LIMIT 1", ())
        )

    def _requeue_stale_running(self, book_id: str) -> str:
        """Recover an expired owner into an actual queued row before launching."""

        now_text = _timestamp(self._now())
        connection = self.database.connect()
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    UPDATE ocr_jobs
                    SET status='queued', worker_id=NULL, lease_expires_at=NULL,
                        pause_requested=0, error=NULL, updated_at=?, finished_at=NULL
                    WHERE book_id=? AND status='running'
                      AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                    """,
                    (now_text, book_id, now_text),
                )
                row = connection.execute(
                    "SELECT status FROM ocr_jobs WHERE book_id=?", (book_id,)
                ).fetchone()
        finally:
            connection.close()
        if row is None or row["status"] not in ("queued", "running"):
            raise ValueError("OCR job changed unexpectedly during stale lease recovery")
        return str(row["status"])

    def _has_live_lease(self, now: datetime) -> bool:
        return bool(
            self._select(
                """
                SELECT 1 AS found FROM ocr_jobs
                WHERE status='running' AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?
                LIMIT 1
                """,
                (_timestamp(now),),
            )
        )

    def _select(self, sql: str, parameters: tuple[object, ...]) -> list[dict[str, Any]]:
        connection = self.database.connect()
        try:
            rows = connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in rows]

    def _ensure_worker_started(self) -> bool:
        root_fd = self._open_bound_directory(
            self.paths.root,
            self._root_identity,
            "project root",
        )
        ocr_fd: int | None = None
        try:
            ocr_fd = self._open_project_directory(
                root_fd,
                self.paths.ocr,
                "OCR runtime",
                create=True,
            )
            lock_fd = self._open_private_file(
                ocr_fd,
                _LOCK_NAME,
                os.O_CREAT | os.O_RDWR,
                "OCR worker launch lock",
            )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                now = self._now()
                if not self._has_queued_job() or self._has_live_lease(now):
                    return False
                if self._marker_is_live(ocr_fd, now):
                    return False
                log_fd = self._open_worker_log(root_fd)
                try:
                    environment = self._worker_environment()
                    token = self._new_token()
                    self._atomic_write_marker(
                        ocr_fd,
                        token=token,
                        state="launching",
                        pid=None,
                        now=now,
                    )
                    argv = self._bound_worker_argv(root_fd)
                    try:
                        process = self._popen(
                            argv,
                            shell=False,
                            cwd="/",
                            start_new_session=True,
                            stdin=subprocess.DEVNULL,
                            stdout=log_fd,
                            stderr=log_fd,
                            close_fds=True,
                            pass_fds=(root_fd,),
                            env=environment,
                        )
                    except (KeyboardInterrupt, SystemExit) as exc:
                        self._cleanup_failed_launch(ocr_fd, token, exc)
                        raise
                    except Exception as exc:
                        self._cleanup_failed_launch(ocr_fd, token, exc)
                        raise RuntimeError(
                            f"Unable to start detached OCR worker: {exc}"
                        ) from exc
                    if type(process.pid) is not int or process.pid <= 0:
                        invalid_pid = RuntimeError(
                            "Detached OCR worker returned an invalid pid"
                        )
                        self._cleanup_failed_launch(ocr_fd, token, invalid_pid)
                        raise invalid_pid
                    self._atomic_write_marker(
                        ocr_fd,
                        token=token,
                        state="running",
                        pid=process.pid,
                        now=now,
                    )
                finally:
                    os.close(log_fd)
                return True
            finally:
                os.close(lock_fd)
        finally:
            if ocr_fd is not None:
                os.close(ocr_fd)
            os.close(root_fd)

    def _open_worker_log(self, root_fd: int) -> int:
        logs_fd = self._open_project_directory(
            root_fd,
            self.paths.ocr_logs,
            "OCR logs",
            create=True,
        )
        try:
            return self._open_private_file(
                logs_fd,
                _LOG_NAME,
                os.O_CREAT | os.O_WRONLY | os.O_APPEND,
                "OCR worker log (non-symlink)",
            )
        finally:
            os.close(logs_fd)

    @staticmethod
    def _open_private_file(
        directory_fd: int,
        name: str,
        base_flags: int,
        label: str,
    ) -> int:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if type(nofollow) is not int or nofollow == 0:
            raise RuntimeError(f"This platform lacks O_NOFOLLOW for {label}")
        try:
            descriptor = os.open(
                name,
                base_flags | nofollow | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ValueError(f"{label} must be a regular non-symlink file") from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"{label} must be a regular file without aliases")
            os.fchmod(descriptor, 0o600)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _worker_environment(self) -> dict[str, str]:
        return {
            "PATH": SAFE_PATH,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "BOOK_LIBRARY_ROOT": ".",
            "BOOK_LIBRARY_OBSIDIAN_VAULT": str(self.paths.vault.absolute()),
            "BOOK_LIBRARY_LIGHT_OCR_NODE": self._validated_light_ocr_node(),
        }

    @staticmethod
    def _validated_light_ocr_node() -> str:
        configured = os.environ.get("BOOK_LIBRARY_LIGHT_OCR_NODE", "").strip()
        if not configured:
            return ""
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            return ""
        try:
            resolved = candidate.resolve(strict=True)
            info = resolved.lstat()
        except OSError:
            return ""
        if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
            return ""
        return str(resolved)

    def _bound_worker_argv(self, root_fd: int) -> list[str]:
        device, inode = self._root_identity
        return [
            sys.executable,
            "-I",
            "-c",
            _BOUND_WORKER_BOOTSTRAP,
            str(root_fd),
            str(device),
            str(inode),
            sys.executable,
        ]

    @staticmethod
    def _directory_identity(path: Path, label: str) -> tuple[int, int]:
        descriptor = OcrService._open_directory(path, label)
        try:
            info = os.fstat(descriptor)
            return info.st_dev, info.st_ino
        finally:
            os.close(descriptor)

    @staticmethod
    def _open_directory(path: Path, label: str) -> int:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError(f"{label} must be an absolute directory")
        nofollow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if type(nofollow) is not int or type(directory_flag) is not int:
            raise RuntimeError(f"This platform lacks secure flags for {label}")
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | nofollow | directory_flag | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as exc:
            raise ValueError(f"{label} cannot be opened safely") from exc
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ValueError(f"{label} must be a directory")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_bound_directory(
        path: Path,
        expected_identity: tuple[int, int],
        label: str,
    ) -> int:
        descriptor = OcrService._open_directory(path, label)
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != expected_identity:
            os.close(descriptor)
            raise ValueError(f"{label} identity changed before worker launch")
        return descriptor

    def _open_project_directory(
        self,
        root_fd: int,
        configured: Path,
        label: str,
        *,
        create: bool,
    ) -> int:
        try:
            relative = configured.absolute().relative_to(self.paths.root.absolute())
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"{label} must be beneath project root") from exc
        current_fd = os.dup(root_fd)
        try:
            for component in relative.parts:
                if not component or component in (os.curdir, os.pardir):
                    raise ValueError(f"{label} contains an unsafe path component")
                if create:
                    try:
                        os.mkdir(component, 0o700, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                info = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise ValueError(f"{label} contains a non-directory or symlink")
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=current_fd,
                )
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except BaseException:
            os.close(current_fd)
            raise

    def _marker_is_live(self, ocr_fd: int, now: datetime) -> bool:
        try:
            descriptor = self._open_private_file(
                ocr_fd,
                _MARKER_NAME,
                os.O_RDONLY,
                "OCR worker marker",
            )
        except ValueError as exc:
            try:
                marker_info = os.stat(
                    _MARKER_NAME,
                    dir_fd=ocr_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return False
            if stat.S_ISLNK(marker_info.st_mode):
                raise ValueError("OCR worker marker must not be a symlink") from exc
            raise
        try:
            marker_identity = os.fstat(descriptor)
            raw = os.read(descriptor, _MAXIMUM_MARKER_BYTES + 1)
        finally:
            os.close(descriptor)
        live = False
        try:
            if len(raw) > _MAXIMUM_MARKER_BYTES:
                raise ValueError("marker is too large")
            payload = json.loads(raw.decode("utf-8"))
            if type(payload) is not dict or set(payload) != {
                "schema_version",
                "state",
                "token",
                "pid",
                "root_device",
                "root_inode",
                "launched_at",
            }:
                raise ValueError("marker schema is invalid")
            if payload["schema_version"] != 1:
                raise ValueError("marker schema version is invalid")
            state = payload["state"]
            token = payload["token"]
            pid = payload["pid"]
            root_device = payload["root_device"]
            root_inode = payload["root_inode"]
            if (
                state not in ("launching", "running")
                or type(token) is not str
                or _TOKEN_PATTERN.fullmatch(token) is None
                or type(root_device) is not int
                or type(root_inode) is not int
            ):
                raise ValueError("marker identity is invalid")
            launched_at = _parse_timestamp(payload["launched_at"], "launched_at")
            age = now.astimezone(timezone.utc) - launched_at
            identity_matches = self._root_identity == (root_device, root_inode)
            within_grace = timedelta(0) <= age <= timedelta(
                seconds=WORKER_STARTUP_GRACE_SECONDS
            )
            if state == "launching":
                if pid is not None:
                    raise ValueError("launching marker pid must be null")
                live = identity_matches and within_grace
            else:
                if type(pid) is not int or pid <= 0:
                    raise ValueError("running marker pid is invalid")
                live = identity_matches and within_grace and bool(self._pid_probe(pid))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError, TypeError):
            live = False
        if not live:
            if self._unlink_if_same(
                ocr_fd,
                _MARKER_NAME,
                marker_identity,
            ):
                os.fsync(ocr_fd)
        return live

    def _atomic_write_marker(
        self,
        ocr_fd: int,
        *,
        token: str,
        state: str,
        pid: int | None,
        now: datetime,
    ) -> None:
        if _TOKEN_PATTERN.fullmatch(token) is None:
            raise ValueError("OCR worker marker token is invalid")
        if state not in ("launching", "running"):
            raise ValueError("OCR worker marker state is invalid")
        if state == "launching" and pid is not None:
            raise ValueError("launching marker pid must be null")
        if state == "running" and (type(pid) is not int or pid <= 0):
            raise ValueError("running marker pid must be a positive integer")
        temp_name = f".worker-marker-{token}.tmp"
        descriptor = self._open_private_file(
            ocr_fd,
            temp_name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            "OCR temporary marker",
        )
        temp_identity = os.fstat(descriptor)
        try:
            payload = json.dumps(
                {
                    "schema_version": 1,
                    "state": state,
                    "token": token,
                    "pid": pid,
                    "root_device": self._root_identity[0],
                    "root_inode": self._root_identity[1],
                    "launched_at": _timestamp(now),
                },
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(
                temp_name,
                _MARKER_NAME,
                src_dir_fd=ocr_fd,
                dst_dir_fd=ocr_fd,
            )
            os.fsync(ocr_fd)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            self._unlink_if_same(ocr_fd, temp_name, temp_identity)
            raise

    def _new_token(self) -> str:
        token = self._token_factory()
        if type(token) is not str or _TOKEN_PATTERN.fullmatch(token) is None:
            raise ValueError("token_factory must return exactly 32 lowercase hex chars")
        return token

    def _remove_marker_for_token(self, ocr_fd: int, token: str) -> None:
        try:
            descriptor = self._open_private_file(
                ocr_fd,
                _MARKER_NAME,
                os.O_RDONLY,
                "OCR worker marker",
            )
        except ValueError:
            return
        try:
            identity = os.fstat(descriptor)
            raw = os.read(descriptor, _MAXIMUM_MARKER_BYTES + 1)
        finally:
            os.close(descriptor)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if type(payload) is not dict or payload.get("token") != token:
            return
        if self._unlink_if_same(ocr_fd, _MARKER_NAME, identity):
            os.fsync(ocr_fd)

    def _cleanup_failed_launch(
        self,
        ocr_fd: int,
        token: str,
        original_error: BaseException,
    ) -> None:
        try:
            self._remove_marker_for_token(ocr_fd, token)
        except Exception as cleanup_error:
            original_error.add_note(
                "OCR launch marker cleanup also failed: "
                f"{str(cleanup_error).strip() or cleanup_error.__class__.__name__}"
            )

    @staticmethod
    def _unlink_if_same(
        directory_fd: int,
        name: str,
        expected: os.stat_result,
    ) -> bool:
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            return False
        os.unlink(name, dir_fd=directory_fd)
        return True

    def _now(self) -> datetime:
        value = self._now_factory()
        _timestamp(value)
        return value


__all__ = [
    "DEFAULT_PENDING_LIMIT",
    "DEFAULT_STATUS_LIMIT",
    "MAXIMUM_RESULT_LIMIT",
    "OcrService",
]
