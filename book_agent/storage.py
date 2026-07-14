from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import stat
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from itertools import chain
from pathlib import Path
from typing import Any

from book_agent.models import Passage, SearchHit
from book_agent.ocr.models import OcrPageOutcome
from book_agent.vault import _managed_directory_beneath


_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    book_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    source_format TEXT NOT NULL,
    content_sha256 TEXT UNIQUE NOT NULL,
    original_path TEXT NOT NULL,
    parsed_path TEXT,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS passages (
    passage_id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    section TEXT,
    page_start INTEGER,
    page_end INTEGER,
    page_label TEXT,
    markdown_path TEXT NOT NULL,
    anchor TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    embedding BLOB,
    UNIQUE(book_id, ordinal)
);

CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    passage_id UNINDEXED,
    book_id UNINDEXED,
    text,
    tokenize='trigram'
);

CREATE TABLE IF NOT EXISTS ocr_jobs (
    book_id TEXT PRIMARY KEY REFERENCES books(book_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('queued','running','paused','failed','completed')),
    total_pages INTEGER NOT NULL CHECK(total_pages > 0),
    completed_pages INTEGER NOT NULL DEFAULT 0 CHECK(completed_pages >= 0),
    current_page INTEGER,
    language_config TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    pause_requested INTEGER NOT NULL DEFAULT 0 CHECK(pause_requested IN (0,1)),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    error TEXT,
    worker_id TEXT,
    lease_expires_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS ocr_pages (
    book_id TEXT NOT NULL REFERENCES ocr_jobs(book_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK(page_number > 0),
    page_label TEXT,
    text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'recognized',
    engine TEXT,
    strategy TEXT NOT NULL DEFAULT 'legacy',
    detail TEXT,
    mean_confidence REAL,
    duration_ms INTEGER NOT NULL CHECK(duration_ms >= 0),
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(book_id, page_number)
);

CREATE INDEX IF NOT EXISTS ocr_jobs_queue_idx
ON ocr_jobs(status, created_at, book_id);
"""


_HIT_COLUMNS = """
    p.passage_id,
    p.book_id,
    b.title,
    p.text,
    p.section,
    p.page_start,
    p.page_end,
    p.page_label,
    p.markdown_path,
    p.anchor
"""

_SEARCHABLE_STATUSES_SQL = "('ready', 'keyword_only')"
_BOOK_ID_PATTERN = re.compile(r"[0-9a-f]{24}")
_SQLITE_INT_MAX = 2**63 - 1


def _validate_book_id(book_id: object) -> str:
    if type(book_id) is not str or _BOOK_ID_PATTERN.fullmatch(book_id) is None:
        raise ValueError("book_id must be exactly 24 lowercase hexadecimal characters")
    return book_id


def _validate_positive_int(value: object, name: str) -> int:
    if type(value) is not int or not 0 < value <= _SQLITE_INT_MAX:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _SQLITE_INT_MAX:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_nonblank_string(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _validate_optional_string(value: object, name: str) -> str | None:
    if value is not None and type(value) is not str:
        raise ValueError(f"{name} must be a string or None")
    return value


def _timestamp(value: datetime | None = None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    elif type(value) is not datetime:
        raise ValueError("now must be a datetime or None")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must include timezone information")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _lease_timestamp(now: datetime, lease_seconds: int) -> str:
    try:
        return _timestamp(now + timedelta(seconds=lease_seconds))
    except (OverflowError, ValueError) as exc:
        raise ValueError("lease_seconds exceeds the supported datetime range") from exc


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _quote_fts_term(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


class Database:
    def __init__(self, path: str | Path, *, root: str | Path | None = None) -> None:
        try:
            self.path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
            self.root = (
                Path(self.path.anchor)
                if root is None
                else Path(os.path.abspath(os.fspath(Path(root).expanduser())))
            )
            self.path.relative_to(self.root)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Database path must be beneath its configured root") from exc

    def connect(self) -> sqlite3.Connection:
        with self._safe_parent(create=False) as directory_fd:
            parent_before_open = self._open_parent_identity(directory_fd)
            before_open = self._inspect_safe_leaf(directory_fd, allow_missing=True)
            connection = sqlite3.connect(self.path)
            try:
                # sqlite3.connect accepts a pathname rather than a trusted dir_fd.
                # Reopening narrows same-user swaps, but check-open-check cannot make
                # an ABA rename race atomic without lower-level SQLite file controls.
                with self._safe_parent(create=False) as current_directory_fd:
                    current_parent = self._open_parent_identity(current_directory_fd)
                    if current_parent != parent_before_open:
                        raise ValueError(
                            "Database parent changed while the file was being opened"
                        )
                    after_open = self._inspect_safe_leaf(
                        current_directory_fd,
                        allow_missing=False,
                    )
                    after_open_from_original_parent = self._inspect_safe_leaf(
                        directory_fd,
                        allow_missing=False,
                    )
                    if self._stat_identity(after_open) != self._stat_identity(
                        after_open_from_original_parent
                    ):
                        raise ValueError(
                            "Database file changed while its parent was being reopened"
                        )
                if before_open is not None and (
                    self._stat_identity(before_open)
                    != self._stat_identity(after_open)
                ):
                    raise ValueError("Database file changed while it was being opened")
            except BaseException:
                connection.close()
                raise
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _safe_parent(self, *, create: bool) -> Iterator[int]:
        with _managed_directory_beneath(
            self.root,
            self.path.parent,
            "database parent",
            create=create,
        ) as (_, directory_fd):
            yield directory_fd

    @staticmethod
    def _stat_identity(info: os.stat_result) -> tuple[int, int]:
        return info.st_dev, info.st_ino

    @staticmethod
    def _open_parent_identity(directory_fd: int) -> tuple[int, int]:
        try:
            parent_info = os.fstat(directory_fd)
        except OSError as exc:
            raise ValueError("Database parent cannot be inspected safely") from exc
        if not stat.S_ISDIR(parent_info.st_mode):
            raise ValueError("Database parent must be a directory")
        return Database._stat_identity(parent_info)

    def _inspect_safe_leaf(
        self,
        directory_fd: int,
        *,
        allow_missing: bool,
    ) -> os.stat_result | None:
        try:
            leaf_info = os.stat(
                self.path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if allow_missing:
                return None
            raise ValueError("Database file was not created safely") from None
        except OSError as exc:
            raise ValueError("Database file cannot be inspected safely") from exc
        if stat.S_ISLNK(leaf_info.st_mode):
            raise ValueError("Database file must not be a symlink")
        if not stat.S_ISREG(leaf_info.st_mode):
            raise ValueError("Database path must be a regular file")
        if leaf_info.st_nlink != 1:
            raise ValueError("Database file must not have hard-link aliases")
        return leaf_info

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._safe_parent(create=True) as directory_fd:
            self._inspect_safe_leaf(directory_fd, allow_missing=True)
        try:
            with self._connection() as connection:
                connection.executescript(_SCHEMA)
                self._migrate_ocr_page_schema(connection)
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "trigram" in message or "fts5" in message:
                raise RuntimeError(
                    "SQLite FTS5 with the trigram tokenizer is required to initialize the book index"
                ) from exc
            raise

    @staticmethod
    def _migrate_ocr_page_schema(connection: sqlite3.Connection) -> None:
        """Add OCR result metadata to databases created by earlier releases."""

        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(ocr_pages)")
        }
        migrations = (
            ("outcome", "ALTER TABLE ocr_pages ADD COLUMN outcome TEXT NOT NULL DEFAULT 'recognized'"),
            ("engine", "ALTER TABLE ocr_pages ADD COLUMN engine TEXT"),
            ("strategy", "ALTER TABLE ocr_pages ADD COLUMN strategy TEXT NOT NULL DEFAULT 'legacy'"),
            ("detail", "ALTER TABLE ocr_pages ADD COLUMN detail TEXT"),
        )
        for column, statement in migrations:
            if column not in columns:
                connection.execute(statement)
        connection.execute(
            """
            UPDATE ocr_pages
            SET engine = 'apple_vision'
            WHERE engine IS NULL AND outcome = 'recognized' AND strategy = 'legacy'
            """
        )

    def create_book(
        self,
        book_id: str,
        title: str,
        author: str | None,
        source_format: str,
        content_sha256: str,
        original_path: str,
        status: str = "processing",
        parsed_path: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO books (
                    book_id,
                    title,
                    author,
                    source_format,
                    content_sha256,
                    original_path,
                    parsed_path,
                    status,
                    error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    book_id,
                    title,
                    author,
                    source_format,
                    content_sha256,
                    original_path,
                    parsed_path,
                    status,
                    error,
                ),
            )

    def update_book_status(
        self,
        book_id: str,
        status: str,
        error: str | None = None,
        parsed_path: str | None = None,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE books
                SET status = ?,
                    error = ?,
                    parsed_path = COALESCE(?, parsed_path),
                    updated_at = CURRENT_TIMESTAMP
                WHERE book_id = ?
                """,
                (status, error, parsed_path, book_id),
            )

    def update_book_metadata(self, book_id: str, title: str, author: str | None) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE books
                SET title = ?, author = ?, updated_at = CURRENT_TIMESTAMP
                WHERE book_id = ?
                """,
                (title, author, book_id),
            )

    def update_book_original_path(self, book_id: str, original_path: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE books
                SET original_path = ?, updated_at = CURRENT_TIMESTAMP
                WHERE book_id = ?
                """,
                (original_path, book_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Unknown book_id: {book_id}")

    def find_book_by_hash(self, content_sha256: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM books WHERE content_sha256 = ?", (content_sha256,)
            ).fetchone()
        return None if row is None else dict(row)

    def get_book(self, book_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM books WHERE book_id = ?", (book_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def list_books(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM books"
        parameters: tuple[str, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            parameters = (status,)
        sql += " ORDER BY created_at, title, book_id"
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

    def queue_ocr_job(
        self,
        book_id: str,
        total_pages: int,
        languages: tuple[str, ...],
        schema_version: int = 1,
    ) -> dict[str, Any]:
        validated_book_id = _validate_book_id(book_id)
        validated_total_pages = _validate_positive_int(total_pages, "total_pages")
        if type(languages) is not tuple or not languages or any(
            type(language) is not str or not language.strip() for language in languages
        ):
            raise ValueError(
                "languages must be a nonempty tuple of nonblank native strings"
            )
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError("schema_version must be 1")
        language_config = json.dumps(
            list(languages), ensure_ascii=False, separators=(",", ":")
        )
        created_at = _timestamp()

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            book = connection.execute(
                "SELECT source_format, status FROM books WHERE book_id = ?",
                (validated_book_id,),
            ).fetchone()
            if (
                book is None
                or book["source_format"] != "pdf"
                or book["status"] != "needs_ocr"
            ):
                raise ValueError(
                    "OCR jobs require an existing PDF book in needs_ocr status"
                )
            existing = connection.execute(
                "SELECT * FROM ocr_jobs WHERE book_id = ?", (validated_book_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO ocr_jobs (
                        book_id, status, total_pages, language_config,
                        schema_version, created_at, updated_at
                    ) VALUES (?, 'queued', ?, ?, ?, ?, ?)
                    """,
                    (
                        validated_book_id,
                        validated_total_pages,
                        language_config,
                        schema_version,
                        created_at,
                        created_at,
                    ),
                )
            else:
                if (
                    existing["total_pages"] != validated_total_pages
                    or existing["language_config"] != language_config
                    or existing["schema_version"] != schema_version
                ):
                    raise ValueError(
                        "existing OCR job configuration does not match the request"
                    )
                if existing["status"] in ("paused", "failed"):
                    connection.execute(
                        """
                        UPDATE ocr_jobs
                        SET status = 'queued', pause_requested = 0, error = NULL,
                            worker_id = NULL, lease_expires_at = NULL,
                            updated_at = ?, finished_at = NULL
                        WHERE book_id = ?
                        """,
                        (created_at, validated_book_id),
                    )
                elif existing["status"] not in ("queued", "running"):
                    raise ValueError(
                        f"invalid OCR job transition from {existing['status']} to queued"
                    )
            row = connection.execute(
                "SELECT * FROM ocr_jobs WHERE book_id = ?", (validated_book_id,)
            ).fetchone()
        assert row is not None
        return dict(row)

    def claim_next_ocr_job(
        self,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        validated_worker_id = _validate_nonblank_string(worker_id, "worker_id")
        validated_lease_seconds = _validate_positive_int(
            lease_seconds, "lease_seconds"
        )
        now_value = now if now is not None else datetime.now(timezone.utc)
        now_text = _timestamp(now_value)
        lease_text = _lease_timestamp(now_value, validated_lease_seconds)

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            selected = connection.execute(
                """
                SELECT book_id
                FROM ocr_jobs
                WHERE status = 'queued'
                   OR (
                       status = 'running'
                       AND lease_expires_at IS NOT NULL
                       AND lease_expires_at <= ?
                   )
                ORDER BY created_at, book_id
                LIMIT 1
                """,
                (now_text,),
            ).fetchone()
            if selected is None:
                return None
            connection.execute(
                """
                UPDATE ocr_jobs
                SET status = 'running', worker_id = ?, lease_expires_at = ?,
                    attempt_count = attempt_count + 1,
                    error = NULL,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?, finished_at = NULL
                WHERE book_id = ?
                """,
                (
                    validated_worker_id,
                    lease_text,
                    now_text,
                    now_text,
                    selected["book_id"],
                ),
            )
            row = connection.execute(
                "SELECT * FROM ocr_jobs WHERE book_id = ?", (selected["book_id"],)
            ).fetchone()
        assert row is not None
        return dict(row)

    def renew_ocr_lease(
        self,
        book_id: str,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        validated_book_id = _validate_book_id(book_id)
        validated_worker_id = _validate_nonblank_string(worker_id, "worker_id")
        validated_lease_seconds = _validate_positive_int(
            lease_seconds, "lease_seconds"
        )
        now_value = now if now is not None else datetime.now(timezone.utc)
        now_text = _timestamp(now_value)
        lease_text = _lease_timestamp(now_value, validated_lease_seconds)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE ocr_jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE book_id = ? AND status = 'running' AND worker_id = ?
                  AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                """,
                (
                    lease_text,
                    now_text,
                    validated_book_id,
                    validated_worker_id,
                    now_text,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("OCR lease is expired or worker_id is not its owner")
            row = connection.execute(
                "SELECT * FROM ocr_jobs WHERE book_id = ?", (validated_book_id,)
            ).fetchone()
        assert row is not None
        return dict(row)

    def save_ocr_page_result(
        self,
        book_id: str,
        worker_id: str,
        page_number: int,
        page_label: str | None,
        outcome: OcrPageOutcome,
        text: str | None,
        text_sha256: str | None,
        mean_confidence: float | None,
        duration_ms: int,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        validated_book_id = _validate_book_id(book_id)
        validated_worker_id = _validate_nonblank_string(worker_id, "worker_id")
        validated_page_number = _validate_positive_int(page_number, "page_number")
        validated_page_label = _validate_optional_string(page_label, "page_label")
        if type(outcome) is not OcrPageOutcome:
            raise ValueError("outcome must be an OcrPageOutcome")
        if outcome.status == "recognized":
            if type(text) is not str:
                raise ValueError("recognized pages require native string text")
            validated_text = text
            validated_hash = _validate_nonblank_string(text_sha256, "text_sha256")
        else:
            if text is not None or text_sha256 is not None:
                raise ValueError("blank and skipped pages must not provide text or hashes")
            validated_text = ""
            validated_hash = ""
        if mean_confidence is not None and (
            type(mean_confidence) not in (int, float)
            or not 0 <= mean_confidence <= 1
            or not math.isfinite(mean_confidence)
        ):
            raise ValueError("mean_confidence must be None or a finite number from 0 to 1")
        validated_duration = _validate_nonnegative_int(duration_ms, "duration_ms")
        now_text = _timestamp(now)

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT status, total_pages, worker_id, lease_expires_at
                FROM ocr_jobs WHERE book_id = ?
                """,
                (validated_book_id,),
            ).fetchone()
            if job is None:
                raise ValueError(f"Unknown OCR job book_id: {validated_book_id}")
            if job["status"] != "running":
                raise ValueError("invalid OCR job transition: page save requires running")
            if job["worker_id"] != validated_worker_id:
                raise ValueError("worker_id is not the OCR job owner")
            if (
                job["lease_expires_at"] is None
                or job["lease_expires_at"] <= now_text
            ):
                raise ValueError("OCR worker lease has expired")
            if validated_page_number > job["total_pages"]:
                raise ValueError("page_number must not exceed the job total_pages")
            connection.execute(
                """
                INSERT INTO ocr_pages (
                    book_id, page_number, page_label, text, text_sha256,
                    outcome, engine, strategy, detail, mean_confidence,
                    duration_ms, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(book_id, page_number) DO UPDATE SET
                    page_label = excluded.page_label,
                    text = excluded.text,
                    text_sha256 = excluded.text_sha256,
                    outcome = excluded.outcome,
                    engine = excluded.engine,
                    strategy = excluded.strategy,
                    detail = excluded.detail,
                    mean_confidence = excluded.mean_confidence,
                    duration_ms = excluded.duration_ms,
                    completed_at = excluded.completed_at
                """,
                (
                    validated_book_id,
                    validated_page_number,
                    validated_page_label,
                    validated_text,
                    validated_hash,
                    outcome.status,
                    outcome.engine,
                    outcome.strategy,
                    outcome.detail,
                    mean_confidence,
                    validated_duration,
                    now_text,
                ),
            )
            connection.execute(
                """
                UPDATE ocr_jobs
                SET completed_pages = (
                        SELECT COUNT(*) FROM ocr_pages WHERE book_id = ?
                    ),
                    current_page = ?, updated_at = ?
                WHERE book_id = ?
                """,
                (
                    validated_book_id,
                    validated_page_number,
                    now_text,
                    validated_book_id,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM ocr_pages
                WHERE book_id = ? AND page_number = ?
                """,
                (validated_book_id, validated_page_number),
            ).fetchone()
        assert row is not None
        return dict(row)

    def save_ocr_page(
        self,
        book_id: str,
        worker_id: str,
        page_number: int,
        page_label: str | None,
        text: str,
        text_sha256: str,
        mean_confidence: float | None,
        duration_ms: int,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Persist a legacy Apple Vision page checkpoint.

        New callers must use :meth:`save_ocr_page_result` so every non-standard
        page records its terminal outcome explicitly.
        """

        return self.save_ocr_page_result(
            book_id,
            worker_id,
            page_number,
            page_label,
            OcrPageOutcome("recognized", "apple_vision", "legacy"),
            text,
            text_sha256,
            mean_confidence,
            duration_ms,
            now=now,
        )

    def ocr_page_outcome_counts(self, book_id: str) -> dict[str, int]:
        validated_book_id = _validate_book_id(book_id)
        counts = {outcome: 0 for outcome in ("recognized", "blank", "skipped")}
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT outcome, COUNT(*) AS count
                FROM ocr_pages
                WHERE book_id = ?
                GROUP BY outcome
                """,
                (validated_book_id,),
            ).fetchall()
        for row in rows:
            counts[row["outcome"]] = row["count"]
        return counts

    def list_skipped_ocr_pages(self, book_id: str) -> list[dict[str, Any]]:
        validated_book_id = _validate_book_id(book_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT page_number, page_label, strategy, detail
                FROM ocr_pages
                WHERE book_id = ? AND outcome = 'skipped'
                ORDER BY page_number
                """,
                (validated_book_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_ocr_pages(self, book_id: str) -> list[dict[str, Any]]:
        validated_book_id = _validate_book_id(book_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM ocr_pages WHERE book_id = ? ORDER BY page_number",
                (validated_book_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_ocr_job(self, book_id: str) -> dict[str, Any] | None:
        validated_book_id = _validate_book_id(book_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM ocr_jobs WHERE book_id = ?", (validated_book_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def list_ocr_jobs(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM ocr_jobs ORDER BY created_at, book_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def request_ocr_pause(
        self,
        book_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        validated_book_id = _validate_book_id(book_id)
        now_text = _timestamp(now)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE ocr_jobs
                SET pause_requested = 1, updated_at = ?
                WHERE book_id = ? AND status = 'running'
                """,
                (now_text, validated_book_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("invalid OCR job transition: pause request requires running")
            row = connection.execute(
                "SELECT * FROM ocr_jobs WHERE book_id = ?", (validated_book_id,)
            ).fetchone()
        assert row is not None
        return dict(row)

    def pause_ocr_job(
        self,
        book_id: str,
        worker_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return self._finish_running_ocr_job(
            book_id,
            worker_id,
            status="paused",
            error=None,
            current_page=None,
            require_complete=False,
            now=now,
        )

    def fail_ocr_job(
        self,
        book_id: str,
        worker_id: str,
        error: str,
        current_page: int,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        validated_error = _validate_nonblank_string(error, "error")
        validated_current_page = _validate_positive_int(current_page, "current_page")
        return self._finish_running_ocr_job(
            book_id,
            worker_id,
            status="failed",
            error=validated_error,
            current_page=validated_current_page,
            require_complete=False,
            now=now,
        )

    def complete_ocr_job(
        self,
        book_id: str,
        worker_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return self._finish_running_ocr_job(
            book_id,
            worker_id,
            status="completed",
            error=None,
            current_page=None,
            require_complete=True,
            now=now,
        )

    def _finish_running_ocr_job(
        self,
        book_id: str,
        worker_id: str,
        *,
        status: str,
        error: str | None,
        current_page: int | None,
        require_complete: bool,
        now: datetime | None,
    ) -> dict[str, Any]:
        validated_book_id = _validate_book_id(book_id)
        validated_worker_id = _validate_nonblank_string(worker_id, "worker_id")
        now_text = _timestamp(now)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT j.*, b.status AS book_status
                FROM ocr_jobs AS j
                JOIN books AS b ON b.book_id = j.book_id
                WHERE j.book_id = ?
                """,
                (validated_book_id,),
            ).fetchone()
            if job is None or job["status"] != "running":
                raise ValueError(f"invalid OCR job transition to {status}")
            if job["worker_id"] != validated_worker_id:
                raise ValueError("worker_id is not the OCR job owner")
            if (
                job["lease_expires_at"] is None
                or job["lease_expires_at"] <= now_text
            ):
                raise ValueError("OCR worker lease has expired")
            if current_page is not None and current_page > job["total_pages"]:
                raise ValueError("current_page must not exceed total_pages")
            if require_complete and job["completed_pages"] != job["total_pages"]:
                raise ValueError("cannot complete OCR job until all pages are saved")
            if require_complete and job["book_status"] not in (
                "ready",
                "keyword_only",
            ):
                raise ValueError(
                    "cannot complete OCR job until the book is searchable"
                )
            connection.execute(
                """
                UPDATE ocr_jobs
                SET status = ?, error = ?,
                    current_page = COALESCE(?, current_page),
                    pause_requested = 0, worker_id = NULL,
                    lease_expires_at = NULL, updated_at = ?,
                    finished_at = CASE WHEN ? = 'paused' THEN NULL ELSE ? END
                WHERE book_id = ?
                """,
                (
                    status,
                    error,
                    current_page,
                    now_text,
                    status,
                    now_text,
                    validated_book_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM ocr_jobs WHERE book_id = ?", (validated_book_id,)
            ).fetchone()
        assert row is not None
        return dict(row)

    def delete_ocr_page_checkpoints(self, book_id: str) -> None:
        validated_book_id = _validate_book_id(book_id)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT j.status, b.status AS book_status
                FROM ocr_jobs AS j
                JOIN books AS b ON b.book_id = j.book_id
                WHERE j.book_id = ?
                """,
                (validated_book_id,),
            ).fetchone()
            if job is None:
                raise ValueError(f"Unknown OCR job book_id: {validated_book_id}")
            if job["status"] != "completed":
                raise ValueError("checkpoint cleanup requires a completed OCR job")
            if job["book_status"] not in ("ready", "keyword_only"):
                raise ValueError("checkpoint cleanup requires a searchable book")
            connection.execute(
                "DELETE FROM ocr_pages WHERE book_id = ?", (validated_book_id,)
            )

    def count_passages(self, book_id: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM passages"
        parameters: tuple[str, ...] = ()
        if book_id is not None:
            sql += " WHERE book_id = ?"
            parameters = (book_id,)
        with self._connection() as connection:
            return int(connection.execute(sql, parameters).fetchone()[0])

    def replace_passages(self, book_id: str, passages: Iterable[Passage]) -> None:
        materialized = list(passages)
        for passage in materialized:
            if passage.book_id != book_id:
                raise ValueError(
                    f"passage {passage.passage_id!r} has book_id {passage.book_id!r}; "
                    f"expected book_id {book_id!r}"
                )

        passage_rows = [
            (
                passage.passage_id,
                passage.book_id,
                passage.ordinal,
                passage.text,
                passage.section,
                passage.page_start,
                passage.page_end,
                passage.page_label,
                passage.markdown_path,
                passage.anchor,
                passage.text_sha256,
                passage.embedding,
            )
            for passage in materialized
        ]
        fts_rows = [
            (passage.passage_id, passage.book_id, passage.text) for passage in materialized
        ]

        with self._connection() as connection:
            connection.execute("DELETE FROM passages_fts WHERE book_id = ?", (book_id,))
            connection.execute("DELETE FROM passages WHERE book_id = ?", (book_id,))
            connection.executemany(
                """
                INSERT INTO passages (
                    passage_id,
                    book_id,
                    ordinal,
                    text,
                    section,
                    page_start,
                    page_end,
                    page_label,
                    markdown_path,
                    anchor,
                    text_sha256,
                    embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                passage_rows,
            )
            connection.executemany(
                """
                INSERT INTO passages_fts (passage_id, book_id, text)
                VALUES (?, ?, ?)
                """,
                fts_rows,
            )

    def keyword_search(
        self,
        query: str,
        limit: int,
        book_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """Search source passages and metadata using strict all-terms semantics.

        Nonblank whitespace-delimited terms must all occur in the passage text or
        all occur across that passage's title, author, and section metadata. Text
        matches rank before metadata-only matches, and an adjacent exact phrase
        ranks before separated multi-term text matches.
        """
        normalized = query.strip()
        if not normalized or book_ids is not None and not book_ids:
            return []
        safe_limit = max(1, min(int(limit), 20))
        terms = tuple(dict.fromkeys(normalized.split()))
        exact_phrase = " ".join(terms)
        exact_pattern = f"%{_escape_like(exact_phrase)}%"
        fts_terms = [term for term in terms if len(term) >= 3]
        like_terms = [term for term in terms if len(term) < 3]

        if fts_terms:
            fts_expression = " AND ".join(_quote_fts_term(term) for term in fts_terms)
            text_sql = f"""
                SELECT {_HIT_COLUMNS}, bm25(passages_fts) AS rank,
                       CASE WHEN p.text LIKE ? ESCAPE '\\' THEN 0 ELSE 1 END
                           AS exact_match
                FROM passages_fts
                JOIN passages AS p ON p.passage_id = passages_fts.passage_id
                JOIN books AS b ON b.book_id = p.book_id
                WHERE b.status IN {_SEARCHABLE_STATUSES_SQL}
                  AND passages_fts MATCH ?
            """
            text_parameters: list[Any] = [exact_pattern, fts_expression]
            for term in like_terms:
                text_sql += " AND p.text LIKE ? ESCAPE '\\'"
                text_parameters.append(f"%{_escape_like(term)}%")
            text_order = " ORDER BY exact_match, rank, p.book_id, p.ordinal, p.passage_id"
        else:
            text_sql = f"""
                SELECT {_HIT_COLUMNS},
                       CASE WHEN p.text LIKE ? ESCAPE '\\' THEN 0 ELSE 1 END
                           AS exact_match
                FROM passages AS p
                JOIN books AS b ON b.book_id = p.book_id
                WHERE b.status IN {_SEARCHABLE_STATUSES_SQL}
            """
            text_parameters = [exact_pattern]
            for term in terms:
                text_sql += " AND p.text LIKE ? ESCAPE '\\'"
                text_parameters.append(f"%{_escape_like(term)}%")
            text_order = " ORDER BY exact_match, p.book_id, p.ordinal, p.passage_id"

        if book_ids is not None:
            placeholders = ", ".join("?" for _ in book_ids)
            text_sql += f" AND p.book_id IN ({placeholders})"
            text_parameters.extend(book_ids)
        text_sql += text_order + " LIMIT ?"
        text_parameters.append(safe_limit)

        metadata_conditions: list[str] = []
        metadata_parameters: list[Any] = []
        for term in terms:
            metadata_conditions.append(
                """(
                    b.title LIKE ? ESCAPE '\\'
                    OR COALESCE(b.author, '') LIKE ? ESCAPE '\\'
                    OR COALESCE(p.section, '') LIKE ? ESCAPE '\\'
                )"""
            )
            pattern = f"%{_escape_like(term)}%"
            metadata_parameters.extend((pattern, pattern, pattern))
        metadata_sql = f"""
            SELECT {_HIT_COLUMNS}
            FROM passages AS p
            JOIN books AS b ON b.book_id = p.book_id
            WHERE b.status IN {_SEARCHABLE_STATUSES_SQL}
              AND {' AND '.join(metadata_conditions)}
        """
        if book_ids is not None:
            placeholders = ", ".join("?" for _ in book_ids)
            metadata_sql += f" AND p.book_id IN ({placeholders})"
            metadata_parameters.extend(book_ids)
        metadata_sql += " ORDER BY p.book_id, p.ordinal, p.passage_id LIMIT ?"
        metadata_parameters.append(safe_limit)

        with self._connection() as connection:
            text_rows = connection.execute(text_sql, text_parameters).fetchall()
            metadata_rows = connection.execute(
                metadata_sql, metadata_parameters
            ).fetchall()

        hits: list[SearchHit] = []
        seen_passage_ids: set[str] = set()
        text_candidates = (
            ((row, -float(row["rank"])) for row in text_rows)
            if fts_terms
            else ((row, 1.0) for row in text_rows)
        )
        metadata_candidates = ((row, 0.5) for row in metadata_rows)
        for row, score in chain(text_candidates, metadata_candidates):
            passage_id = str(row["passage_id"])
            if passage_id in seen_passage_ids:
                continue
            seen_passage_ids.add(passage_id)
            hits.append(self._hit(row, score))
            if len(hits) >= safe_limit:
                break
        return hits

    def get_passages(self, passage_ids: Sequence[str]) -> list[SearchHit]:
        requested = list(passage_ids)
        if not requested:
            return []
        placeholders = ", ".join("?" for _ in requested)
        sql = f"""
            SELECT {_HIT_COLUMNS}
            FROM passages AS p
            JOIN books AS b ON b.book_id = p.book_id
            WHERE b.status IN {_SEARCHABLE_STATUSES_SQL}
              AND p.passage_id IN ({placeholders})
        """
        with self._connection() as connection:
            rows = connection.execute(sql, requested).fetchall()
        by_id = {row["passage_id"]: self._hit(row, 0.0) for row in rows}
        return [by_id[passage_id] for passage_id in requested if passage_id in by_id]

    def get_neighbors(self, book_id: str, ordinal: int, distance: int) -> list[SearchHit]:
        safe_distance = max(0, int(distance))
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {_HIT_COLUMNS}
                FROM passages AS p
                JOIN books AS b ON b.book_id = p.book_id
                WHERE b.status IN {_SEARCHABLE_STATUSES_SQL}
                  AND p.book_id = ? AND p.ordinal BETWEEN ? AND ?
                ORDER BY p.ordinal
                """,
                (book_id, ordinal - safe_distance, ordinal + safe_distance),
            ).fetchall()
        return [self._hit(row, 0.0) for row in rows]

    def get_ordinal(self, passage_id: str) -> int | None:
        with self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT p.ordinal
                FROM passages AS p
                JOIN books AS b ON b.book_id = p.book_id
                WHERE b.status IN {_SEARCHABLE_STATUSES_SQL}
                  AND p.passage_id = ?
                """,
                (passage_id,),
            ).fetchone()
        return None if row is None else int(row["ordinal"])

    def iter_embeddings(
        self, book_ids: Sequence[str] | None = None
    ) -> Iterator[tuple[SearchHit, bytes]]:
        if book_ids is not None and not book_ids:
            return
        sql = f"""
            SELECT {_HIT_COLUMNS}, p.embedding
            FROM passages AS p
            JOIN books AS b ON b.book_id = p.book_id
            WHERE b.status IN {_SEARCHABLE_STATUSES_SQL}
              AND p.embedding IS NOT NULL AND length(p.embedding) > 0
        """
        parameters: list[str] = []
        if book_ids is not None:
            placeholders = ", ".join("?" for _ in book_ids)
            sql += f" AND p.book_id IN ({placeholders})"
            parameters.extend(book_ids)
        sql += " ORDER BY p.book_id, p.ordinal"
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        for row in rows:
            yield self._hit(row, 0.0), bytes(row["embedding"])

    def set_embeddings(
        self,
        embeddings: Mapping[str, bytes] | Iterable[tuple[str, bytes]],
    ) -> None:
        items = embeddings.items() if isinstance(embeddings, Mapping) else embeddings
        parameters = [(embedding, passage_id) for passage_id, embedding in items]
        with self._connection() as connection:
            connection.executemany(
                "UPDATE passages SET embedding = ? WHERE passage_id = ?", parameters
            )

    def status_counts(self) -> dict[str, int]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM books GROUP BY status ORDER BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    @staticmethod
    def _hit(row: sqlite3.Row, score: float) -> SearchHit:
        return SearchHit(
            passage_id=row["passage_id"],
            book_id=row["book_id"],
            title=row["title"],
            text=row["text"],
            section=row["section"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            page_label=row["page_label"],
            markdown_path=row["markdown_path"],
            anchor=row["anchor"],
            score=score,
        )
