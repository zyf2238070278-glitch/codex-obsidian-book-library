from __future__ import annotations

import sqlite3
import os
import stat
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Any

from book_agent.models import Passage, SearchHit
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
            before_open = self._inspect_safe_leaf(directory_fd, allow_missing=True)
            connection = sqlite3.connect(self.path)
            try:
                after_open = self._inspect_safe_leaf(directory_fd, allow_missing=False)
                if before_open is not None and (
                    before_open.st_dev,
                    before_open.st_ino,
                ) != (
                    after_open.st_dev,
                    after_open.st_ino,
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
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "trigram" in message or "fts5" in message:
                raise RuntimeError(
                    "SQLite FTS5 with the trigram tokenizer is required to initialize the book index"
                ) from exc
            raise

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
