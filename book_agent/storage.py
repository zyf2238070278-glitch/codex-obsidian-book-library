from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from book_agent.models import Passage, SearchHit


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


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        normalized = query.strip()
        if not normalized or book_ids is not None and not book_ids:
            return []
        safe_limit = max(1, min(int(limit), 20))

        if len(normalized) < 3:
            escaped = (
                normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            sql = f"""
                SELECT {_HIT_COLUMNS}
                FROM passages AS p
                JOIN books AS b ON b.book_id = p.book_id
                WHERE b.status IN {_SEARCHABLE_STATUSES_SQL}
                  AND p.text LIKE ? ESCAPE '\\'
            """
            parameters: list[Any] = [f"%{escaped}%"]
            if book_ids is not None:
                placeholders = ", ".join("?" for _ in book_ids)
                sql += f" AND p.book_id IN ({placeholders})"
                parameters.extend(book_ids)
            sql += " ORDER BY p.book_id, p.ordinal LIMIT ?"
            parameters.append(safe_limit)
            with self._connection() as connection:
                rows = connection.execute(sql, parameters).fetchall()
            return [self._hit(row, 1.0) for row in rows]

        phrase = '"' + normalized.replace('"', '""') + '"'
        sql = f"""
            SELECT {_HIT_COLUMNS}, bm25(passages_fts) AS rank
            FROM passages_fts
            JOIN passages AS p ON p.passage_id = passages_fts.passage_id
            JOIN books AS b ON b.book_id = p.book_id
            WHERE b.status IN {_SEARCHABLE_STATUSES_SQL}
              AND passages_fts MATCH ?
        """
        parameters = [phrase]
        if book_ids is not None:
            placeholders = ", ".join("?" for _ in book_ids)
            sql += f" AND p.book_id IN ({placeholders})"
            parameters.extend(book_ids)
        sql += " ORDER BY rank, p.book_id, p.ordinal LIMIT ?"
        parameters.append(safe_limit)
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._hit(row, -float(row["rank"])) for row in rows]

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
