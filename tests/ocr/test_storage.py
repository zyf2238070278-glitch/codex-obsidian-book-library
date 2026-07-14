from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from book_agent.models import Passage
from book_agent.storage import Database


BOOK_A = "a" * 24
BOOK_B = "b" * 24
NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "state" / "books.sqlite3")
    database.initialize()
    return database


def _book(
    database: Database,
    book_id: str,
    *,
    source_format: str = "pdf",
    status: str = "needs_ocr",
) -> None:
    database.create_book(
        book_id=book_id,
        title=f"Book {book_id}",
        author="Author",
        source_format=source_format,
        content_sha256=f"hash-{book_id}",
        original_path=f"/books/{book_id}.{source_format}",
        status=status,
    )


def _job(
    database: Database,
    book_id: str = BOOK_A,
    *,
    total_pages: int = 3,
) -> dict[str, object]:
    _book(database, book_id)
    return database.queue_ocr_job(
        book_id,
        total_pages=total_pages,
        languages=("zh-Hans", "en-US"),
    )


def _claim(
    database: Database,
    book_id: str = BOOK_A,
    *,
    total_pages: int = 3,
    worker_id: str = "worker-a",
) -> dict[str, object]:
    _job(database, book_id, total_pages=total_pages)
    claimed = database.claim_next_ocr_job(worker_id, 60, now=NOW)
    assert claimed is not None
    return claimed


def test_initialize_migrates_without_changing_existing_book_or_passages(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state" / "books.sqlite3")
    database.initialize()
    _book(database, BOOK_A)
    passage = Passage(
        passage_id="existing-passage",
        book_id=BOOK_A,
        ordinal=0,
        text="原有段落",
        section="第一章",
        page_start=1,
        page_end=1,
        page_label="1",
        markdown_path=f"书库/20-解析文本/{BOOK_A}/正文.md",
        anchor="passage-0",
        text_sha256="passage-hash",
    )
    database.replace_passages(BOOK_A, [passage])
    with database._connection() as connection:
        connection.execute("DROP INDEX ocr_jobs_queue_idx")
        connection.execute("DROP TABLE ocr_pages")
        connection.execute("DROP TABLE ocr_jobs")

    database.initialize()

    assert database.get_book(BOOK_A)["status"] == "needs_ocr"
    assert database.count_passages(BOOK_A) == 1
    assert database.get_ocr_job(BOOK_A) is None
    with database.connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    assert {"ocr_jobs", "ocr_pages"} <= tables
    assert "ocr_jobs_queue_idx" in indexes


def test_queue_requires_existing_needs_ocr_pdf_and_serializes_languages(db: Database) -> None:
    _book(db, BOOK_A)

    queued = db.queue_ocr_job(BOOK_A, 4, ("zh-Hans", "en-US"))

    assert queued["status"] == "queued"
    assert queued["language_config"] == '["zh-Hans","en-US"]'
    assert queued["schema_version"] == 1

    with pytest.raises(ValueError, match="existing PDF.*needs_ocr"):
        db.queue_ocr_job(BOOK_B, 4, ("zh-Hans",))
    _book(db, BOOK_B, source_format="txt")
    with pytest.raises(ValueError, match="existing PDF.*needs_ocr"):
        db.queue_ocr_job(BOOK_B, 4, ("zh-Hans",))


def test_requeue_is_idempotent_and_resumes_without_deleting_checkpoints(
    db: Database,
) -> None:
    _claim(db)
    db.save_ocr_page(BOOK_A, "worker-a", 1, "i", "第一页", "sha-1", 0.9, 80)
    db.fail_ocr_job(BOOK_A, "worker-a", "page failed", 2)

    resumed = db.queue_ocr_job(BOOK_A, 3, ("zh-Hans", "en-US"))
    repeated = db.queue_ocr_job(BOOK_A, 3, ("zh-Hans", "en-US"))

    assert resumed["status"] == repeated["status"] == "queued"
    assert resumed["current_page"] == 2
    assert resumed["attempt_count"] == 1
    assert resumed["error"] is None
    assert [page["page_number"] for page in db.list_ocr_pages(BOOK_A)] == [1]


def test_concurrent_identical_queue_requests_are_idempotent(db: Database) -> None:
    _book(db, BOOK_A)
    barrier = threading.Barrier(2)

    def queue() -> dict[str, object]:
        barrier.wait()
        return db.queue_ocr_job(BOOK_A, 3, ("zh-Hans", "en-US"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: queue(), range(2)))

    assert [result["status"] for result in results] == ["queued", "queued"]
    assert [job["book_id"] for job in db.list_ocr_jobs()] == [BOOK_A]


def test_claim_is_fifo_exclusive_and_recovers_expired_lease(db: Database) -> None:
    _job(db, BOOK_B)
    _job(db, BOOK_A)
    assert [job["book_id"] for job in db.list_ocr_jobs()] == [BOOK_B, BOOK_A]

    first = db.claim_next_ocr_job("worker-a", 60, now=NOW)
    second = db.claim_next_ocr_job("worker-b", 60, now=NOW)

    assert first is not None and first["book_id"] == BOOK_B
    assert first["status"] == "running"
    assert first["attempt_count"] == 1
    assert second is not None and second["book_id"] == BOOK_A
    assert db.claim_next_ocr_job("worker-c", 60, now=NOW) is None

    recovered = db.claim_next_ocr_job(
        "worker-c", 30, now=NOW + timedelta(seconds=61)
    )
    assert recovered is not None and recovered["book_id"] == BOOK_B
    assert recovered["worker_id"] == "worker-c"
    assert recovered["attempt_count"] == 2


def test_concurrent_claimers_cannot_claim_the_same_job(db: Database) -> None:
    _job(db)
    barrier = threading.Barrier(2)

    def claim(worker_id: str) -> dict[str, object] | None:
        barrier.wait()
        return db.claim_next_ocr_job(worker_id, 60, now=NOW)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, ("worker-a", "worker-b")))

    assert len([result for result in results if result is not None]) == 1


def test_lease_renewal_requires_the_live_owner(db: Database) -> None:
    _claim(db)

    renewed = db.renew_ocr_lease(BOOK_A, "worker-a", 120, now=NOW)

    assert renewed["lease_expires_at"] > renewed["started_at"]
    with pytest.raises(ValueError, match="owner|lease"):
        db.renew_ocr_lease(BOOK_A, "worker-b", 120, now=NOW)
    with pytest.raises(ValueError, match="owner|lease"):
        db.renew_ocr_lease(
            BOOK_A,
            "worker-a",
            120,
            now=NOW + timedelta(seconds=121),
        )


def test_reclaimed_job_rejects_stale_worker_checkpoint(db: Database) -> None:
    _claim(db, total_pages=1)
    recovered = db.claim_next_ocr_job(
        "worker-b", 60, now=NOW + timedelta(seconds=61)
    )
    assert recovered is not None and recovered["worker_id"] == "worker-b"

    with pytest.raises(ValueError, match="owner"):
        db.save_ocr_page(
            BOOK_A, "worker-a", 1, None, "stale", "stale-sha", 0.1, 10
        )

    assert db.list_ocr_pages(BOOK_A) == []
    assert db.get_ocr_job(BOOK_A)["completed_pages"] == 0
    db.save_ocr_page(
        BOOK_A, "worker-b", 1, None, "current", "current-sha", 0.9, 10
    )
    assert db.list_ocr_pages(BOOK_A)[0]["text"] == "current"


def test_page_checkpoint_is_idempotent_ordered_and_counts_actual_rows(db: Database) -> None:
    _claim(db, total_pages=3)

    db.save_ocr_page(BOOK_A, "worker-a", 2, "ii", "第二页", "sha-2", 0.8, 90)
    db.save_ocr_page(BOOK_A, "worker-a", 1, None, "第一页", "sha-1", 0.9, 80)
    db.save_ocr_page(BOOK_A, "worker-a", 1, "i", "第一页修订", "sha-1b", 0.95, 85)

    pages = db.list_ocr_pages(BOOK_A)
    assert [page["page_number"] for page in pages] == [1, 2]
    assert pages[0]["text"] == "第一页修订"
    assert db.get_ocr_job(BOOK_A)["completed_pages"] == 2


def test_empty_page_text_is_a_completed_checkpoint(db: Database) -> None:
    _claim(db, total_pages=1)

    saved = db.save_ocr_page(
        BOOK_A, "worker-a", 1, None, "", "empty-sha", None, 0
    )

    assert saved["text"] == ""
    assert db.get_ocr_job(BOOK_A)["completed_pages"] == 1


def test_pause_request_and_owner_pause_transition_preserve_pages(db: Database) -> None:
    _claim(db)
    db.save_ocr_page(BOOK_A, "worker-a", 1, None, "one", "sha-1", 1.0, 1)

    requested = db.request_ocr_pause(BOOK_A)
    paused = db.pause_ocr_job(BOOK_A, "worker-a")

    assert requested["pause_requested"] == 1
    assert paused["status"] == "paused"
    assert paused["pause_requested"] == 0
    assert paused["worker_id"] is None
    assert len(db.list_ocr_pages(BOOK_A)) == 1
    with pytest.raises(ValueError, match="owner|transition"):
        db.pause_ocr_job(BOOK_A, "worker-a")


def test_failure_and_resume_persist_page_and_attempt_state(db: Database) -> None:
    _claim(db, total_pages=2)
    db.save_ocr_page(BOOK_A, "worker-a", 1, None, "one", "sha-1", 0.7, 50)

    failed = db.fail_ocr_job(BOOK_A, "worker-a", "Vision failed", 2)
    resumed = db.queue_ocr_job(BOOK_A, 2, ("zh-Hans", "en-US"))
    claimed = db.claim_next_ocr_job("worker-b", 60, now=NOW)

    assert failed["status"] == "failed"
    assert failed["current_page"] == 2
    assert failed["error"] == "Vision failed"
    assert resumed["status"] == "queued"
    assert resumed["current_page"] == 2
    assert claimed is not None and claimed["attempt_count"] == 2
    assert len(db.list_ocr_pages(BOOK_A)) == 1


def test_completion_requires_all_pages_and_cleanup_is_explicit(db: Database) -> None:
    _claim(db, total_pages=2)
    db.save_ocr_page(BOOK_A, "worker-a", 1, None, "one", "sha-1", 0.7, 50)
    with pytest.raises(ValueError, match="all pages|complete"):
        db.complete_ocr_job(BOOK_A, "worker-a")
    db.save_ocr_page(BOOK_A, "worker-a", 2, None, "two", "sha-2", 0.8, 60)
    with pytest.raises(ValueError, match="searchable"):
        db.complete_ocr_job(BOOK_A, "worker-a")
    db.update_book_status(BOOK_A, "keyword_only")

    completed = db.complete_ocr_job(BOOK_A, "worker-a")

    assert completed["status"] == "completed"
    assert completed["completed_pages"] == 2
    assert len(db.list_ocr_pages(BOOK_A)) == 2
    db.delete_ocr_page_checkpoints(BOOK_A)
    assert db.list_ocr_pages(BOOK_A) == []
    assert db.get_ocr_job(BOOK_A)["completed_pages"] == 2


def test_deleting_a_book_cascades_ocr_job_and_pages(db: Database) -> None:
    _claim(db, total_pages=1)
    db.save_ocr_page(BOOK_A, "worker-a", 1, None, "one", "sha-1", 1.0, 1)

    with db._connection() as connection:
        connection.execute("DELETE FROM books WHERE book_id = ?", (BOOK_A,))

    assert db.get_ocr_job(BOOK_A) is None
    assert db.list_ocr_pages(BOOK_A) == []


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda db: db.queue_ocr_job(True, 1, ("zh-Hans",)), "book_id"),
        (lambda db: db.queue_ocr_job("A" * 24, 1, ("zh-Hans",)), "book_id"),
        (lambda db: db.queue_ocr_job(BOOK_A, True, ("zh-Hans",)), "total_pages"),
        (lambda db: db.queue_ocr_job(BOOK_A, 1, ["zh-Hans"]), "languages"),
        (lambda db: db.queue_ocr_job(BOOK_A, 1, (" ",)), "languages"),
        (lambda db: db.queue_ocr_job(BOOK_A, 1, ("zh-Hans",), 2), "schema_version"),
        (lambda db: db.claim_next_ocr_job(" ", 1), "worker_id"),
        (lambda db: db.claim_next_ocr_job("worker", True), "lease_seconds"),
    ],
)
def test_strict_job_input_validation(db: Database, call, match: str) -> None:
    _book(db, BOOK_A)
    with pytest.raises(ValueError, match=match):
        call(db)


def test_oversized_native_numbers_raise_validation_errors(db: Database) -> None:
    _book(db, BOOK_A)
    with pytest.raises(ValueError, match="total_pages"):
        db.queue_ocr_job(BOOK_A, 2**63, ("zh-Hans",))
    db.queue_ocr_job(BOOK_A, 1, ("zh-Hans",))
    with pytest.raises(ValueError, match="lease_seconds"):
        db.claim_next_ocr_job("worker-a", 10**100, now=NOW)
    claimed = db.claim_next_ocr_job("worker-a", 60, now=NOW)
    assert claimed is not None
    with pytest.raises(ValueError, match="mean_confidence"):
        db.save_ocr_page(
            BOOK_A, "worker-a", 1, None, "", "sha", 10**100, 0
        )
    with pytest.raises(ValueError, match="duration_ms"):
        db.save_ocr_page(
            BOOK_A, "worker-a", 1, None, "", "sha", None, 2**63
        )


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((0, None, "text", "sha", 0.5, 1), "page_number"),
        ((4, None, "text", "sha", 0.5, 1), "total_pages"),
        ((1, None, 1, "sha", 0.5, 1), "text"),
        ((1, None, "text", " ", 0.5, 1), "text_sha256"),
        ((1, None, "text", "sha", float("nan"), 1), "mean_confidence"),
        ((1, None, "text", "sha", 1.1, 1), "mean_confidence"),
        ((1, None, "text", "sha", 0.5, True), "duration_ms"),
    ],
)
def test_strict_page_input_validation(db: Database, args, match: str) -> None:
    _claim(db)
    with pytest.raises(ValueError, match=match):
        db.save_ocr_page(BOOK_A, "worker-a", *args)


def test_invalid_state_transitions_and_owner_checks(db: Database) -> None:
    _job(db)
    with pytest.raises(ValueError, match="transition"):
        db.request_ocr_pause(BOOK_A)
    claimed = db.claim_next_ocr_job("worker-a", 60, now=NOW)
    assert claimed is not None
    with pytest.raises(ValueError, match="owner"):
        db.fail_ocr_job(BOOK_A, "worker-b", "failure", 1)
    with pytest.raises(ValueError, match="error"):
        db.fail_ocr_job(BOOK_A, "worker-a", " ", 1)
    db.fail_ocr_job(BOOK_A, "worker-a", "failure", 1)
    with pytest.raises(ValueError, match="transition"):
        db.complete_ocr_job(BOOK_A, "worker-a")


def test_ocr_query_results_are_plain_json_safe_values(db: Database) -> None:
    _claim(db, total_pages=1)
    page = db.save_ocr_page(
        BOOK_A, "worker-a", 1, None, "", "empty-sha", None, 0
    )

    values = [db.get_ocr_job(BOOK_A), db.list_ocr_jobs(), page, db.list_ocr_pages(BOOK_A)]

    json.dumps(values, allow_nan=False)
    assert type(values[0]) is dict
    assert all(type(item) is dict for item in values[1])
    assert type(page) is dict


def test_ocr_operations_explicitly_close_connections(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_connection = db.connect()

    class TrackedConnection:
        def __init__(self) -> None:
            self.closed = False

        def __enter__(self):
            raw_connection.__enter__()
            return self

        def __exit__(self, *args):
            return raw_connection.__exit__(*args)

        def execute(self, *args, **kwargs):
            return raw_connection.execute(*args, **kwargs)

        def close(self) -> None:
            self.closed = True
            raw_connection.close()

    tracked = TrackedConnection()
    monkeypatch.setattr(db, "connect", lambda: tracked)

    assert db.get_ocr_job(BOOK_A) is None

    assert tracked.closed
