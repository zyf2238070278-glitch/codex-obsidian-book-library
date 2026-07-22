from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import fitz
import pytest

import book_agent.ocr.service as service_module
from book_agent.config import AppPaths
from book_agent.ocr.models import OcrJobSummary
from book_agent.ocr.service import OcrService
from book_agent.storage import Database


NOW = datetime(2026, 7, 14, 4, 5, 6, tzinfo=timezone.utc)
BOOK_A = "a" * 24
BOOK_B = "b" * 24


class FakeProcess:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid


class RecordingPopen:
    def __init__(self, *, pid: int = 4242, error: BaseException | None = None) -> None:
        self.pid = pid
        self.error = error
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self._lock = threading.Lock()

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        with self._lock:
            self.calls.append((list(argv), dict(kwargs)))
        if self.error is not None:
            raise self.error
        return FakeProcess(self.pid)


@pytest.fixture
def app(tmp_path: Path) -> tuple[OcrService, Database, RecordingPopen, AppPaths]:
    paths = AppPaths.from_root(tmp_path / "project", tmp_path / "vault")
    paths.root.mkdir(parents=True)
    paths.originals.mkdir(parents=True)
    database = Database(paths.database, root=paths.root)
    database.initialize()
    launcher = RecordingPopen()
    service = OcrService(
        paths,
        database,
        popen_factory=launcher,
        now_factory=lambda: NOW,
        pid_probe=lambda pid: pid == launcher.pid,
    )
    return service, database, launcher, paths


def _pdf(path: Path, pages: int, *, encrypted: bool = False) -> str:
    document = fitz.open()
    for _ in range(pages):
        document.new_page()
    if encrypted:
        document.save(
            path,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="reader",
        )
    else:
        document.save(path)
    document.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _book(
    database: Database,
    paths: AppPaths,
    book_id: str,
    *,
    pages: int = 2,
    status: str = "needs_ocr",
    source_format: str = "pdf",
    encrypted: bool = False,
) -> Path:
    suffix = ".pdf" if source_format == "pdf" else f".{source_format}"
    original = paths.originals / f"{book_id}{suffix}"
    if source_format == "pdf":
        digest = _pdf(original, pages, encrypted=encrypted)
    else:
        original.write_text("not a PDF", encoding="utf-8")
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
    database.create_book(
        book_id=book_id,
        title=f"Book {book_id}",
        author="Author",
        source_format=source_format,
        content_sha256=digest,
        original_path=str(original.absolute()),
        status=status,
    )
    return original


def _payload(summary: OcrJobSummary) -> dict[str, object]:
    return {
        "book_id": summary.book_id,
        "status": summary.status,
        "total_pages": summary.total_pages,
        "completed_pages": summary.completed_pages,
        "percent_complete": summary.percent_complete,
        "estimated_remaining_seconds": summary.estimated_remaining_seconds,
    }


def test_start_ocr_persists_job_and_returns_without_running_pages(app) -> None:
    service, database, launcher, paths = app
    original = _book(database, paths, BOOK_A, pages=3)
    before = original.read_bytes()

    result = service.start_ocr(BOOK_A)

    assert isinstance(result, OcrJobSummary)
    assert _payload(result) == {
        "book_id": BOOK_A,
        "status": "queued",
        "total_pages": 3,
        "completed_pages": 0,
        "percent_complete": 0.0,
        "estimated_remaining_seconds": None,
    }
    assert launcher.calls[0][0][:4] == [
        sys.executable,
        "-I",
        "-c",
        service_module._BOUND_WORKER_BOOTSTRAP,
    ]
    assert database.list_ocr_pages(BOOK_A) == []
    assert original.read_bytes() == before


@pytest.mark.parametrize(
    ("book_id", "message"),
    [
        ("A" * 24, "24 lowercase hexadecimal"),
        ("a" * 23, "24 lowercase hexadecimal"),
        ("a" * 25, "24 lowercase hexadecimal"),
        (True, "24 lowercase hexadecimal"),
    ],
)
def test_start_rejects_invalid_book_ids_without_side_effects(
    app, book_id: object, message: str
) -> None:
    service, database, launcher, _ = app

    with pytest.raises(ValueError, match=message):
        service.start_ocr(book_id)  # type: ignore[arg-type]

    assert database.list_ocr_jobs() == []
    assert launcher.calls == []


def test_start_rejects_unknown_non_pdf_and_non_eligible_book_states(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A, source_format="txt")
    _book(database, paths, BOOK_B, status="ready")

    with pytest.raises(ValueError, match="Unknown book_id"):
        service.start_ocr("c" * 24)
    with pytest.raises(ValueError, match="PDF"):
        service.start_ocr(BOOK_A)
    with pytest.raises(ValueError, match="already searchable"):
        service.start_ocr(BOOK_B)

    assert database.list_ocr_jobs() == []
    assert launcher.calls == []


@pytest.mark.parametrize("status", ["processing", "failed"])
def test_start_rejects_other_book_states_stably(app, status: str) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A, status=status)

    with pytest.raises(ValueError, match=f"status {status} is not eligible"):
        service.start_ocr(BOOK_A)

    assert database.list_ocr_jobs() == []
    assert launcher.calls == []


def test_start_rejects_external_symlink_mismatched_and_encrypted_pdfs(app) -> None:
    service, database, launcher, paths = app
    external = paths.vault / "external.pdf"
    _pdf(external, 1)
    database.create_book(
        book_id=BOOK_A,
        title="External",
        author=None,
        source_format="pdf",
        content_sha256=hashlib.sha256(external.read_bytes()).hexdigest(),
        original_path=str(external),
        status="needs_ocr",
    )
    with pytest.raises(ValueError, match="managed originals"):
        service.start_ocr(BOOK_A)

    target = paths.originals / "target.pdf"
    _pdf(target, 1)
    link = paths.originals / f"{BOOK_B}.pdf"
    link.symlink_to(target)
    database.create_book(
        book_id=BOOK_B,
        title="Link",
        author=None,
        source_format="pdf",
        content_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        original_path=str(link),
        status="needs_ocr",
    )
    with pytest.raises(ValueError, match="symlink|safely"):
        service.start_ocr(BOOK_B)

    encrypted_id = "c" * 24
    _book(database, paths, encrypted_id, encrypted=True)
    with pytest.raises(ValueError, match="encrypted"):
        service.start_ocr(encrypted_id)

    mismatch_id = "d" * 24
    original = _book(database, paths, mismatch_id)
    original.write_bytes(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        service.start_ocr(mismatch_id)

    assert database.list_ocr_jobs() == []
    assert launcher.calls == []


def test_start_rejects_corrupt_pdf(app) -> None:
    service, database, _, paths = app
    corrupt = paths.originals / f"{BOOK_A}.pdf"
    corrupt.write_bytes(b"%PDF-broken")
    database.create_book(
        book_id=BOOK_A,
        title="Broken",
        author=None,
        source_format="pdf",
        content_sha256=hashlib.sha256(corrupt.read_bytes()).hexdigest(),
        original_path=str(corrupt),
        status="needs_ocr",
    )
    with pytest.raises(ValueError, match="invalid or damaged"):
        service.start_ocr(BOOK_A)


def test_pdf_changed_in_place_during_page_count_is_rejected_even_if_mtime_restored(
    app, monkeypatch
) -> None:
    service, database, launcher, paths = app
    original = _book(database, paths, BOOK_A, pages=2)
    original_bytes = original.read_bytes()
    original_stat = original.stat()
    actual_open = service_module.fitz.open

    class MutatingDocument:
        def __init__(self, document: fitz.Document) -> None:
            self.document = document
            self.needs_pass = document.needs_pass

        @property
        def page_count(self) -> int:
            count = self.document.page_count
            changed = bytearray(original_bytes)
            changed[-2] ^= 1
            original.write_bytes(changed)
            os.utime(
                original,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            return count

        def close(self) -> None:
            self.document.close()

    monkeypatch.setattr(
        service_module.fitz,
        "open",
        lambda *args, **kwargs: MutatingDocument(actual_open(*args, **kwargs)),
    )

    with pytest.raises(ValueError, match="changed during validation|content hash"):
        service.start_ocr(BOOK_A)

    assert database.list_ocr_jobs() == []
    assert launcher.calls == []
    assert original.read_bytes() != original_bytes


def test_transient_pdf_change_and_byte_restore_is_detected_by_fd_metadata(
    app, monkeypatch
) -> None:
    service, database, launcher, paths = app
    original = _book(database, paths, BOOK_A, pages=2)
    original_bytes = original.read_bytes()
    original_stat = original.stat()
    actual_open = service_module.fitz.open

    class TransientDocument:
        def __init__(self, document: fitz.Document) -> None:
            self.document = document
            self.needs_pass = document.needs_pass

        @property
        def page_count(self) -> int:
            count = self.document.page_count
            changed = bytearray(original_bytes)
            changed[-2] ^= 1
            original.write_bytes(changed)
            time.sleep(0.001)
            original.write_bytes(original_bytes)
            os.utime(
                original,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            return count

        def close(self) -> None:
            self.document.close()

    monkeypatch.setattr(
        service_module.fitz,
        "open",
        lambda *args, **kwargs: TransientDocument(actual_open(*args, **kwargs)),
    )

    with pytest.raises(ValueError, match="changed during validation"):
        service.start_ocr(BOOK_A)

    assert database.list_ocr_jobs() == []
    assert launcher.calls == []
    assert original.read_bytes() == original_bytes



def test_start_rejects_zero_page_pdf_reported_by_parser(app, monkeypatch) -> None:
    service, database, _, paths = app
    _book(database, paths, BOOK_A, pages=1)

    class EmptyDocument:
        needs_pass = False
        page_count = 0

        def close(self) -> None:
            pass

    monkeypatch.setattr("book_agent.ocr.service.fitz.open", lambda *_: EmptyDocument())

    with pytest.raises(ValueError, match="zero pages"):
        service.start_ocr(BOOK_A)


def test_idempotent_start_and_resume_preserve_checkpoints(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A, pages=3)
    service.start_ocr(BOOK_A)
    claimed = database.claim_next_ocr_job("worker", 60, now=NOW)
    assert claimed is not None
    database.save_ocr_page(
        BOOK_A, "worker", 1, "1", "page", "sha", 0.9, 100, now=NOW
    )
    database.fail_ocr_job(BOOK_A, "worker", "retry", 2, now=NOW)

    resumed = service.start_ocr(BOOK_A)
    repeated = service.start_ocr(BOOK_A)

    assert resumed.status == repeated.status == "queued"
    assert resumed.completed_pages == repeated.completed_pages == 1
    assert [row["page_number"] for row in database.list_ocr_pages(BOOK_A)] == [1]
    assert len(launcher.calls) == 1


def test_existing_queued_job_revalidates_hash_without_mutating_job(app) -> None:
    service, database, launcher, paths = app
    original = _book(database, paths, BOOK_A, pages=2)
    service.start_ocr(BOOK_A)
    before_job = database.get_ocr_job(BOOK_A)
    original.write_bytes(b"%PDF-replaced-and-corrupt")
    replaced = original.read_bytes()

    with pytest.raises(ValueError, match="content hash"):
        service.start_ocr(BOOK_A)

    assert database.get_ocr_job(BOOK_A) == before_job
    assert database.list_ocr_pages(BOOK_A) == []
    assert original.read_bytes() == replaced
    assert len(launcher.calls) == 1


def test_existing_running_job_revalidates_corrupt_pdf_without_mutation(app) -> None:
    service, database, launcher, paths = app
    original = _book(database, paths, BOOK_A, pages=2)
    service.start_ocr(BOOK_A)
    claimed = database.claim_next_ocr_job("worker", 60, now=NOW)
    assert claimed is not None
    original.write_bytes(b"%PDF-corrupt")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    with database._connection() as connection:
        connection.execute(
            "UPDATE books SET content_sha256=? WHERE book_id=?", (digest, BOOK_A)
        )
    before_job = database.get_ocr_job(BOOK_A)

    with pytest.raises(ValueError, match="invalid or damaged"):
        service.start_ocr(BOOK_A)

    assert database.get_ocr_job(BOOK_A) == before_job
    assert database.list_ocr_pages(BOOK_A) == []
    assert original.read_bytes() == b"%PDF-corrupt"
    assert len(launcher.calls) == 1


def test_existing_job_rejects_changed_page_count_before_spawn(app) -> None:
    service, database, launcher, paths = app
    original = _book(database, paths, BOOK_A, pages=2)
    service.start_ocr(BOOK_A)
    original.unlink()
    digest = _pdf(original, 3)
    with database._connection() as connection:
        connection.execute(
            "UPDATE books SET content_sha256=? WHERE book_id=?", (digest, BOOK_A)
        )
    before_job = database.get_ocr_job(BOOK_A)
    before = original.read_bytes()

    with pytest.raises(ValueError, match="page count.*job|job.*page count"):
        service.start_ocr(BOOK_A)

    assert database.get_ocr_job(BOOK_A) == before_job
    assert original.read_bytes() == before
    assert len(launcher.calls) == 1


def test_pending_batch_revalidates_existing_job_and_reports_replacement(app) -> None:
    service, database, launcher, paths = app
    original = _book(database, paths, BOOK_A, pages=2)
    database.queue_ocr_job(BOOK_A, 2, ("zh-Hans", "en-US"))
    original.write_bytes(b"replaced")
    before_job = database.get_ocr_job(BOOK_A)

    result = service.start_pending_ocr()

    assert result["jobs"] == []
    assert result["error_count"] == 1
    assert result["errors"][0]["book_id"] == BOOK_A
    assert "content hash" in result["errors"][0]["error"]
    assert database.get_ocr_job(BOOK_A) == before_job
    assert database.list_ocr_pages(BOOK_A) == []
    assert launcher.calls == []


def test_resume_paused_job_preserves_checkpoints(app) -> None:
    service, database, _, paths = app
    _book(database, paths, BOOK_A, pages=3)
    service.start_ocr(BOOK_A)
    claimed = database.claim_next_ocr_job("worker", 60, now=NOW)
    assert claimed is not None
    database.save_ocr_page(
        BOOK_A, "worker", 1, "1", "page", "sha", 0.9, 100, now=NOW
    )
    database.pause_ocr_job(BOOK_A, "worker", now=NOW)

    resumed = service.start_ocr(BOOK_A)

    assert resumed.status == "queued"
    assert resumed.completed_pages == 1
    assert [row["page_number"] for row in database.list_ocr_pages(BOOK_A)] == [1]


@pytest.mark.parametrize("terminal_status", ["paused", "failed"])
def test_explicit_resume_clears_pause_flag_and_preserves_checkpoints(
    app, terminal_status: str
) -> None:
    service, database, _, paths = app
    _book(database, paths, BOOK_A, pages=2)
    database.queue_ocr_job(BOOK_A, 2, ("zh-Hans", "en-US"))
    claimed = database.claim_next_ocr_job("worker", 60, now=NOW)
    assert claimed is not None
    database.save_ocr_page(
        BOOK_A, "worker", 1, "1", "checkpoint", "sha", 0.8, 100, now=NOW
    )
    if terminal_status == "paused":
        database.pause_ocr_job(BOOK_A, "worker", now=NOW)
    else:
        database.fail_ocr_job(BOOK_A, "worker", "failure", 2, now=NOW)
    with database._connection() as connection:
        connection.execute(
            "UPDATE ocr_jobs SET pause_requested=1 WHERE book_id=?", (BOOK_A,)
        )

    resumed = service.start_ocr(BOOK_A)

    job = database.get_ocr_job(BOOK_A)
    assert resumed.status == "queued"
    assert job["pause_requested"] == 0
    assert job["worker_id"] is None
    assert job["lease_expires_at"] is None
    assert [page["page_number"] for page in database.list_ocr_pages(BOOK_A)] == [1]


def test_start_completed_job_is_a_noop(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A, pages=1)
    service.start_ocr(BOOK_A)
    claimed = database.claim_next_ocr_job("worker", 60, now=NOW)
    assert claimed is not None
    database.save_ocr_page(
        BOOK_A, "worker", 1, "1", "page", "sha", 0.9, 100, now=NOW
    )
    database.update_book_status(BOOK_A, "ready")
    database.complete_ocr_job(BOOK_A, "worker", now=NOW)

    result = service.start_ocr(BOOK_A)

    assert result.status == "completed"
    assert len(launcher.calls) == 1


def test_start_pending_is_bounded_ordered_and_json_safe(app) -> None:
    service, database, launcher, paths = app
    for book_id in ("c" * 24, BOOK_A, BOOK_B):
        _book(database, paths, book_id, pages=1)

    first = service.start_pending_ocr(limit=2, offset=0)
    second = service.start_pending_ocr(limit=2, offset=2)

    assert [item["book_id"] for item in first["jobs"]] == [BOOK_A, BOOK_B]
    assert first["has_more"] is True
    assert first["next_offset"] == 2
    assert [item["book_id"] for item in second["jobs"]] == ["c" * 24]
    assert second["has_more"] is False
    assert len(launcher.calls) == 1
    json.dumps(first, ensure_ascii=False, allow_nan=False)
    assert all(database.list_ocr_pages(book_id) == [] for book_id in (BOOK_A, BOOK_B, "c" * 24))


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(True, 0), (0, 0), (101, 0), (1, True), (1, -1)],
)
def test_start_pending_rejects_invalid_bounds(app, limit: object, offset: object) -> None:
    service, database, launcher, _ = app
    with pytest.raises(ValueError, match="limit|offset"):
        service.start_pending_ocr(limit=limit, offset=offset)  # type: ignore[arg-type]
    assert database.list_ocr_jobs() == []
    assert launcher.calls == []


def test_status_single_all_bounds_and_recent_duration_estimate(app) -> None:
    service, database, _, paths = app
    _book(database, paths, BOOK_A, pages=8)
    service.start_ocr(BOOK_A)
    claimed = database.claim_next_ocr_job("worker", 60, now=NOW)
    assert claimed is not None
    for page_number, duration in enumerate((1000, 2000, 3000, 4000, 5000), 1):
        database.save_ocr_page(
            BOOK_A,
            "worker",
            page_number,
            str(page_number),
            "text",
            f"sha-{page_number}",
            0.9,
            duration,
            now=NOW,
        )

    single = service.status(BOOK_A)
    all_jobs = service.status(limit=1, offset=0)

    assert isinstance(single, OcrJobSummary)
    assert single.estimated_remaining_seconds == 9
    assert single.percent_complete == 62.5
    assert all_jobs["count"] == 1
    assert all_jobs["jobs"][0]["estimated_remaining_seconds"] == 9
    assert "text" not in json.dumps(all_jobs)
    database.list_ocr_pages = lambda *_: pytest.fail("unbounded page scan")  # type: ignore[method-assign]
    assert service.status(BOOK_A).completed_pages == 5


def test_status_has_no_estimate_before_five_pages_and_rejects_bounds(app) -> None:
    service, database, _, paths = app
    _book(database, paths, BOOK_A, pages=2)
    service.start_ocr(BOOK_A)
    assert service.status(BOOK_A).estimated_remaining_seconds is None
    with pytest.raises(ValueError, match="limit"):
        service.status(limit=101)
    with pytest.raises(ValueError, match="offset"):
        service.status(offset=-1)
    with pytest.raises(ValueError, match="Unknown OCR job"):
        service.status(BOOK_B)


def test_status_rejects_corrupt_duration_data(app) -> None:
    service, database, _, paths = app
    _book(database, paths, BOOK_A, pages=6)
    service.start_ocr(BOOK_A)
    with database._connection() as connection:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.executemany(
            """
            INSERT INTO ocr_pages (
                book_id, page_number, text, text_sha256, duration_ms, completed_at
            ) VALUES (?, ?, '', ?, ?, ?)
            """,
            [
                (BOOK_A, number, f"sha-{number}", -1 if number == 5 else 10, "now")
                for number in range(1, 6)
            ],
        )
        connection.execute(
            "UPDATE ocr_jobs SET completed_pages=5 WHERE book_id=?", (BOOK_A,)
        )
    with pytest.raises(ValueError, match="duration"):
        service.status(BOOK_A)


def test_pause_queued_running_and_terminal_jobs(app) -> None:
    service, database, _, paths = app
    _book(database, paths, BOOK_A, pages=2)
    service.start_ocr(BOOK_A)

    paused = service.pause(BOOK_A)
    repeated = service.pause(BOOK_A)
    assert paused.status == repeated.status == "paused"
    assert database.get_ocr_job(BOOK_A)["pause_requested"] == 0

    service.start_ocr(BOOK_A)
    claimed = database.claim_next_ocr_job("worker", 60, now=NOW)
    assert claimed is not None
    running = service.pause(BOOK_A)
    assert running.status == "running"
    assert database.get_ocr_job(BOOK_A)["pause_requested"] == 1

    with pytest.raises(ValueError, match="Unknown OCR job"):
        service.pause(BOOK_B)


def test_detached_launcher_uses_exact_safe_process_contract(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)

    service.start_ocr(BOOK_A)

    argv, kwargs = launcher.calls[0]
    assert argv[:4] == [
        sys.executable,
        "-I",
        "-c",
        service_module._BOUND_WORKER_BOOTSTRAP,
    ]
    assert argv[-1] == sys.executable
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == "/"
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["close_fds"] is True
    assert len(kwargs["pass_fds"]) == 1
    assert argv[4] == str(kwargs["pass_fds"][0])
    assert isinstance(kwargs["stdout"], int)
    assert kwargs["stderr"] == kwargs["stdout"]
    assert kwargs["env"] == {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "BOOK_LIBRARY_ROOT": ".",
        "BOOK_LIBRARY_OBSIDIAN_VAULT": str(paths.vault.absolute()),
        "BOOK_LIBRARY_LIGHT_OCR_NODE": "",
    }
    log = paths.ocr_logs / "worker.log"
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert log.is_file() and not log.is_symlink()


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS fchdir/exec")
def test_macos_bootstrap_executes_worker_from_renamed_original_root(
    tmp_path: Path,
) -> None:
    original = tmp_path / "project"
    original_worker = original / "book_agent" / "ocr_worker.py"
    original_worker.parent.mkdir(parents=True)
    (original_worker.parent / "__init__.py").write_text("", encoding="utf-8")
    original_worker.write_text(
        "import json, os, sys\n"
        "print(json.dumps({'source':'original','inode':os.stat('.').st_ino,"
        "'orig_argv':sys.orig_argv}))\n",
        encoding="utf-8",
    )
    descriptor = os.open(original, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    identity = os.fstat(descriptor)
    moved = tmp_path / "moved-original"
    original.rename(moved)
    replacement_worker = original / "book_agent" / "ocr_worker.py"
    replacement_worker.parent.mkdir(parents=True)
    (replacement_worker.parent / "__init__.py").write_text("", encoding="utf-8")
    replacement_worker.write_text("print('replacement')\n", encoding="utf-8")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "BOOK_LIBRARY_ROOT": ".",
        "BOOK_LIBRARY_OBSIDIAN_VAULT": str(tmp_path / "vault"),
        "BOOK_LIBRARY_LIGHT_OCR_NODE": "",
    }
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                service_module._BOUND_WORKER_BOOTSTRAP,
                str(descriptor),
                str(identity.st_dev),
                str(identity.st_ino),
                sys.executable,
            ],
            cwd="/",
            pass_fds=(descriptor,),
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    finally:
        os.close(descriptor)
    payload = json.loads(completed.stdout)
    assert payload["source"] == "original"
    assert payload["inode"] == identity.st_ino
    assert payload["orig_argv"][-3:] == [sys.executable, "-m", "book_agent.ocr_worker"]


def test_worker_environment_preserves_validated_light_ocr_node(
    app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _, _ = app
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\n", encoding="utf-8")
    node.chmod(0o755)
    monkeypatch.setenv("BOOK_LIBRARY_LIGHT_OCR_NODE", str(node))

    environment = service._worker_environment()

    assert environment["BOOK_LIBRARY_LIGHT_OCR_NODE"] == str(node.resolve())


def test_queue_and_pause_refresh_catalog_status(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path / "project", tmp_path / "vault")
    paths.originals.mkdir(parents=True)
    database = Database(paths.database, root=paths.root)
    database.initialize()
    _book(database, paths, BOOK_A)
    catalog_updates: list[str] = []

    class Catalog:
        def sync_book(self, book: dict[str, object]) -> Path:
            catalog_updates.append(str(book["book_id"]))
            return Path("card.md")

    service = OcrService(
        paths,
        database,
        catalog=Catalog(),
        popen_factory=RecordingPopen(),
        pid_probe=lambda _: True,
    )

    service.start_ocr(BOOK_A)
    service.pause(BOOK_A)

    assert catalog_updates == [BOOK_A, BOOK_A]


def test_launcher_cwd_fd_stays_on_original_root_during_path_replacement(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project", tmp_path / "vault")
    paths.root.mkdir(parents=True)
    paths.originals.mkdir(parents=True)
    database = Database(paths.database, root=paths.root)
    database.initialize()
    _book(database, paths, BOOK_A)
    database.queue_ocr_job(BOOK_A, 2, ("zh-Hans", "en-US"))
    original_identity = (paths.root.stat().st_dev, paths.root.stat().st_ino)
    moved = tmp_path / "original-project"
    observed: dict[str, object] = {}

    class ReplacingPopen:
        def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
            root_info = os.fstat(kwargs["pass_fds"][0])
            observed["root_identity"] = (root_info.st_dev, root_info.st_ino)
            observed["cwd"] = kwargs["cwd"]
            observed["env_root"] = kwargs["env"]["BOOK_LIBRARY_ROOT"]
            paths.root.rename(moved)
            paths.root.mkdir()
            return FakeProcess()

    service = OcrService(
        paths,
        database,
        popen_factory=ReplacingPopen(),
        pid_probe=lambda _: True,
    )
    try:
        assert service._ensure_worker_started() is True
        assert observed["root_identity"] == original_identity
        assert observed["cwd"] == "/"
        assert observed["env_root"] == "."
    finally:
        paths.root.rmdir()
        moved.rename(paths.root)


def test_worker_log_is_append_only_and_launch_descriptors_are_closed(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    paths.ocr_logs.mkdir(parents=True)
    log = paths.ocr_logs / "worker.log"
    log.write_text("existing\n", encoding="utf-8")

    service.start_ocr(BOOK_A)

    descriptor = launcher.calls[0][1]["stdout"]
    assert log.read_text(encoding="utf-8") == "existing\n"
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_launcher_failure_keeps_job_queued_and_does_not_claim_success(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path / "project", tmp_path / "vault")
    paths.root.mkdir(parents=True)
    paths.originals.mkdir(parents=True)
    database = Database(paths.database, root=paths.root)
    database.initialize()
    launcher = RecordingPopen(error=OSError("spawn denied"))
    service = OcrService(paths, database, popen_factory=launcher)
    _book(database, paths, BOOK_A)

    with pytest.raises(RuntimeError, match="spawn denied"):
        service.start_ocr(BOOK_A)

    assert database.get_ocr_job(BOOK_A)["status"] == "queued"
    assert database.list_ocr_pages(BOOK_A) == []
    assert not (paths.ocr / "worker.json").exists()
    assert not list(paths.ocr.glob(".worker-marker-*.tmp"))


@pytest.mark.parametrize("failure", [OSError("post marker failed"), KeyboardInterrupt()])
def test_post_spawn_atomic_replace_failure_keeps_complete_pre_marker(
    app, monkeypatch, failure: BaseException
) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    actual_replace = service_module.os.replace
    replace_calls = 0

    def fail_second_replace(*args: Any, **kwargs: Any) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise failure
        actual_replace(*args, **kwargs)

    monkeypatch.setattr(service_module.os, "replace", fail_second_replace)

    with pytest.raises(type(failure), match="post marker failed" if isinstance(failure, OSError) else None):
        service.start_ocr(BOOK_A)

    marker = json.loads((paths.ocr / "worker.json").read_text(encoding="utf-8"))
    assert marker["state"] == "launching"
    assert marker["pid"] is None
    assert marker["token"]
    assert stat.S_IMODE((paths.ocr / "worker.json").stat().st_mode) == 0o600
    assert not list(paths.ocr.glob(".worker-marker-*.tmp"))
    monkeypatch.setattr(service_module.os, "replace", actual_replace)
    service.start_ocr(BOOK_A)
    assert len(launcher.calls) == 1
    assert database.get_ocr_job(BOOK_A)["status"] == "queued"


def test_pre_spawn_marker_write_failure_never_spawns_and_cleans_temp(
    app, monkeypatch
) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    actual_write = service_module.os.write
    failed = False

    def fail_first_write(descriptor: int, data: bytes) -> int:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("pre marker write failed")
        return actual_write(descriptor, data)

    monkeypatch.setattr(service_module.os, "write", fail_first_write)

    with pytest.raises(OSError, match="pre marker write failed"):
        service.start_ocr(BOOK_A)

    assert launcher.calls == []
    assert not (paths.ocr / "worker.json").exists()
    assert not list(paths.ocr.glob(".worker-marker-*.tmp"))
    assert database.get_ocr_job(BOOK_A)["status"] == "queued"


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_pre_spawn_atomic_marker_failure_cleans_temp_and_never_spawns(
    app, monkeypatch, operation: str
) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    actual = getattr(service_module.os, operation)
    calls = 0

    def fail_first(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(f"pre marker {operation} failed")
        return actual(*args, **kwargs)

    monkeypatch.setattr(service_module.os, operation, fail_first)

    with pytest.raises(OSError, match=f"pre marker {operation} failed"):
        service.start_ocr(BOOK_A)

    assert launcher.calls == []
    assert not (paths.ocr / "worker.json").exists()
    assert not list(paths.ocr.glob(".worker-marker-*.tmp"))
    assert database.get_ocr_job(BOOK_A)["status"] == "queued"


@pytest.mark.parametrize(("operation", "failure_call"), [("write", 2), ("fsync", 3)])
def test_post_spawn_atomic_marker_failure_preserves_pre_marker_and_blocks_retry(
    app, monkeypatch, operation: str, failure_call: int
) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    actual = getattr(service_module.os, operation)
    calls = 0

    def fail_post(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError(f"post marker {operation} failed")
        return actual(*args, **kwargs)

    monkeypatch.setattr(service_module.os, operation, fail_post)

    with pytest.raises(OSError, match=f"post marker {operation} failed"):
        service.start_ocr(BOOK_A)

    marker = json.loads((paths.ocr / "worker.json").read_text(encoding="utf-8"))
    assert marker["state"] == "launching"
    assert marker["pid"] is None
    assert not list(paths.ocr.glob(".worker-marker-*.tmp"))
    monkeypatch.setattr(service_module.os, operation, actual)
    service.start_ocr(BOOK_A)
    assert len(launcher.calls) == 1
    assert database.get_ocr_job(BOOK_A)["status"] == "queued"


@pytest.mark.parametrize(
    ("failure_call", "expected_state", "expected_spawn_count"),
    [(2, "launching", 0), (4, "running", 1)],
)
def test_directory_fsync_failure_leaves_a_complete_recovery_marker(
    app,
    monkeypatch,
    failure_call: int,
    expected_state: str,
    expected_spawn_count: int,
) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    actual_fsync = service_module.os.fsync
    calls = 0

    def fail_directory_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("directory marker fsync failed")
        actual_fsync(descriptor)

    monkeypatch.setattr(service_module.os, "fsync", fail_directory_sync)

    with pytest.raises(OSError, match="directory marker fsync failed"):
        service.start_ocr(BOOK_A)

    marker = json.loads((paths.ocr / "worker.json").read_text(encoding="utf-8"))
    assert marker["state"] == expected_state
    assert len(launcher.calls) == expected_spawn_count
    assert not list(paths.ocr.glob(".worker-marker-*.tmp"))
    monkeypatch.setattr(service_module.os, "fsync", actual_fsync)
    service.start_ocr(BOOK_A)
    assert len(launcher.calls) == expected_spawn_count
    assert database.get_ocr_job(BOOK_A)["status"] == "queued"


@pytest.mark.parametrize("collision_kind", ["symlink", "hardlink"])
def test_marker_temp_collision_is_not_followed_or_deleted(
    tmp_path: Path, collision_kind: str
) -> None:
    paths = AppPaths.from_root(tmp_path / "project", tmp_path / "vault")
    paths.root.mkdir(parents=True)
    paths.originals.mkdir(parents=True)
    database = Database(paths.database, root=paths.root)
    database.initialize()
    launcher = RecordingPopen()
    token = "1" * 32
    service = OcrService(
        paths,
        database,
        popen_factory=launcher,
        token_factory=lambda: token,
    )
    _book(database, paths, BOOK_A)
    paths.ocr.mkdir(parents=True)
    target = paths.ocr / "target"
    target.write_text("keep", encoding="utf-8")
    collision = paths.ocr / f".worker-marker-{token}.tmp"
    if collision_kind == "symlink":
        collision.symlink_to(target)
    else:
        os.link(target, collision)

    with pytest.raises(ValueError, match="temporary marker"):
        service.start_ocr(BOOK_A)

    assert collision.exists() or collision.is_symlink()
    assert target.read_text(encoding="utf-8") == "keep"
    if collision_kind == "hardlink":
        assert collision.stat().st_ino == target.stat().st_ino
    assert launcher.calls == []


def test_launcher_rejects_log_symlink_and_recovers_stale_marker(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    paths.ocr_logs.mkdir(parents=True)
    target = paths.ocr / "target.log"
    target.write_text("target", encoding="utf-8")
    (paths.ocr_logs / "worker.log").symlink_to(target)
    with pytest.raises(ValueError, match="log.*symlink|symlink.*log"):
        service.start_ocr(BOOK_A)
    assert launcher.calls == []
    assert database.get_ocr_job(BOOK_A)["status"] == "queued"

    (paths.ocr_logs / "worker.log").unlink()
    marker = paths.ocr / "worker.json"
    marker.write_text(
        json.dumps(
            {
                "pid": 999999,
                "root_device": paths.root.stat().st_dev,
                "root_inode": paths.root.stat().st_ino,
                "launched_at": NOW.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    service.start_ocr(BOOK_A)
    assert len(launcher.calls) == 1


def test_live_lease_prevents_second_worker_launch(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    _book(database, paths, BOOK_B)
    service.start_ocr(BOOK_A)
    claimed = database.claim_next_ocr_job("worker", 60, now=NOW)
    assert claimed is not None
    marker = paths.ocr / "worker.json"
    marker.unlink()

    service.start_ocr(BOOK_B)

    assert len(launcher.calls) == 1
    assert database.get_ocr_job(BOOK_B)["status"] == "queued"


def test_expired_running_lease_is_requeued_before_worker_restart(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    database.queue_ocr_job(BOOK_A, 2, ("zh-Hans", "en-US"))
    claimed = database.claim_next_ocr_job(
        "dead-worker", 60, now=NOW - timedelta(seconds=120)
    )
    assert claimed is not None and claimed["status"] == "running"

    result = service.start_ocr(BOOK_A)

    assert result.status == "queued"
    assert database.get_ocr_job(BOOK_A)["worker_id"] is None
    assert len(launcher.calls) == 1


def test_expired_pause_requested_worker_requeues_cleanly_and_keeps_checkpoint(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A, pages=2)
    database.queue_ocr_job(BOOK_A, 2, ("zh-Hans", "en-US"))
    old_now = NOW - timedelta(seconds=120)
    claimed = database.claim_next_ocr_job("dead-worker", 60, now=old_now)
    assert claimed is not None
    database.save_ocr_page(
        BOOK_A,
        "dead-worker",
        1,
        "1",
        "checkpoint",
        "sha",
        0.8,
        100,
        now=old_now,
    )
    database.request_ocr_pause(BOOK_A, now=old_now)

    result = service.start_ocr(BOOK_A)

    job = database.get_ocr_job(BOOK_A)
    assert result.status == "queued"
    assert job["pause_requested"] == 0
    assert job["worker_id"] is None
    assert job["lease_expires_at"] is None
    assert [page["page_number"] for page in database.list_ocr_pages(BOOK_A)] == [1]
    assert len(launcher.calls) == 1


def test_worker_launch_uses_time_sampled_after_flock(monkeypatch, app) -> None:
    _, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    clock = {"now": NOW}
    service = OcrService(
        paths,
        database,
        popen_factory=launcher,
        now_factory=lambda: clock["now"],
        pid_probe=lambda pid: pid == launcher.pid,
    )
    real_flock = service_module.fcntl.flock

    def delayed_flock(descriptor: int, operation: int) -> None:
        real_flock(descriptor, operation)
        clock["now"] += timedelta(seconds=31)

    monkeypatch.setattr(service_module.fcntl, "flock", delayed_flock)

    service.start_ocr(BOOK_A)
    monkeypatch.setattr(service_module.fcntl, "flock", real_flock)
    service.start_ocr(BOOK_A)

    assert len(launcher.calls) == 1
    marker = json.loads((paths.ocr / "worker.json").read_text(encoding="utf-8"))
    assert marker["launched_at"].startswith("2026-07-14T04:05:37")


def test_fresh_post_lock_time_does_not_treat_expired_lease_as_live(
    monkeypatch, app
) -> None:
    _, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    _book(database, paths, BOOK_B)
    database.queue_ocr_job(BOOK_A, 2, ("zh-Hans", "en-US"))
    claimed = database.claim_next_ocr_job("old-worker", 10, now=NOW)
    assert claimed is not None
    database.queue_ocr_job(BOOK_B, 2, ("zh-Hans", "en-US"))
    clock = {"now": NOW}
    service = OcrService(
        paths,
        database,
        popen_factory=launcher,
        now_factory=lambda: clock["now"],
        pid_probe=lambda _: False,
    )
    real_flock = service_module.fcntl.flock

    def delayed_flock(descriptor: int, operation: int) -> None:
        real_flock(descriptor, operation)
        clock["now"] += timedelta(seconds=20)

    monkeypatch.setattr(service_module.fcntl, "flock", delayed_flock)

    result = service.start_ocr(BOOK_B)

    assert result.status == "queued"
    assert len(launcher.calls) == 1


def test_two_concurrent_start_calls_spawn_at_most_one_worker(app) -> None:
    service, database, launcher, paths = app
    _book(database, paths, BOOK_A)
    other = OcrService(
        paths,
        database,
        popen_factory=launcher,
        now_factory=lambda: NOW,
        pid_probe=lambda pid: pid == launcher.pid,
    )
    barrier = threading.Barrier(2)

    def start(candidate: OcrService) -> OcrJobSummary:
        barrier.wait()
        return candidate.start_ocr(BOOK_A)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(start, (service, other)))

    assert [result.status for result in results] == ["queued", "queued"]
    assert len(launcher.calls) == 1


def test_keyboard_interrupt_and_system_exit_are_not_swallowed(app) -> None:
    _, database, _, paths = app
    _book(database, paths, BOOK_A)
    for error in (KeyboardInterrupt(), SystemExit(2)):
        launcher = RecordingPopen(error=error)
        service = OcrService(paths, database, popen_factory=launcher)
        with pytest.raises(type(error)):
            service.start_ocr(BOOK_A)
        marker = paths.ocr / "worker.json"
        if marker.exists():
            marker.unlink()
