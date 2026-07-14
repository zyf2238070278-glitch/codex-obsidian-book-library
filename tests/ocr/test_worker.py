from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from book_agent.config import AppPaths
from book_agent.ocr.worker import OcrWorker
from book_agent.storage import Database


def _app(tmp_path: Path, pages: int = 3):
    root = tmp_path.resolve()
    paths = AppPaths.from_root(root)
    paths.originals.mkdir(parents=True)
    paths.database.parent.mkdir(parents=True)
    database = Database(paths.database, root=root)
    database.initialize()
    pdf = paths.originals / "synthetic.pdf"
    document = fitz.open()
    for _ in range(pages):
        document.new_page()
    document.save(pdf)
    document.close()
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    book_id = "a" * 24
    database.create_book(book_id, "Synthetic", None, "pdf", digest, str(pdf), status="needs_ocr")
    database.queue_ocr_job(book_id, pages, ("zh-Hans",))
    return paths, database, pdf, book_id


class _Engine:
    def __init__(self, results):
        self.results = results
        self.requested_pages: list[int] = []

    def recognize_page(self, pdf: Path, *, page_index: int):
        page = page_index + 1
        self.requested_pages.append(page)
        value = self.results[page]
        if isinstance(value, BaseException):
            raise value
        return value


class _Indexer:
    def __init__(self, database: Database, book_id: str, *, status="ready", passages=1):
        self.database, self.book_id = database, book_id
        self.status, self.passages = status, passages

    def index_parsed_book(self, **kwargs):
        if self.passages:
            self.database.update_book_status(self.book_id, self.status)
        return type("Result", (), {"status": self.status, "passage_count": self.passages})()


def test_worker_resumes_first_missing_page_and_cleans_after_searchable(tmp_path):
    paths, database, _pdf, book_id = _app(tmp_path)
    database.claim_next_ocr_job("worker-a", 60)
    database.save_ocr_page(book_id, "worker-a", 1, "1", "existing", "x" * 64, 0.9, 1)
    engine = _Engine({2: "second", 3: "third"})
    worker = OcrWorker(paths, database, engine, _Indexer(database, book_id), worker_id="worker-a")

    assert worker.run_once() is True
    assert engine.requested_pages == [2, 3]
    assert database.get_ocr_job(book_id)["status"] == "completed"
    assert database.list_ocr_pages(book_id) == []


def test_blank_pages_still_complete(tmp_path):
    paths, database, _pdf, book_id = _app(tmp_path, pages=2)
    engine = _Engine({1: "", 2: ""})
    worker = OcrWorker(paths, database, engine, _Indexer(database, book_id, passages=0), worker_id="worker-a")

    assert worker.run_once() is True
    assert database.get_ocr_job(book_id)["status"] == "failed"
    assert database.get_book(book_id)["status"] == "needs_ocr"


def test_page_fails_after_two_retries_and_worker_can_continue(tmp_path):
    paths, database, _pdf, book_id = _app(tmp_path, pages=1)
    engine = _Engine({1: RuntimeError("synthetic engine failure")})
    worker = OcrWorker(paths, database, engine, _Indexer(database, book_id), worker_id="worker-a")

    assert worker.run_once() is True
    assert engine.requested_pages == [1, 1, 1]
    assert database.get_ocr_job(book_id)["status"] == "failed"
    assert database.list_ocr_pages(book_id) == []


def test_invalid_ocr_payload_never_becomes_page_text(tmp_path):
    paths, database, _pdf, book_id = _app(tmp_path, pages=1)
    engine = _Engine({1: {"text": b"not text"}})
    worker = OcrWorker(paths, database, engine, _Indexer(database, book_id), worker_id="worker-a")

    assert worker.run_once() is True
    assert database.get_ocr_job(book_id)["status"] == "failed"
    assert database.list_ocr_pages(book_id) == []


def test_pdf_rewrite_during_engine_is_rejected_before_checkpoint(tmp_path):
    paths, database, pdf, book_id = _app(tmp_path, pages=2)

    class MutatingEngine(_Engine):
        def recognize_page(self, pdf_path: Path, *, page_index: int):
            result = super().recognize_page(pdf_path, page_index=page_index)
            if page_index == 0:
                pdf_path.write_bytes(pdf_path.read_bytes() + b"changed")
            return result

    engine = MutatingEngine({1: "should not save", 2: "second"})
    worker = OcrWorker(paths, database, engine, _Indexer(database, book_id), worker_id="worker-a")

    assert worker.run_once() is True
    assert database.get_ocr_job(book_id)["status"] == "failed"
    assert database.list_ocr_pages(book_id) == []
    assert database.get_book(book_id)["status"] == "needs_ocr"


def test_original_path_escape_is_rejected(tmp_path):
    paths, database, pdf, book_id = _app(tmp_path, pages=1)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(pdf.read_bytes())
    database.update_book_original_path(book_id, str(outside))
    engine = _Engine({1: "must not run"})
    worker = OcrWorker(paths, database, engine, _Indexer(database, book_id), worker_id="worker-a")

    assert worker.run_once() is True
    assert engine.requested_pages == []
    assert database.list_ocr_pages(book_id) == []
