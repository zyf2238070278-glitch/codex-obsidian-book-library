# Apple Vision OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resumable, background, fully local Apple Vision OCR for scanned PDFs, expose it through Codex MCP tools, validate it only with generated test books, and publish a macOS Apple Silicon `v0.2.0-beta.1` release without starting OCR on the user's current books.

**Architecture:** Python owns job persistence, queueing, PDF rendering, checkpointing, indexing, and MCP responses. A small arm64 Swift executable owns only one-page Apple Vision recognition and returns versioned JSON. OCR jobs remain separate from book search status, so incomplete OCR text is never searchable; after all pages are checkpointed, a shared indexing service atomically publishes the same Markdown, passages, and embeddings used by ordinary imports.

**Tech Stack:** Python 3.12, SQLite/FTS5, PyMuPDF, NumPy, FastMCP, Swift 6, Vision, ImageIO, pytest, existing deterministic ZIP release builder.

---

## File Structure

### New runtime files

- `book_agent/indexing.py` — shared `ParsedBook` to Markdown/passages/embedding/status pipeline.
- `book_agent/ocr/__init__.py` — public OCR package exports.
- `book_agent/ocr/models.py` — frozen OCR result and job value objects.
- `book_agent/ocr/vision.py` — bounded subprocess contract, page rendering, and reading-order normalization.
- `book_agent/ocr/service.py` — MCP-facing queue, status, pause, and worker-launch orchestration.
- `book_agent/ocr/worker.py` — single-worker queue loop and page checkpoint processing.
- `book_agent/ocr_worker.py` — minimal `python -m book_agent.ocr_worker` entrypoint.
- `native/book_vision_ocr/main.swift` — Apple Vision single-image recognizer.
- `scripts/build_vision_helper.py` — deterministic invocation and validation of the Swift build.

### New tests

- `tests/ocr/test_models.py`
- `tests/ocr/test_storage.py`
- `tests/ocr/test_vision.py`
- `tests/ocr/test_service.py`
- `tests/ocr/test_worker.py`
- `tests/ocr/test_native_vision.py`
- `tests/test_indexing.py`
- `tests/test_build_vision_helper.py`

### Existing files modified

- `book_agent/config.py`
- `book_agent/storage.py`
- `book_agent/importer.py`
- `book_agent/tools.py`
- `book_agent/mcp_server.py`
- `installer/install_macos.py`
- `scripts/build_macos_release.py`
- `install-from-github.command`
- `distribution/release.json`
- `AGENTS.md`
- `README.md`
- `docs/安装说明.md`
- `docs/使用说明.md`
- `docs/常见问题.md`
- `docs/隐私与数据存放.md`
- related existing tests for each modified file

---

### Task 1: OCR paths and immutable value objects

**Files:**
- Modify: `book_agent/config.py`
- Create: `book_agent/ocr/__init__.py`
- Create: `book_agent/ocr/models.py`
- Create: `tests/ocr/test_models.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing path and value-object tests**

```python
def test_app_paths_places_ocr_runtime_under_project_data(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)

    assert paths.ocr == tmp_path.resolve() / "data" / "ocr"
    assert paths.ocr_logs == paths.ocr / "logs"
    assert paths.vision_helper == tmp_path.resolve() / "bin" / "book-vision-ocr"


def test_vision_page_result_orders_top_to_bottom_then_left_to_right() -> None:
    result = VisionPageResult(
        schema_version=1,
        lines=(
            VisionLine("右", 0.9, BoundingBox(0.60, 0.70, 0.10, 0.05)),
            VisionLine("下一行", 0.8, BoundingBox(0.10, 0.50, 0.30, 0.05)),
            VisionLine("左", 0.95, BoundingBox(0.10, 0.70, 0.10, 0.05)),
        ),
    )

    assert result.ordered_text() == "左 右\n下一行"
    assert result.mean_confidence == pytest.approx(0.8833333333)
```

- [ ] **Step 2: Run the tests and verify the expected RED state**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_models.py tests/test_config.py`

Expected: collection or assertion failure because the OCR package and path fields do not exist.

- [ ] **Step 3: Add focused path fields and value objects**

Add `ocr`, `ocr_logs`, and `vision_helper` to `AppPaths`. Define frozen `BoundingBox`, `VisionLine`, `VisionPageResult`, and `OcrJobSummary` dataclasses. Validate finite confidence values in `[0, 1]`, nonblank text, positive page counts, and the exact schema version. Implement stable line grouping with a documented vertical tolerance rather than relying on Apple observation order.

```python
@dataclass(frozen=True)
class VisionPageResult:
    schema_version: int
    lines: tuple[VisionLine, ...]

    def ordered_text(self) -> str:
        grouped = _group_lines(self.lines, vertical_tolerance=0.0125)
        return "\n".join(
            " ".join(line.text for line in sorted(row, key=lambda item: item.box.x))
            for row in grouped
        ).strip()
```

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_models.py tests/test_config.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add book_agent/config.py book_agent/ocr tests/ocr/test_models.py tests/test_config.py
git commit -m "feat: add OCR paths and value objects"
```

### Task 2: Backward-compatible OCR job persistence

**Files:**
- Modify: `book_agent/storage.py`
- Create: `tests/ocr/test_storage.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Write failing migration, queue, lease, and checkpoint tests**

Cover all of these independently:

```python
def test_initialize_adds_ocr_tables_without_changing_existing_books(db: Database) -> None:
    _book(db, book_id="scan-book", content_hash="scan-hash")
    db.update_book_status("scan-book", "needs_ocr", error="需要 OCR")

    db.initialize()

    assert db.get_book("scan-book")["status"] == "needs_ocr"
    assert db.get_ocr_job("scan-book") is None


def test_claim_next_ocr_job_is_atomic_and_respects_live_lease(db: Database) -> None:
    _needs_ocr_book(db, "book-a")
    db.queue_ocr_job("book-a", total_pages=12, languages=("zh-Hans", "en-US"))

    claimed = db.claim_next_ocr_job("worker-a", lease_seconds=60)
    second = db.claim_next_ocr_job("worker-b", lease_seconds=60)

    assert claimed["book_id"] == "book-a"
    assert claimed["status"] == "running"
    assert second is None


def test_page_checkpoint_is_idempotent_and_updates_completed_count(db: Database) -> None:
    _queued_job(db, "book-a", total_pages=3)

    db.claim_next_ocr_job("worker-a", lease_seconds=60)
    db.save_ocr_page("book-a", "worker-a", 1, None, "第一页", "sha", 0.92, 840)
    db.save_ocr_page("book-a", "worker-a", 1, None, "第一页", "sha", 0.92, 840)

    assert db.get_ocr_job("book-a")["completed_pages"] == 1
    assert db.list_ocr_pages("book-a")[0]["page_number"] == 1
```

Also test FIFO order, stale lease recovery, pause request, failed-page resume, empty page checkpoints, completion cleanup, foreign keys, invalid state transitions, JSON-safe return values, and connection closure.

- [ ] **Step 2: Run the new storage tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_storage.py tests/test_storage.py`

Expected: failures because OCR schema and database methods do not exist.

- [ ] **Step 3: Extend `_SCHEMA` with strict OCR tables**

```sql
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
    mean_confidence REAL,
    duration_ms INTEGER NOT NULL CHECK(duration_ms >= 0),
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(book_id, page_number)
);

CREATE INDEX IF NOT EXISTS ocr_jobs_queue_idx
ON ocr_jobs(status, created_at, book_id);
```

- [ ] **Step 4: Implement validated database operations**

Add `queue_ocr_job`, `claim_next_ocr_job`, `renew_ocr_lease`, `save_ocr_page`, `list_ocr_pages`, `get_ocr_job`, `list_ocr_jobs`, `request_ocr_pause`, `pause_ocr_job`, `fail_ocr_job`, `complete_ocr_job`, and `delete_ocr_page_checkpoints`. Every owner mutation requires the current `worker_id` and a lease that is still live at an injectable `now`, atomically rejecting expired or replaced workers. Store `duration_ms` with every page checkpoint so `ocr_status` can compute a rolling estimate from the most recent completed pages. `claim_next_ocr_job` must use `BEGIN IMMEDIATE` and one connection so two workers cannot claim the same job. Persist every OCR timestamp in the same UTC ISO-8601 `Z` format; do not mix SQLite `CURRENT_TIMESTAMP` strings with aware timestamps.

- [ ] **Step 5: Run storage tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_storage.py tests/test_storage.py`

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add book_agent/storage.py tests/ocr/test_storage.py tests/test_storage.py
git commit -m "feat: persist resumable OCR jobs"
```

### Task 3: Extract the shared indexing service without changing imports

**Files:**
- Create: `book_agent/indexing.py`
- Create: `tests/test_indexing.py`
- Modify: `book_agent/importer.py`
- Modify: `tests/test_importer.py`

- [ ] **Step 1: Write failing characterization tests for a public indexing service**

```python
def test_indexer_publishes_markdown_passages_and_ready_status(app) -> None:
    paths, db, provider = app
    parsed = ParsedBook(
        title="扫描测试",
        author="作者",
        source_format="pdf",
        units=(SourceUnit("第一页正文足够长。", page_start=1, page_end=1),),
    )

    result = BookIndexer(paths, db, provider).index_parsed_book(
        book_id="a" * 24,
        parsed=parsed,
        original_path=paths.originals / "scan.pdf",
    )

    assert result.status == "ready"
    assert db.count_passages("a" * 24) == 1
    assert Path(result.parsed_path).read_text(encoding="utf-8").find("PDF 页 1") >= 0
```

Add characterization tests showing that ordinary TXT/PDF imports retain their current result statuses, messages, embedding validation, rollback behavior, and searchable output.

- [ ] **Step 2: Run tests and verify RED for the missing service**

Run: `.venv/bin/python -m pytest -q tests/test_indexing.py tests/test_importer.py`

Expected: `BookIndexer` import failure while existing importer tests still pass.

- [ ] **Step 3: Move only shared indexing behavior into `BookIndexer`**

Move chunking, atomic Markdown rendering, passage replacement, embedding validation/building, and final book status writing from `ImportService` into `BookIndexer`. Return a frozen `IndexResult` containing status, parsed path, passage count, error, and message. Keep source copying, content-hash locks, parser invocation, and `needs_ocr` classification in `ImportService`.

```python
indexer = BookIndexer(
    paths=self.paths,
    database=self.database,
    embedding_provider=self.embedding_provider,
    vault_root_identity=self.vault_root_identity,
)
result = indexer.index_parsed_book(
    book_id=book_id,
    parsed=parsed,
    original_path=original.path,
)
```

- [ ] **Step 4: Run indexing and importer tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/test_indexing.py tests/test_importer.py`

Expected: all selected tests pass with no importer behavior changes.

- [ ] **Step 5: Commit**

```bash
git add book_agent/indexing.py book_agent/importer.py tests/test_indexing.py tests/test_importer.py
git commit -m "refactor: share book indexing pipeline"
```

### Task 4: Build and validate the native Apple Vision helper

**Files:**
- Create: `native/book_vision_ocr/main.swift`
- Create: `scripts/build_vision_helper.py`
- Create: `tests/test_build_vision_helper.py`
- Create: `tests/ocr/test_native_vision.py`

- [ ] **Step 1: Write failing builder and contract tests**

The builder tests inject a fake command runner and assert the exact non-shell argv, target, output path, executable validation, Mach-O arm64 validation, and capabilities call. The real integration test is marked `@pytest.mark.macos_vision` and skips unless both macOS and a built helper are present.

```python
def test_builder_invokes_swiftc_for_arm64_macos_13(tmp_path: Path) -> None:
    calls = []
    output = tmp_path / "book-vision-ocr"

    build_vision_helper(
        source=PROJECT_ROOT / "native/book_vision_ocr/main.swift",
        output=output,
        run_command=_fake_swift_runner(calls, output),
    )

    assert calls[0][:5] == [
        "xcrun", "swiftc", "-O", "-target", "arm64-apple-macos13.0"
    ]
    assert "shell" not in calls[0]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_build_vision_helper.py tests/ocr/test_native_vision.py`

Expected: missing module/source failures.

- [ ] **Step 3: Implement the Swift JSON contract**

`main.swift` must support:

```text
book-vision-ocr --version
book-vision-ocr --capabilities
book-vision-ocr --image /absolute/page.png --languages zh-Hans,en-US
```

Use `CGImageSourceCreateWithURL`, `VNImageRequestHandler`, and `VNRecognizeTextRequest`. Set `.accurate`, `usesLanguageCorrection = true`, and requested languages only after verifying support. Encode this schema with `JSONEncoder`:

```json
{"schema_version":1,"lines":[{"text":"示例","confidence":0.98,"box":{"x":0.1,"y":0.7,"width":0.2,"height":0.05}}]}
```

Reject relative image paths, non-file URLs, missing files, unsupported languages, empty arguments, and more than 100,000 recognized characters. Send JSON only to stdout and human-readable errors only to stderr.

- [ ] **Step 4: Implement the Python builder**

Invoke `xcrun swiftc` without a shell, apply an ad-hoc signature with `codesign --force --sign -`, set mode `0755`, reject symlinks and non-regular output, verify arm64 with `lipo -archs`, verify the signature with `codesign --verify --strict`, then call `--capabilities` and require schema `1`, `zh-Hans`, and `en-US`.

- [ ] **Step 5: Run builder tests, build the real helper, and run the native smoke test**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_build_vision_helper.py
.venv/bin/python -m scripts.build_vision_helper --output bin/book-vision-ocr
.venv/bin/python -m pytest -q -m macos_vision tests/ocr/test_native_vision.py
```

Expected: builder tests pass, the helper reports arm64/schema 1/required languages, and the real smoke test passes.

- [ ] **Step 6: Commit source and tests, but not the local compiled binary**

Ensure `/bin/book-vision-ocr` is ignored for source Git while the release builder can consume it explicitly.

```bash
git add .gitignore native/book_vision_ocr/main.swift scripts/build_vision_helper.py tests/test_build_vision_helper.py tests/ocr/test_native_vision.py
git commit -m "feat: add Apple Vision OCR helper"
```

### Task 5: Render bounded PDF pages and call Vision safely

**Files:**
- Create: `book_agent/ocr/vision.py`
- Create: `tests/ocr/test_vision.py`

- [ ] **Step 1: Write failing renderer and client tests**

Test a generated PDF at normal and extreme page dimensions. Assert grayscale rendering, 300 DPI until the cap applies, maximum 20 million pixels, one temporary file, cleanup after success/failure, absolute helper argv, 120-second timeout, 1 MiB stdout cap, schema validation, line ordering, and no shell.

```python
def test_renderer_caps_pixels_and_deletes_png_after_recognition(tmp_path: Path) -> None:
    pdf = _write_large_page_pdf(tmp_path / "large.pdf")
    seen = []

    result = VisionOcrEngine(
        helper=tmp_path / "helper",
        run_helper=_fake_helper(seen, "正文"),
        temp_root=tmp_path / "ocr-temp",
    ).recognize_page(pdf, page_index=0)

    assert result.text == "正文"
    assert seen[0].pixel_count <= 20_000_000
    assert list((tmp_path / "ocr-temp").glob("*.png")) == []
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_vision.py`

Expected: missing `VisionOcrEngine` failure.

- [ ] **Step 3: Implement bounded rendering and helper parsing**

Open the managed PDF with PyMuPDF, render the requested physical page to `fitz.csGRAY`, scale from 300 DPI down only when long-edge or 20-million-pixel limits require it, and save a mode-`0600` random PNG under `AppPaths.ocr`. Invoke the helper with explicit argv and environment, reject oversized/non-UTF-8/malformed output, and always remove the PNG in `finally`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_vision.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add book_agent/ocr/vision.py tests/ocr/test_vision.py
git commit -m "feat: recognize bounded PDF pages with Vision"
```

### Task 6: OCR service and detached worker startup

**Files:**
- Create: `book_agent/ocr/service.py`
- Create: `tests/ocr/test_service.py`

- [ ] **Step 1: Write failing service tests**

Cover validation of a real `needs_ocr` PDF, unknown IDs, non-PDF books, ready books, queue-all ordering, idempotent start, resume from paused/failed state, status bounds, pause requests, rolling remaining-time estimates after five duration checkpoints, and detached launch.

```python
def test_start_ocr_persists_job_and_returns_without_running_pages(app) -> None:
    service, db, launcher = app
    _needs_ocr_pdf(db, "b" * 24, pages=138)

    result = service.start_ocr("b" * 24)

    assert result.status == "queued"
    assert result.total_pages == 138
    assert launcher.calls == [[sys.executable, "-m", "book_agent.ocr_worker"]]
    assert db.list_ocr_pages("b" * 24) == []
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_service.py`

Expected: missing service failure.

- [ ] **Step 3: Implement queue and worker launch orchestration**

`OcrService.start_ocr`, `start_pending_ocr`, `status`, and `pause` must validate IDs and book state, obtain page count without extracting text, persist the job, and ensure the worker process is alive. Launch with current `sys.executable`, `start_new_session=True`, closed stdin, append-only local log, project cwd, and only required book-library environment variables. Never interpolate paths into shell commands.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_service.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add book_agent/ocr/service.py tests/ocr/test_service.py
git commit -m "feat: queue background OCR jobs"
```

### Task 7: Resumable single-worker processing and final indexing

**Files:**
- Create: `book_agent/ocr/worker.py`
- Create: `book_agent/ocr_worker.py`
- Create: `tests/ocr/test_worker.py`

- [ ] **Step 1: Write failing worker tests with a fake Vision engine**

Cover FIFO processing, one worker claim, page-by-page checkpoints, blank page completion, two retries then failure, failed-page resume, pause at page boundary, lease renewal, process interruption recovery, original hash drift rejection, no OCR text in logs, no search visibility before finalization, and `ready`/`keyword_only` final states.

```python
def test_worker_resumes_at_first_missing_page_and_indexes_only_after_completion(app) -> None:
    worker, db, engine, retriever = app
    _queued_job(db, "c" * 24, total_pages=3)
    db.claim_next_ocr_job("worker-a", lease_seconds=60)
    db.save_ocr_page("c" * 24, "worker-a", 1, "1", "已有第一页", "sha", 0.95, 800)
    engine.results = {2: "第二页", 3: "第三页"}

    worker.run_once()

    assert engine.requested_pages == [2, 3]
    assert db.get_book("c" * 24)["status"] == "ready"
    assert [hit.page_start for hit in retriever.search("第二页", mode="quote")] == [2]
    assert db.list_ocr_pages("c" * 24) == []
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_worker.py`

Expected: missing worker failure.

- [ ] **Step 3: Implement `OcrWorker`**

For each claimed job: re-read and validate the managed original, verify content SHA-256, open the PDF, process missing physical pages in ascending order, checkpoint each result with monotonic `duration_ms`, renew the lease after every page, and honor pause requests only at page boundaries. Retry a page twice after the first failure, then persist `failed` with the physical page number.

When all pages are complete, construct `ParsedBook` from nonempty `ocr_pages`, preserving physical page number and page label, call `BookIndexer`, mark the job completed, and delete page checkpoints only after the final book state is searchable. If no passages are produced, leave the book `needs_ocr` and mark the OCR job failed with a readable message.

- [ ] **Step 4: Implement the minimal module entrypoint**

`book_agent/ocr_worker.py` must build normal paths/database/provider/indexer/engine, run until the queue is empty, and return nonzero only for worker-level failures. Book-level failures remain persisted and allow the worker to continue to the next queued book.

- [ ] **Step 5: Run worker tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/ocr/test_worker.py`

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add book_agent/ocr/worker.py book_agent/ocr_worker.py tests/ocr/test_worker.py
git commit -m "feat: process resumable OCR jobs"
```

### Task 8: Expose bounded OCR MCP tools and update book guidance

**Files:**
- Modify: `book_agent/tools.py`
- Modify: `book_agent/mcp_server.py`
- Modify: `book_agent/importer.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_mcp_server.py`
- Modify: `tests/test_importer.py`

- [ ] **Step 1: Write failing facade and MCP tests**

```python
def test_mcp_exports_exact_ocr_tools() -> None:
    assert set(mcp_server.TOOL_NAMES) >= {
        "start_ocr", "start_pending_ocr", "ocr_status", "pause_ocr"
    }


def test_needs_ocr_import_tells_user_to_confirm_before_starting(library, scan_pdf) -> None:
    result = library.import_book(str(scan_pdf))

    assert result["status"] == "needs_ocr"
    assert "开始 OCR" in result["message"]
    assert library.ocr_service.status()["count"] == 0
```

Also test JSON safety, unknown IDs, invalid types, status result limits, no OCR正文 in status, regular exception wrapping, and `KeyboardInterrupt` propagation.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_tools.py tests/test_mcp_server.py tests/test_importer.py`

Expected: missing OCR facade methods/tools and old import message failures.

- [ ] **Step 3: Wire `OcrService` into `LibraryTools` and FastMCP**

Add four guarded methods with explicit type validation and four `@mcp.tool()` functions. `ocr_status` must return metadata only. Update `build_tools` to construct one shared database, provider, indexer, importer, retriever, notes service, and OCR service.

- [ ] **Step 4: Update the `needs_ocr` message without auto-starting**

Return: `原书已保存，但该 PDF 没有可提取文字。请明确说“开始 OCR 这本书”后再进行本机识别。`

- [ ] **Step 5: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/test_tools.py tests/test_mcp_server.py tests/test_importer.py`

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add book_agent/tools.py book_agent/mcp_server.py book_agent/importer.py tests/test_tools.py tests/test_mcp_server.py tests/test_importer.py
git commit -m "feat: expose OCR tools to Codex"
```

### Task 9: Install and package the native helper

**Files:**
- Modify: `installer/install_macos.py`
- Modify: `tests/installer/test_install_macos.py`
- Modify: `scripts/build_macos_release.py`
- Modify: `tests/test_build_macos_release.py`
- Modify: `distribution/release.json`
- Modify: `install-from-github.command`
- Modify: `tests/installer/test_install_from_github_launcher.py`

- [ ] **Step 1: Write failing installer tests**

Test that installation rejects a missing, symlinked, non-regular, non-executable, wrong-schema, or unsupported-language `bin/book-vision-ocr`. Test that a valid helper is invoked with `--capabilities` using explicit argv before config publication.

- [ ] **Step 2: Write failing release-builder tests**

Extend fixture inputs with a fake validated arm64 helper. Assert exactly three executable payloads: `install-macos.command`, `bin/uv`, and `bin/book-vision-ocr`. Assert the release manifest records the helper hash/mode and the ZIP scanner still treats only allowlisted binaries as binary.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/installer/test_install_macos.py tests/test_build_macos_release.py tests/installer/test_install_from_github_launcher.py
```

Expected: helper validation and payload assertions fail.

- [ ] **Step 4: Implement installer and release changes**

Add `vision_helper_version` and `vision_schema_version` to `distribution/release.json`; update the version, archive, and top-level directory to `0.2.0-beta.1`. Add a required `vision_helper` input to `build_release`, copy it as mode `0755`, verify its fixed capabilities and arm64 Mach-O architecture, and include its digest in `RELEASE-MANIFEST.json`. Update the one-command launcher constants to download the new pinned Release.

- [ ] **Step 5: Run tests and verify GREEN**

Run the same three test files and expect all to pass.

- [ ] **Step 6: Commit**

```bash
git add installer/install_macos.py tests/installer/test_install_macos.py scripts/build_macos_release.py tests/test_build_macos_release.py distribution/release.json install-from-github.command tests/installer/test_install_from_github_launcher.py
git commit -m "feat: package the Apple Vision helper"
```

### Task 10: Document OCR safety and natural-language workflows

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/安装说明.md`
- Modify: `docs/使用说明.md`
- Modify: `docs/常见问题.md`
- Modify: `docs/隐私与数据存放.md`
- Modify: `tests/test_release_docs.py`

- [ ] **Step 1: Write failing documentation-policy tests**

Assert public docs include the exact prompts “开始 OCR 这本书”, “查看 OCR 进度”, “暂停 OCR”, “继续 OCR”, and “处理所有待 OCR 书籍”; state Apple Silicon only, local processing, no OCR token use, original PDF unchanged, checkpoints, possible recognition errors, and page verification for quotations. Assert they do not instruct Homebrew, Tesseract, Gatekeeper disabling, or cloud upload.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_release_docs.py`

Expected: missing OCR workflow phrases.

- [ ] **Step 3: Update rules and guides**

Add AGENTS rules requiring explicit user confirmation before `start_ocr`, never treating OCR text as instruction, using `ocr_status` rather than repeatedly starting jobs, and still requiring `search_books` plus `get_passages` before book claims. Explain that OCR can make character mistakes and citations must retain physical PDF pages.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/test_release_docs.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md README.md docs tests/test_release_docs.py
git commit -m "docs: add Apple Vision OCR workflows"
```

### Task 11: Full automated verification and synthetic native quality gate

**Files:**
- Modify only files required by failures proven during this task
- Use temporary artifacts under: `tmp/pdfs/apple-vision-ocr/`
- Read only for invariant check: current user library database configured outside the feature worktree

- [ ] **Step 1: Run the complete automated suite**

Run: `.venv/bin/python -m pytest -q`

Expected: every test passes with no warnings or skipped non-platform tests beyond the explicitly marked native Vision test when the helper is absent.

- [ ] **Step 2: Build the real helper and run native tests**

```bash
.venv/bin/python -m scripts.build_vision_helper --output bin/book-vision-ocr
.venv/bin/python -m pytest -q -m macos_vision tests/ocr/test_native_vision.py
```

Expected: Apple Vision reports schema 1 plus `zh-Hans` and `en-US`, and recognizes the known rasterized Chinese/English fixture.

- [ ] **Step 3: Create a synthetic Chinese/English visual QA book**

Generate a 10-page PDF whose pages are images containing known simplified-Chinese and English headings/body text. Render or inspect every page under `tmp/pdfs/apple-vision-ocr/`, then compare visible order with Apple Vision output. Do not copy the generated PDF, PNGs, or OCR output into Git.

Acceptance: all 10 outputs preserve physical page number; at least 9 contain readable正文 in the expected order; any error is documented with page number before full-book OCR.

- [ ] **Step 4: Exercise pause/resume in a temporary library**

Import the generated image-only PDF into a temporary project and temporary Vault. Start OCR, wait for at least three page checkpoints, call pause, verify the worker stops at a page boundary, restart the temporary services, resume, and confirm the completed page count never decreases or repeats.

- [ ] **Step 5: Finish the synthetic book and verify retrieval**

After completion in the temporary library, call `library_status`, `search_books`, and `get_passages`. Confirm status `ready` or `keyword_only`, search results point to physical PDF pages, returned evidence matches only the known generated text, and Obsidian links open the parsed Markdown.

- [ ] **Step 6: Prove the user's current books were not touched**

Before and after all native/integration tests, read the current library status without starting any OCR tool. Confirm every existing book retains its prior status and that no `ocr_jobs` or `ocr_pages` rows were created for those book IDs. Do not call `start_ocr` or `start_pending_ocr` against the current project.

- [ ] **Step 7: Re-run the complete suite after synthetic validation**

Run: `.venv/bin/python -m pytest -q`

Expected: every automated test still passes.

- [ ] **Step 8: Commit only code/test corrections, never runtime data**

Verify `git status --short` contains no PDF, PNG, SQLite, OCR checkpoint, model, log, or user path before committing any proven correction.

### Task 12: Build, independently install, and publish `v0.2.0-beta.1`

**Files:**
- Build output: `dist/codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip`
- Build output: `dist/SHA256SUMS`
- Public clean repository staging: `/private/tmp/codex-obsidian-book-library-public-v0.2`

- [ ] **Step 1: Build the all-in-one release**

Run the Vision helper builder, then call `scripts.build_macos_release` with the pinned model snapshot, pinned `uv`, and validated Vision helper. Expected: ZIP and checksum are published atomically and `unzip -tq` reports no errors.

- [ ] **Step 2: Install into a fresh path containing Chinese and spaces**

Use a clean temporary HOME/project location and a freshly copied ZIP that retains normal downloaded-file quarantine behavior. Run `install-macos.command`, verify Python dependencies, the ad-hoc signed Vision helper, Vision capabilities, generated `.codex/config.toml`, empty library status, and no writes outside the chosen project/vault.

- [ ] **Step 3: Run a tiny end-to-end OCR in the fresh install**

Import a generated two-page image-only Chinese/English PDF, explicitly start OCR, poll status to completion, search for known text, expand the passage, and verify physical pages 1 and 2.

- [ ] **Step 4: Build a one-commit public source snapshot**

Copy only the reviewed public allowlist. Exclude `.codex/config.toml`, user guides containing private paths, runtime data, PDFs, databases, OCR checkpoints, compiled local helper, `.venv`, model files, logs, build artifacts, and private planning documents. Run the full public-tree tests and secret/path scan before commit.

- [ ] **Step 5: Push source and create the prerelease**

Push `main`, tag `v0.2.0-beta.1`, and create a GitHub prerelease containing exactly:

- the all-in-one ZIP
- `SHA256SUMS`
- `INSTALL-MACOS-ZH.md`
- `USER-GUIDE-ZH.md`

- [ ] **Step 6: Verify the public user path anonymously**

Anonymous-clone the public repository, run `bash -n install-from-github.command`, confirm the README one-line command points at the public repository, anonymously download `SHA256SUMS`, and verify its digest matches the local ZIP and GitHub asset digest.

- [ ] **Step 7: Final handoff**

Report the repository URL, Release URL, one-line install command, SHA-256, supported macOS/architecture, pilot OCR result, test counts, and the remaining limitation that Windows OCR is not included.

---

## Plan Self-Review Checklist

- Every design requirement maps to a task: explicit start (Tasks 6/8), background queue and resume (Tasks 2/6/7), direct indexing without a new PDF (Tasks 3/7), existing books (Tasks 6/8/11), `zh-Hans` + `en-US` Vision (Tasks 4/5), local privacy (Tasks 5/7/10), packaging (Tasks 9/12), and real-book validation (Task 11).
- New signatures are consistent: `BookIndexer.index_parsed_book`, `OcrService.start_ocr/start_pending_ocr/status/pause`, `OcrWorker.run_once`, and database OCR methods retain the same names across tasks.
- The plan contains no optional implementation branches: Swift helper + Python orchestration is the only production route.
- Runtime OCR artifacts and real book pages never enter Git or the public ZIP.
- No task starts OCR on the user's current books; native and end-to-end validation use generated image-only PDFs in temporary libraries.
