# macOS Light OCR Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Light OCR 0.3.0 as the third macOS local OCR fallback after Apple Vision and RapidOCR, with safe page outcomes and offline installation.

**Architecture:** A persistent Node JSONL sidecar owns the Light OCR model while a Python adapter exposes the existing image-engine protocol. The router renders once, tries RapidOCR then Light OCR, and classifies final empty results as blank, image-only, or skipped. The active macOS install uses the current Node 24 runtime and pinned npm packages; the release builder stages the same runtime tree for offline use.

**Tech Stack:** Python 3.11+, Node.js 24, `@arcships/light-ocr@0.3.0`, PP-OCRv6 Small, PyMuPDF, pytest, Node built-in test runner.

---

### Task 1: Node JSONL sidecar and Python adapter

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Create: `scripts/light_ocr_worker.mjs`
- Create: `book_agent/ocr/light.py`
- Create: `tests/ocr/test_light.py`
- Create: `tests/node/light_ocr_worker.test.mjs`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing Python tests using a fake process for successful lines, invalid JSON, timeout, request-ID mismatch, and premature exit**

```python
engine = LightOcrEngine(node=Path("/fake/node"), worker=Path("worker.mjs"), process_factory=factory)
result = engine.recognize_image(image)
assert result.engine == "light_ocr"
assert result.ordered_text() == "测试文字"
```

- [ ] **Step 2: Run `pytest tests/ocr/test_light.py -q` and confirm module-not-found failure**

- [ ] **Step 3: Implement a lazy persistent subprocess with request IDs, strict bounded JSON schema, normalized top-left-to-bottom-left boxes, stderr isolation, timeout, `close()`, and context-manager cleanup**

- [ ] **Step 4: Implement the Node worker with one `ocr` instance, PNG/JPEG path validation, one JSON response per request, and no stdout logging**

- [ ] **Step 5: Pin npm packages with `npm install --package-lock-only`, run Python and Node protocol tests, and commit with `feat: add Light OCR sidecar adapter`**

### Task 2: Third-engine routing and image-only outcomes

**Files:**
- Modify: `book_agent/ocr/models.py`
- Modify: `book_agent/ocr/quality.py`
- Modify: `book_agent/ocr/router.py`
- Modify: `book_agent/storage.py`
- Modify: `book_agent/ocr/report.py`
- Modify: `tests/ocr/test_models.py`
- Modify: `tests/ocr/test_quality.py`
- Modify: `tests/ocr/test_router.py`
- Modify: `tests/ocr/test_storage.py`
- Modify: `tests/ocr/test_report.py`

- [ ] **Step 1: Add failing tests asserting exact order `apple_vision`, `rapidocr`, `light_ocr`, plus `image_only` terminal classification for nonblank visual pages with no text**

```python
decision = router.recognize_page(pdf, page_index=0)
assert calls == ["vision", "rapid", "light"]
assert decision.outcome.status == "image_only"
assert decision.attempts == (
    "standard:apple_vision", "enhanced:rapidocr", "enhanced:light_ocr"
)
```

- [ ] **Step 2: Run the five targeted test files and confirm failures**

- [ ] **Step 3: Extend page outcome validation/counts, accept optional `light` image engine, reuse one rendered PNG, and classify terminal non-text pages without treating them as skipped**

- [ ] **Step 4: Update reports to show recognized/blank/image-only/skipped totals while listing only skipped pages as failures**

- [ ] **Step 5: Run targeted tests and commit with `feat: add third OCR fallback and image-only outcomes`**

### Task 3: Worker composition and lifecycle

**Files:**
- Modify: `book_agent/config.py`
- Modify: `book_agent/ocr_worker.py`
- Modify: `book_agent/ocr/worker.py`
- Modify: `tests/test_config.py`
- Modify: `tests/ocr/test_worker.py`

- [ ] **Step 1: Add failing tests for Light OCR runtime paths and guaranteed sidecar close after queue completion or fatal error**
- [ ] **Step 2: Run targeted tests and confirm failures**
- [ ] **Step 3: Add `light_ocr_node`, `light_ocr_worker`, and `light_ocr_modules` paths; construct `LightOcrEngine`; close it in `finally` after `run_until_empty()`**
- [ ] **Step 4: Run targeted tests and commit with `feat: wire Light OCR into macOS worker`**

### Task 4: Active installation and offline release inputs

**Files:**
- Modify: `installer/install_macos.py`
- Modify: `scripts/build_macos_release.py`
- Create: `distribution/light-ocr-manifest.json`
- Modify: `distribution/release.json`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `tests/installer/test_install_macos.py`
- Modify: `tests/test_build_macos_release.py`

- [ ] **Step 1: Add failing tests that reject missing/wrong-architecture Node, missing npm artifacts, manifest hash mismatch, and runtime model downloads**
- [ ] **Step 2: Run installer/release targeted tests and confirm failures**
- [ ] **Step 3: Make installer prefer `runtime/node/bin/node`, otherwise validate system Node 22/24 arm64; install pinned production packages for the active project and validate an offline synthetic-image smoke test**
- [ ] **Step 4: Extend the release builder to stage the validated Node runtime, production `node_modules`, worker script, manifests and licenses without local paths or caches**
- [ ] **Step 5: Run targeted tests and commit with `build: package Light OCR for macOS`**

### Task 5: Install, regression test, and live smoke test

**Files:**
- Runtime-only: `node_modules/`
- Runtime-only: `data/ocr/`

- [ ] **Step 1: Run `npm install --omit=dev` and verify the installed tree is pinned by `package-lock.json`**
- [ ] **Step 2: Run all Python tests plus Node protocol tests; fix only failures caused by this feature and rerun to green**
- [ ] **Step 3: Run a synthetic Chinese/English PNG through `LightOcrEngine` and assert nonempty text with valid normalized boxes**
- [ ] **Step 4: Run one authorized fallback-only page smoke test without starting OCR for any existing book**
- [ ] **Step 5: Verify `git diff --check`, confirm original book hashes remain unchanged, and commit any final test/documentation adjustments**
