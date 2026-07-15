# Active Library In-Place Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Codex project that actually serves the user's existing book database to the two-engine OCR implementation, then publish a friend-ready Apple Silicon Mac ZIP from the same verified source.

**Architecture:** Fast-forward the active local `main` worktree to the already-tested `codex/multi-engine-ocr` history while preserving user-owned dirty files and runtime data. Rebuild the native Vision helper, run the installer against the existing project and Vault so RapidOCR dependencies/models are provisioned, then build and verify the deterministic all-in-one release without starting any OCR jobs.

**Tech Stack:** Python 3.12, uv, PyMuPDF, Apple Vision/Swift, RapidOCR 3.9.1, ONNX Runtime 1.27.0, SQLite, Obsidian Markdown, Git, GitHub CLI.

---

## File and Runtime Map

- Active project: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo`
- Source branch worktree: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/.worktrees/multi-engine-ocr`
- Active Codex config: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/.codex/config.toml`
- Existing book database and runtime data: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/data`
- Existing Obsidian Vault: `/Users/zhaoyunfei/Documents/Obsidian_workspace`
- Native helper source: `native/book_vision_ocr/main.swift`, `native/book_vision_ocr/TextBudget.swift`
- Native helper output: `bin/book-vision-ocr`
- RapidOCR runtime models: `data/ocr-models/rapidocr`
- Installer: `installer/install_macos.py`
- Release builder: `scripts/build_macos_release.py`
- Release output: `dist/codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip`

### Task 1: Capture the Active Runtime Baseline

**Files:**
- Inspect: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/.codex/config.toml`
- Preserve: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/AGENTS.md`
- Preserve: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/scripts/vision_ocr_pdf.swift`
- Inspect: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/data/library.sqlite3`

- [ ] **Step 1: Record the dirty worktree and current commit**

Run:

```bash
git -C /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo status --short
git -C /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo rev-parse HEAD
```

Expected: only the known user-owned `AGENTS.md` modification and untracked `scripts/vision_ocr_pdf.swift`; current branch is `main` at or after `d8b5b2f`.

- [ ] **Step 2: Confirm the configured data and Vault paths**

Run:

```bash
/usr/bin/sed -n '1,120p' /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/.codex/config.toml
```

Expected: `BOOK_LIBRARY_ROOT` is `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo` and `BOOK_LIBRARY_OBSIDIAN_VAULT` is `/Users/zhaoyunfei/Documents/Obsidian_workspace`.

- [ ] **Step 3: Record database and book-file counts without reading book contents**

Run:

```bash
find /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/data -type f -maxdepth 3 -print
find /Users/zhaoyunfei/Documents/Obsidian_workspace/书库/10-原始书籍 -type f -maxdepth 1 -print
```

Expected: paths can be listed without deletion, migration, or OCR execution.

### Task 2: Upgrade the Active Source In Place

**Files:**
- Merge existing changes from: `book_agent/ocr/`, `book_agent/ocr_worker.py`, `book_agent/storage.py`, `book_agent/vault.py`, `installer/install_macos.py`, `pyproject.toml`, `uv.lock`
- Preserve unchanged user files: `AGENTS.md`, `scripts/vision_ocr_pdf.swift`

- [ ] **Step 1: Verify the merge is a fast-forward**

Run:

```bash
git -C /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo merge-base main codex/multi-engine-ocr
git -C /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo merge-base --is-ancestor main codex/multi-engine-ocr
```

Expected: merge base is the current `main` commit or an ancestor, and the second command exits 0.

- [ ] **Step 2: Fast-forward the active main worktree**

Run:

```bash
git -C /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo merge --ff-only codex/multi-engine-ocr
```

Expected: fast-forward succeeds; no conflict; no reset, checkout, stash, or deletion occurs.

- [ ] **Step 3: Verify user-owned changes survived**

Run:

```bash
git -C /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo status --short
test -f /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/scripts/vision_ocr_pdf.swift
```

Expected: `AGENTS.md` remains modified and `scripts/vision_ocr_pdf.swift` remains untracked.

### Task 3: Verify the OCR Code Before Runtime Installation

**Files:**
- Test: `tests/ocr/test_rendering.py`
- Test: `tests/ocr/test_native_vision.py`
- Test: `tests/ocr/test_router.py`
- Test: `tests/ocr/test_rapid.py`
- Test: `tests/ocr/test_worker.py`
- Test: `tests/installer/test_install_macos.py`

- [ ] **Step 1: Run the renderer and routing regression tests**

Run from the active project:

```bash
.venv/bin/python -m pytest tests/ocr/test_rendering.py tests/ocr/test_router.py tests/ocr/test_rapid.py -q
```

Expected: all selected tests pass, including safe pixel rounding and Apple Vision-to-RapidOCR fallback.

- [ ] **Step 2: Run worker and installer regression tests**

Run:

```bash
.venv/bin/python -m pytest tests/ocr/test_worker.py tests/installer/test_install_macos.py -q
```

Expected: all selected tests pass, including skipped-page continuation, report generation, and RapidOCR model installation.

### Task 4: Provision the Active Two-Engine Runtime

**Files:**
- Generate: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/bin/book-vision-ocr`
- Generate: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/.venv`
- Generate: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/data/ocr-models/rapidocr/*.onnx`
- Rewrite with the same paths: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/.codex/config.toml`

- [ ] **Step 1: Build and validate the schema 2 Apple Vision helper**

Run:

```bash
.venv/bin/python -m scripts.build_vision_helper --output bin/book-vision-ocr
bin/book-vision-ocr --capabilities
/usr/bin/lipo -archs bin/book-vision-ocr
/usr/bin/codesign --verify --strict bin/book-vision-ocr
```

Expected: capabilities JSON reports `schema_version: 2`, includes `zh-Hans` and `en-US`, architecture is `arm64`, and signature verification exits 0.

- [ ] **Step 2: Run the installer against the existing project and Vault**

Run:

```bash
python3 installer/install_macos.py --project-root /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo --vault /Users/zhaoyunfei/Documents/Obsidian_workspace
```

Expected: `uv sync --frozen --extra semantic --extra ocr --python 3.12` succeeds; config paths remain unchanged; no OCR tool is invoked.

- [ ] **Step 3: Verify both engines and all RapidOCR models**

Run:

```bash
.venv/bin/python -c 'from book_agent.ocr.vision import VisionOcrEngine; from book_agent.ocr.rapid import RapidOcrEngine; from book_agent.ocr.router import LocalOcrRouter; print("two-engine OCR imports OK")'
find data/ocr-models/rapidocr -type f -name '*.onnx' -size +0 -print
```

Expected: imports print `two-engine OCR imports OK`; exactly these non-empty models exist: `PP-OCRv6_det_small.onnx`, `PP-OCRv6_rec_small.onnx`, and `ch_ppocr_mobile_v2.0_cls_mobile.onnx`.

- [ ] **Step 4: Confirm no OCR job was restarted**

Run:

```bash
sqlite3 data/library.sqlite3 "SELECT status, COUNT(*) FROM ocr_jobs GROUP BY status ORDER BY status;"
```

Expected: no row changed to `running` as a consequence of the upgrade. This is a read-only query.

### Task 5: Run Full Verification

**Files:**
- Test: `tests/`

- [ ] **Step 1: Run the complete test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all deterministic tests pass. If the macOS Vision synthetic-image test alone returns `Foundation._GenericObjCError`, record it and rerun that exact test once; do not hide the result.

- [ ] **Step 2: Run whitespace and active-config checks**

Run:

```bash
git diff --check
/usr/bin/sed -n '1,120p' .codex/config.toml
```

Expected: no whitespace errors; active config still uses the original project root and Obsidian Vault.

### Task 6: Build and Verify the Friend-Ready ZIP

**Files:**
- Read: `distribution/release.json`
- Read: `distribution/model-manifest.json`
- Read: `docs/word/安装说明.docx`
- Read: `docs/word/使用说明.docx`
- Generate: `dist/codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip`
- Generate: `dist/SHA256SUMS`

- [ ] **Step 1: Build the deterministic all-in-one release**

Run:

```bash
.venv/bin/python -m scripts.build_macos_release --project-root /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo --model-snapshot /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/data/models/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3 --uv-binary /Users/zhaoyunfei/.local/bin/uv --vision-helper /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/bin/book-vision-ocr --output-dir /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/dist
```

Expected: builder exits 0 and prints the ZIP and `SHA256SUMS` paths.

- [ ] **Step 2: Verify archive size, checksum, and required files**

Run:

```bash
ls -lh dist/codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip dist/SHA256SUMS
shasum -a 256 dist/codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip
unzip -l dist/codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip
```

Expected: ZIP is roughly 292 MB; checksum equals the value in `SHA256SUMS`; archive contains `install-macos.command`, `bin/uv`, `bin/book-vision-ocr`, semantic model files, and both Word guides.

### Task 7: Publish and Verify GitHub Delivery

**Files:**
- Publish: `dist/codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip`
- Publish: `dist/SHA256SUMS`

- [ ] **Step 1: Push the verified source branch**

Run:

```bash
git -C /Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/.worktrees/multi-engine-ocr push origin codex/multi-engine-ocr
```

Expected: GitHub branch advances to the implementation-plan commit and later implementation commits, without changing the unrelated legacy `main` history.

- [ ] **Step 2: Update the existing prerelease metadata**

Run:

```bash
/Users/zhaoyunfei/.local/bin/gh release edit v0.2.0-beta.1 --target codex/multi-engine-ocr --title "v0.2.0-beta.1｜双引擎本地 OCR" --notes "Apple Silicon Mac 测试版。Apple Vision + RapidOCR 本地回退、逐页跳过与 OCR 报告；压缩包内含 Word 安装说明和使用说明。"
```

Expected: command prints the public Release URL.

- [ ] **Step 3: Replace the ZIP and checksum assets**

Run:

```bash
/Users/zhaoyunfei/.local/bin/gh release upload v0.2.0-beta.1 dist/codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip dist/SHA256SUMS --clobber
```

Expected: both uploads complete. If the large ZIP connection resets, retry only the missing ZIP through the official uploads endpoint with the CLI keychain token kept in a local shell variable; never print the token.

- [ ] **Step 4: Verify online digests and target branch**

Run:

```bash
/Users/zhaoyunfei/.local/bin/gh release view v0.2.0-beta.1 --json url,targetCommitish,assets
```

Expected: target is `codex/multi-engine-ocr`; online ZIP digest equals local ZIP SHA-256; online `SHA256SUMS` digest equals the local checksum-file SHA-256.

### Task 8: Handoff Without Starting OCR

**Files:**
- Link: `/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/dist/codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip`
- Link: `https://github.com/zyf2238070278-glitch/codex-obsidian-book-library/releases/tag/v0.2.0-beta.1`

- [ ] **Step 1: Report the active local version and required restart**

Report the active commit, the two engine names, helper schema, RapidOCR model count, test totals, ZIP checksum, and Release URL. Tell the user to completely quit and reopen Codex so the MCP server reloads the upgraded code.

- [ ] **Step 2: Leave failed books untouched**

Do not call `start_ocr`, `start_pending_ocr`, or any equivalent command. Ask the user to explicitly say `重新开始这两本书的 OCR` after restarting Codex if they want the two failed jobs retried.
