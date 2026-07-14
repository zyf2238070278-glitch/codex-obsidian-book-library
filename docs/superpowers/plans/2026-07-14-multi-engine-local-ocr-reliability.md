# 完全本地多引擎 OCR 稳定性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将书库 OCR 升级为完全离线的 Apple Vision、RapidOCR、Tesseract 自动降级系统；单页无法恢复时跳过并生成可核验的 Obsidian 报告，而不停止整本书。

**Architecture:** Python 保留队列、断点、文件一致性和索引发布职责。新增统一页面 OCR 结果、共享受限渲染/预处理器、质量路由器及三种引擎适配器；worker 只接收路由器已验证的页面结论。SQLite 将页面检查点扩展为识别、空白和跳过三种结果，OCR 报告由受限 Vault 服务原子写入 `书库/40-OCR报告`，并永远不进入书籍证据索引。

**Tech Stack:** Python 3.12、PyMuPDF、Pillow、NumPy、Apple Vision Swift helper、RapidOCR 3.9.1、ONNX Runtime 1.27.0、Tesseract 5.5.2、SQLite/FTS5、pytest、现有 macOS arm64 发布构建器。

---

## 文件结构

| 路径 | 职责 |
|---|---|
| `book_agent/ocr/models.py` | 引擎无关的页面结果、处理策略、页面结论和任务摘要。 |
| `book_agent/ocr/rendering.py` | 受限灰度渲染、DPI 阶梯、图像变体和分块坐标。 |
| `book_agent/ocr/quality.py` | 空白页检测、文本质量评分和跨引擎结果选择。 |
| `book_agent/ocr/router.py` | 引擎按需路由、错误分类、退避与最终跳过决策。 |
| `book_agent/ocr/vision.py` | Apple Vision 适配器；保留安全的 helper 调用边界。 |
| `book_agent/ocr/rapid.py` | RapidOCR/ONNX 本机适配器和固定模型校验。 |
| `book_agent/ocr/tesseract.py` | Tesseract 私有子进程适配器、语言数据和输出校验。 |
| `book_agent/ocr/report.py` | OCR 统计和受限 Markdown 报告生成。 |
| `book_agent/ocr/worker.py` | 页面检查点、文件身份校验、索引发布和报告调用。 |
| `book_agent/storage.py` | SQLite 迁移、页面结论写入、警告统计和缺失页查询。 |
| `book_agent/config.py`、`book_agent/vault.py` | OCR 模型/二进制路径及 `40-OCR报告` 目录。 |
| `book_agent/ocr_worker.py` | 组装三引擎路由器，不联网。 |
| `book_agent/ocr/service.py`、`book_agent/tools.py`、`book_agent/mcp_server.py` | 有界 OCR 状态、缺失页重试工具和中文用户提示。 |
| `native/book_vision_ocr/main.swift` | 单条异常 Vision 框不终止整页。 |
| `installer/`、`scripts/build_macos_release.py`、`distribution/` | 固定依赖、模型/二进制清单、许可证与离线安装验证。 |
| `tests/ocr/`、`tests/installer/`、`tests/test_build_macos_release.py` | 单元、迁移、路由、报告、离线安装与发布回归测试。 |

### Task 1: 建立引擎无关页面模型和 SQLite 迁移

**Files:**

- Modify: `book_agent/ocr/models.py`
- Modify: `book_agent/storage.py`
- Modify: `book_agent/config.py`
- Modify: `book_agent/vault.py`
- Test: `tests/ocr/test_models.py`
- Test: `tests/ocr/test_storage.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写出失败的模型和迁移测试**

```python
def test_page_outcome_accepts_recognized_blank_and_skipped() -> None:
    assert OcrPageOutcome("recognized", "apple_vision", "standard").status == "recognized"
    assert OcrPageOutcome("blank", None, "blank").status == "blank"
    assert OcrPageOutcome("skipped", None, "all_local_failed", "Vision unavailable").status == "skipped"


def test_database_migrates_existing_ocr_pages_without_losing_text(db: Database) -> None:
    # Create schema-v1 row, call initialize(), then assert outcome defaults to recognized.
    row = db.list_ocr_pages(BOOK_ID)[0]
    assert row["outcome"] == "recognized"
    assert row["engine"] == "apple_vision"


def test_paths_expose_non_evidence_ocr_reports_directory(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    VaultManager(paths).ensure_layout()
    assert paths.ocr_reports == paths.library / "40-OCR报告"
    assert paths.ocr_reports.is_dir()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_models.py tests/ocr/test_storage.py tests/test_config.py`

Expected: import failure for `OcrPageOutcome` and missing `ocr_reports`/SQLite columns.

- [ ] **Step 3: 实现最小的值对象、路径和前向迁移**

```python
_PAGE_OUTCOMES = ("recognized", "blank", "skipped")

@dataclass(frozen=True)
class OcrPageOutcome:
    status: Literal["recognized", "blank", "skipped"]
    engine: str | None
    strategy: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _PAGE_OUTCOMES:
            raise ValueError("unsupported OCR page outcome")
        if self.status == "recognized" and not self.engine:
            raise ValueError("recognized pages require an engine")
        if self.status != "recognized" and self.engine is not None:
            raise ValueError("non-recognized pages must not name an engine")
```

Add `ocr_reports` to `AppPaths`, create it in `VaultManager.ensure_layout`, and add an idempotent `PRAGMA table_info(ocr_pages)` migration that adds `outcome TEXT NOT NULL DEFAULT 'recognized'`, `engine TEXT`, `strategy TEXT NOT NULL DEFAULT 'legacy'`, and `detail TEXT`. Do not alter existing searchable passages or AI notes.

- [ ] **Step 4: 完善数据库写入和统计 API**

Add `save_ocr_page_result` with explicit `book_id`, `worker_id`, `page_number`, `page_label`, `outcome`, `text`, `text_sha256`, `mean_confidence`, `duration_ms` and `now` parameters. Require text/hash only for `recognized`, store empty text/hash for `blank` and `skipped`, and atomically update `completed_pages`. Add `ocr_page_outcome_counts(book_id)` and `list_skipped_ocr_pages(book_id)` returning bounded metadata only.

- [ ] **Step 5: 运行迁移和模型测试**

Run: `.venv/bin/pytest -q tests/ocr/test_models.py tests/ocr/test_storage.py tests/test_config.py`

Expected: PASS; existing rows become `recognized` without text loss.

- [ ] **Step 6: 提交**

```bash
git add book_agent/ocr/models.py book_agent/storage.py book_agent/config.py book_agent/vault.py tests/ocr/test_models.py tests/ocr/test_storage.py tests/test_config.py
git commit -m "feat: persist OCR page outcomes"
```

### Task 2: 修复受限页面渲染并抽取共享渲染策略

**Files:**

- Create: `book_agent/ocr/rendering.py`
- Modify: `book_agent/ocr/vision.py`
- Test: `tests/ocr/test_rendering.py`
- Test: `tests/ocr/test_vision.py`

- [ ] **Step 1: 写出像素边界与 DPI 阶梯失败测试**

```python
def test_safe_scale_leaves_rounding_margin_at_pixel_cap() -> None:
    plan = plan_render(width_points=1983, height_points=2972)
    assert plan.dpi < 300
    assert plan.pixel_width * plan.pixel_height <= 19_600_000


def test_renderer_uses_next_dpi_variant_after_render_error(tmp_path: Path) -> None:
    variants = list(RenderPlanner().variants_for(_write_pdf(tmp_path / "huge.pdf")))
    assert [item.dpi for item in variants] == [300, 240, 180, 144]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_rendering.py tests/ocr/test_vision.py::test_caps_long_edge_and_twenty_million_pixels_proportionally`

Expected: missing `RenderPlanner` and the existing cap test reproduces the no-margin behavior.

- [ ] **Step 3: 实现只负责图像的共享渲染器**

```python
SAFE_PAGE_PIXELS = 19_600_000
DPI_LADDER = (300, 240, 180, 144)

@dataclass(frozen=True)
class RenderedPage:
    png_path: Path
    dpi: int
    strategy: str
    width: int
    height: int

class RenderPlanner:
    def render(self, pdf: Path, page_index: int, dpi: int) -> RenderedPage:
        return self._render_bounded(pdf, page_index, dpi, SAFE_PAGE_PIXELS)
```

Compute scale from both max-edge and `SAFE_PAGE_PIXELS`, then multiply by `0.99` before calling PyMuPDF. If integer dimensions still exceed a cap, reduce by a real 1% each loop, never by `1e-9`; four failures must raise a typed `RenderError`. Preserve grayscale, private temporary files, file-identity checks and cleanup.

- [ ] **Step 4: 让 `VisionOcrEngine` 消费 `RenderedPage`**

Keep the public `recognize_page(pdf, page_index)` compatibility method temporarily; delegate image preparation to `RenderPlanner` and retain current private helper snapshot/codesign behavior. Do not add RapidOCR or Tesseract imports in this task.

- [ ] **Step 5: 运行渲染回归测试**

Run: `.venv/bin/pytest -q tests/ocr/test_rendering.py tests/ocr/test_vision.py`

Expected: PASS; extreme page tests prove the 227-page class no longer stalls on rounded dimensions.

- [ ] **Step 6: 提交**

```bash
git add book_agent/ocr/rendering.py book_agent/ocr/vision.py tests/ocr/test_rendering.py tests/ocr/test_vision.py
git commit -m "fix: make OCR page rendering safely bounded"
```

### Task 3: 让 Apple Vision 单条异常框可恢复

**Files:**

- Modify: `native/book_vision_ocr/main.swift`
- Modify: `book_agent/ocr/vision.py`
- Modify: `tests/ocr/test_native_vision.py`
- Modify: `tests/ocr/test_vision.py`

- [ ] **Step 1: 写出 native 结果可部分恢复的失败测试**

```python
def test_parser_accepts_valid_lines_when_one_native_box_is_discarded(tmp_path: Path) -> None:
    result = _parse_helper_output(_payload([VALID_LINE], discarded_boxes=1))
    assert result.ordered_text() == "有效文字"
    assert result.discarded_observations == 1
```

For Swift, add a source-contract test asserting the recognition loop uses `if let box = normalizedBoxOrNil(observation.boundingBox)` and increments `discardedObservations`, instead of throwing from `normalizedBox` for every observation.

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_native_vision.py tests/ocr/test_vision.py::test_tolerates_only_tiny_bbox_rounding_error`

Expected: missing discarded-observation schema field or helper source still throws on one bad box.

- [ ] **Step 3: 变更 helper 输出为版本 2 的可恢复 schema**

```swift
private func normalizedBoxOrNil(_ rectangle: CGRect) -> BoxPayload? {
    // Clamp tiny finite overflow; return nil for zero-area, non-finite, or major overflow.
}

if let box = normalizedBoxOrNil(observation.boundingBox) {
    indexedLines.append(IndexedLine(index: index, payload: LinePayload(text: text, confidence: confidence, box: box)))
} else {
    discardedObservations += 1
}
```

Add `discarded_observations` to `OCRPayload`; reject invalid confidence as an observation-level discard, preserve text budgets only for accepted lines, and bump the helper/schema version constants together. In Python, parse the new field as a bounded nonnegative integer and expose it on the engine-neutral page result.

- [ ] **Step 4: 运行 native 和 Python 协议测试**

Run: `.venv/bin/pytest -q tests/ocr/test_native_vision.py tests/ocr/test_vision.py`

Expected: PASS in the unsandboxed macOS Vision environment; malformed one-line geometry no longer fails a page.

- [ ] **Step 5: 提交**

```bash
git add native/book_vision_ocr/main.swift book_agent/ocr/vision.py tests/ocr/test_native_vision.py tests/ocr/test_vision.py
git commit -m "fix: tolerate isolated Vision bounding box errors"
```

### Task 4: 实现质量评分和空白页判定

**Files:**

- Create: `book_agent/ocr/quality.py`
- Modify: `book_agent/ocr/models.py`
- Test: `tests/ocr/test_quality.py`

- [ ] **Step 1: 写出文本质量和空白页的失败测试**

```python
def test_nonblank_image_with_empty_ocr_requires_fallback() -> None:
    verdict = assess_page(text="", lines=(), image_ink_ratio=0.08)
    assert verdict.accepted is False
    assert verdict.reason == "unexpected_empty_text"


def test_blank_image_with_empty_ocr_is_a_blank_outcome() -> None:
    verdict = assess_page(text="", lines=(), image_ink_ratio=0.0001)
    assert verdict.outcome.status == "blank"


def test_control_character_heavy_text_is_rejected() -> None:
    assert assess_page("\x00\x01\ufffd\ufffd", (), 0.1).accepted is False
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_quality.py`

Expected: missing `assess_page`.

- [ ] **Step 3: 实现可解释的确定性评分**

```python
@dataclass(frozen=True)
class QualityVerdict:
    accepted: bool
    outcome: OcrPageOutcome | None
    score: float
    reason: str

def assess_page(result: OcrPageResult, image_ink_ratio: float) -> QualityVerdict:
    return _assess_text_and_geometry(result, image_ink_ratio)
```

Use only bounded, reproducible signals: engine confidence, non-control Unicode ratio, replacement-character ratio, longest repeated-character run, nonblank line count, normalized box coverage, and image ink ratio. Do not compare raw confidence values from different engines. Empty text becomes `blank` only below the explicit ink threshold; otherwise it is rejected for fallback.

- [ ] **Step 4: 运行质量测试**

Run: `.venv/bin/pytest -q tests/ocr/test_quality.py tests/ocr/test_models.py`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add book_agent/ocr/quality.py book_agent/ocr/models.py tests/ocr/test_quality.py tests/ocr/test_models.py
git commit -m "feat: assess local OCR page quality"
```

### Task 5: 接入 RapidOCR 离线适配器

**Files:**

- Create: `book_agent/ocr/rapid.py`
- Modify: `book_agent/config.py`
- Modify: `pyproject.toml`
- Test: `tests/ocr/test_rapid.py`

- [ ] **Step 1: 写出不依赖真实模型的失败测试**

```python
def test_rapid_engine_rejects_missing_pinned_model(tmp_path: Path) -> None:
    engine = RapidOcrEngine(model_root=tmp_path / "missing")
    with pytest.raises(RapidOcrError, match="RapidOCR model is missing"):
        engine.recognize_image(_image(tmp_path))


def test_rapid_engine_normalizes_polygons_to_boxes(fake_rapid, tmp_path: Path) -> None:
    result = RapidOcrEngine(tmp_path, factory=fake_rapid).recognize_image(_image(tmp_path))
    assert result.engine == "rapidocr"
    assert result.lines[0].box == BoundingBox(0.1, 0.2, 0.3, 0.1)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_rapid.py`

Expected: missing module/class.

- [ ] **Step 3: 实现惰性、离线、固定根目录的适配器**

```python
class RapidOcrEngine:
    def __init__(self, model_root: Path, *, factory: Callable[[Path], Any] | None = None) -> None:
        self._model_root, self._factory, self._runtime = model_root, factory, None
    def available(self) -> bool:
        return all((self._model_root / name).is_file() for name in REQUIRED_MODEL_FILES)
    def recognize_image(self, image: RenderedPage) -> OcrPageResult:
        return self._normalise(self._load_runtime()(str(image.png_path)), image)
```

Pin `rapidocr==3.9.1` and `onnxruntime==1.27.0` in the OCR extra. Pass only explicit local detector/recognizer/classifier model paths to RapidOCR; reject any configuration that would download models. Convert quadrilateral polygons to clipped normalized `BoundingBox` values, discard single invalid observations, and preserve line ordering. Add `ocr_models` to `AppPaths` beneath the project root.

- [ ] **Step 4: 运行适配器和完整单元测试**

Run: `.venv/bin/pytest -q tests/ocr/test_rapid.py tests/ocr/test_quality.py`

Expected: PASS without network access or model downloads.

- [ ] **Step 5: 提交**

```bash
git add book_agent/ocr/rapid.py book_agent/config.py pyproject.toml tests/ocr/test_rapid.py
git commit -m "feat: add offline RapidOCR fallback"
```

### Task 6: 接入 Tesseract 私有后备适配器

**Files:**

- Create: `book_agent/ocr/tesseract.py`
- Modify: `book_agent/config.py`
- Test: `tests/ocr/test_tesseract.py`

- [ ] **Step 1: 写出私有 argv、语言数据和 TSV 解析的失败测试**

```python
def test_tesseract_uses_only_packaged_binary_and_tessdata(tmp_path: Path) -> None:
    engine = TesseractEngine(binary=_binary(tmp_path), tessdata=_tessdata(tmp_path), runner=_runner)
    engine.recognize_image(_image(tmp_path))
    assert _runner.argv == [str(_binary(tmp_path)), str(_image(tmp_path).png_path), "stdout", "--tessdata-dir", str(_tessdata(tmp_path)), "-l", "chi_sim+chi_tra+eng", "tsv"]
    assert _runner.shell is False


def test_tesseract_discards_invalid_tsv_rows(tmp_path: Path) -> None:
    assert _engine(tmp_path, tsv=VALID_AND_INVALID_ROWS).recognize_image(_image(tmp_path)).ordered_text() == "有效行"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_tesseract.py`

Expected: missing module/class.

- [ ] **Step 3: 实现受限子进程引擎**

```python
class TesseractEngine:
    def recognize_image(self, image: RenderedPage) -> OcrPageResult:
        request = [str(self.binary), str(image.png_path), "stdout", "--tessdata-dir", str(self.tessdata), "-l", "chi_sim+chi_tra+eng", "tsv"]
        completed = subprocess.run(request, shell=False, check=False, text=True, capture_output=True, timeout=120, env=self.environment)
        return self._parse_tsv(completed.stdout)
```

Validate binary and language-data identity before execution, set a minimal explicit environment, enforce output/error byte limits and 120-second timeout, reject nonzero exits with bounded diagnostics, and parse only positive-confidence word rows into normalized boxes. No PATH lookup, Homebrew lookup or network fallback is permitted.

- [ ] **Step 4: 运行 Tesseract 安全边界测试**

Run: `.venv/bin/pytest -q tests/ocr/test_tesseract.py`

Expected: PASS, including timeout/nonzero/malformed TSV cases.

- [ ] **Step 5: 提交**

```bash
git add book_agent/ocr/tesseract.py book_agent/config.py tests/ocr/test_tesseract.py
git commit -m "feat: add packaged Tesseract fallback"
```

### Task 7: 实现页面路由、图像变体与错误分类

**Files:**

- Create: `book_agent/ocr/router.py`
- Modify: `book_agent/ocr/rendering.py`
- Test: `tests/ocr/test_router.py`

- [ ] **Step 1: 写出路由顺序和跳页的失败测试**

```python
def test_router_stops_after_accepted_vision_result() -> None:
    result = _router(vision=_engine("Vision text"), rapid=_unused(), tesseract=_unused()).recognize_page(PDF, 0)
    assert result.outcome.engine == "apple_vision"


def test_router_uses_rapid_after_low_quality_vision() -> None:
    result = _router(vision=_engine("\ufffd\ufffd"), rapid=_engine("Rapid text"), tesseract=_unused()).recognize_page(PDF, 0)
    assert result.outcome.engine == "rapidocr"


def test_router_returns_skipped_after_all_local_strategies_fail() -> None:
    result = _router(vision=_failing(), rapid=_failing(), tesseract=_failing()).recognize_page(PDF, 0)
    assert result.outcome.status == "skipped"
    assert result.text == ""
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_router.py`

Expected: missing `LocalOcrRouter`.

- [ ] **Step 3: 实现明确且有界的策略表**

```python
STRATEGIES = (
    Strategy("standard", 300, "apple_vision"),
    Strategy("enhanced", 240, "rapidocr"),
    Strategy("enhanced", 180, "tesseract"),
    Strategy("tiles", 180, "apple_vision"),
    Strategy("tiles", 144, "rapidocr"),
    Strategy("tiles", 144, "tesseract"),
)
```

Classify typed rendering/schema/model-missing errors as deterministic and advance immediately. Classify helper timeout, Vision service unavailable and transient subprocess failure as transient; retry the same strategy twice using injected `sleep(0.25)` then `sleep(0.75)`, before advancing. For tiles, use a 3% overlap and deduplicate lines whose normalized centers fall in the same overlap region. Return `OcrPageDecision` with bounded attempt metadata, never page text in diagnostics.

- [ ] **Step 4: 运行路由和渲染测试**

Run: `.venv/bin/pytest -q tests/ocr/test_router.py tests/ocr/test_rendering.py tests/ocr/test_quality.py`

Expected: PASS; deterministic pixel-limit errors do not receive identical retries.

- [ ] **Step 5: 提交**

```bash
git add book_agent/ocr/router.py book_agent/ocr/rendering.py tests/ocr/test_router.py
git commit -m "feat: route pages across local OCR engines"
```

### Task 8: 让 worker 保存跳过页、减少重复哈希并完成索引

**Files:**

- Modify: `book_agent/ocr/worker.py`
- Modify: `book_agent/ocr_worker.py`
- Modify: `tests/ocr/test_worker.py`

- [ ] **Step 1: 写出检查点和哈希频率的失败测试**

```python
def test_worker_continues_after_skipped_page_and_indexes_other_pages(app) -> None:
    worker = _worker(app, decisions=[recognized("第一页"), skipped("all_local_failed"), recognized("第三页")])
    assert worker.run_once() is True
    assert app.database.list_skipped_ocr_pages(BOOK_ID) == [{"page_number": 2, "detail": "all_local_failed"}]
    assert app.database.count_passages(BOOK_ID) > 0


def test_worker_hashes_original_only_at_start_and_before_publish(app, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(worker_module, "_sha256", lambda path: calls.append(path) or EXPECTED_HASH)
    _worker(app, decisions=[recognized("a"), recognized("b")]).run_once()
    assert len(calls) == 2
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_worker.py`

Expected: worker fails the job on a skipped page and hashes repeatedly.

- [ ] **Step 3: 让 worker 使用 `LocalOcrRouter` 和新页面结论**

Replace the `VisionOcrEngine` protocol with `PageOcrRouter.recognize_page(pdf, page_index) -> OcrPageDecision`. Save every decision through `save_ocr_page_result`; create `SourceUnit` only for `recognized` nonblank text, while `blank` and `skipped` still count as complete. If no recognized text exists, retain the existing failed/no-searchable-text behavior.

At job start, perform full hash and record `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`. At each page boundary verify this snapshot and page count only. Before indexing, perform the second full hash plus identity validation. Keep interruption, lease, pause and original-path constraints unchanged.

- [ ] **Step 4: 运行 worker、storage 和 service 回归测试**

Run: `.venv/bin/pytest -q tests/ocr/test_worker.py tests/ocr/test_storage.py tests/ocr/test_service.py`

Expected: PASS; completed job can contain skipped pages, but all-empty books still fail.

- [ ] **Step 5: 提交**

```bash
git add book_agent/ocr/worker.py book_agent/ocr_worker.py tests/ocr/test_worker.py
git commit -m "feat: continue OCR jobs past unrecoverable pages"
```

### Task 9: 写入 OCR 报告并暴露简洁状态/缺失页重试

**Files:**

- Create: `book_agent/ocr/report.py`
- Modify: `book_agent/ocr/worker.py`
- Modify: `book_agent/ocr/service.py`
- Modify: `book_agent/ocr/models.py`
- Modify: `book_agent/tools.py`
- Modify: `book_agent/mcp_server.py`
- Test: `tests/ocr/test_report.py`
- Test: `tests/ocr/test_service.py`
- Test: `tests/test_ocr_mcp_tools.py`

- [ ] **Step 1: 写出报告和有界 MCP 元数据的失败测试**

```python
def test_report_lists_only_page_metadata_not_ocr_text(tmp_path: Path) -> None:
    path = OcrReportWriter(paths).write(_summary(skipped_pages=[{"page_number": 7, "detail": "all_local_failed"}]))
    markdown = path.read_text(encoding="utf-8")
    assert "PDF 页 7" in markdown
    assert "OCR 原文" not in markdown


def test_status_exposes_warning_count_and_report_link_without_page_text(app) -> None:
    payload = build_tools(app.root).ocr_status(BOOK_ID)
    assert payload["skipped_pages"] == 1
    assert payload["report_wiki_link"].startswith("[[书库/40-OCR报告/")
    assert "text" not in json.dumps(payload, ensure_ascii=False)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_report.py tests/ocr/test_service.py tests/test_ocr_mcp_tools.py`

Expected: missing report writer/status fields.

- [ ] **Step 3: 实现原子、非证据报告写入**

```python
class OcrReportWriter:
    def write(self, book: Mapping[str, object], job: Mapping[str, object], pages: Sequence[Mapping[str, object]]) -> OcrReport:
        relative = self._relative_path(book)
        markdown = render_report_markdown(book, job, pages)
        return self._publish_markdown(relative, markdown, mode=0o600)
```

Use the existing vault confinement/atomic-publication patterns. Sanitise titles using the same filename rules as notes; include only aggregate counts, engine counts, physical page numbers, bounded failure categories and fixed verification notice. Do not call `replace_passages`, do not write into `20-解析文本` or `30-AI读书笔记`.

Add `retry_skipped_ocr(book_id)` to service/tools/MCP: delete only `skipped` checkpoints for the selected job, requeue it, preserve `recognized`/`blank` pages, and require the same explicit user invocation as any OCR start.

- [ ] **Step 4: 运行报告、MCP 和 vault 测试**

Run: `.venv/bin/pytest -q tests/ocr/test_report.py tests/ocr/test_service.py tests/test_ocr_mcp_tools.py tests/test_vault.py`

Expected: PASS; report is outside evidence paths and no OCR text leaks through MCP metadata.

- [ ] **Step 5: 提交**

```bash
git add book_agent/ocr/report.py book_agent/ocr/worker.py book_agent/ocr/service.py book_agent/ocr/models.py book_agent/tools.py book_agent/mcp_server.py tests/ocr/test_report.py tests/ocr/test_service.py tests/test_ocr_mcp_tools.py
git commit -m "feat: report and retry OCR warning pages"
```

### Task 10: 更新离线安装、发布清单和许可证

**Files:**

- Create: `distribution/ocr-model-manifest.json`
- Create: `distribution/THIRD_PARTY_NOTICES.md`
- Modify: `installer/install_macos.py`
- Modify: `scripts/build_macos_release.py`
- Modify: `distribution/release.json`
- Modify: `pyproject.toml`
- Modify: `tests/installer/test_install_macos.py`
- Modify: `tests/test_build_macos_release.py`

- [ ] **Step 1: 写出发布载荷失败测试**

```python
def test_release_requires_pinned_rapid_models_tesseract_and_notices(tmp_path: Path) -> None:
    with pytest.raises(ReleaseBuildError, match="OCR model manifest"):
        build_release(
            project_root=tmp_path,
            model_snapshot=tmp_path / "semantic-model",
            vision_helper=tmp_path / "book-vision-ocr",
            ocr_payload_root=tmp_path / "ocr-payload",
            ocr_model_manifest=tmp_path / "missing.json",
        )


def test_installer_rejects_missing_tesseract_language_data(tmp_path: Path) -> None:
    with pytest.raises(InstallError, match="chi_sim.traineddata"):
        install_project(
            archive_root=tmp_path / "release",
            destination=tmp_path / "installed",
            run_command=_fake_runner,
        )
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/installer/test_install_macos.py tests/test_build_macos_release.py`

Expected: installer/release builder do not yet require fallback payloads.

- [ ] **Step 3: 添加固定载荷验证**

Create a JSON manifest with exact relative path, byte size and SHA-256 for every RapidOCR ONNX model, `tesseract` executable, required dynamic libraries and `chi_sim`, `chi_tra`, `eng`, `osd` traineddata. Extend release copying to reject unsafe paths/symlinks/non-arm64 binaries/unlisted dependencies; include Apache-2.0 notices for RapidOCR, ONNX Runtime, Tesseract and tessdata. Extend installer self-check to run each engine against one generated local PNG with networking disabled.

- [ ] **Step 4: 运行发布构建测试**

Run: `.venv/bin/pytest -q tests/installer/test_install_macos.py tests/test_build_macos_release.py tests/test_build_vision_helper.py`

Expected: PASS; ZIP builder rejects missing, swapped or unlisted OCR payloads.

- [ ] **Step 5: 提交**

```bash
git add distribution/ocr-model-manifest.json distribution/THIRD_PARTY_NOTICES.md installer/install_macos.py scripts/build_macos_release.py distribution/release.json pyproject.toml tests/installer/test_install_macos.py tests/test_build_macos_release.py
git commit -m "feat: package offline OCR fallback engines"
```

### Task 11: 更新用户文档、全套验证和真实书籍验收门槛

**Files:**

- Modify: `README.md`
- Modify: `docs/安装说明.md`
- Modify: `docs/使用说明.md`
- Modify: `docs/常见问题.md`
- Test: `tests/ocr/test_native_vision.py`

- [ ] **Step 1: 写出文档契约测试**

```python
def test_docs_explain_local_only_warning_page_behavior() -> None:
    text = Path("docs/使用说明.md").read_text(encoding="utf-8")
    assert "完全本地" in text
    assert "缺失页" in text
    assert "重试 OCR 缺失页" in text
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest -q tests/ocr/test_native_vision.py::test_docs_explain_local_only_warning_page_behavior`

Expected: test absent or documentation lacks the user contract.

- [ ] **Step 3: 写入用户可见流程**

Document only the user actions: import, explicit OCR start, status, pause, completion warning and retry skipped pages. State that OCR is fully local, original PDFs are unchanged, skipped physical pages are reported, and reports are not book evidence. Do not expose internal DPI or engine commands as required user steps.

- [ ] **Step 4: 运行全部自动化验证**

Run: `.venv/bin/pytest -q`

Expected: PASS. Run the one real Apple Vision smoke test outside the Codex filesystem sandbox because the sandbox cannot access the system Vision service:

Run: `.venv/bin/pytest -q tests/ocr/test_native_vision.py::test_native_helper_recognizes_synthetic_image_with_normalized_native_json`

Expected: PASS in the macOS host environment.

- [ ] **Step 5: 构建并检查发布 ZIP**

Run: `.venv/bin/python -m scripts.build_macos_release --help`

Expected: command exposes the new explicit OCR model and Tesseract payload arguments. Then run the documented release build command with pinned local payload paths, inspect `dist/*.zip` size, run the release scanner and install into a fresh temporary directory with network disabled.

- [ ] **Step 6: 提交**

```bash
git add README.md docs/安装说明.md docs/使用说明.md docs/常见问题.md tests/ocr/test_native_vision.py
git commit -m "docs: explain resilient local OCR workflow"
```

### Task 12: 用户授权后的两本真实书籍验收

**Files:**

- No code changes required unless automated verification exposes a reproducible defect.

- [ ] **Step 1: 确认用户明确授权启动真实 OCR**

Only proceed after the user explicitly says “开始 OCR 这本书” for the selected book or “处理所有待 OCR 书籍”. Do not infer this permission from installation or implementation approval.

- [ ] **Step 2: 恢复《虚拟现实（VR）影像拍摄与制作》**

Run through the existing `start_ocr` tool for book ID `1a2a1d0c5d84a3b7a89d8f2d`; verify it resumes at physical PDF page 4, records page outcomes, does not modify the original hash, and finishes with an index/report.

- [ ] **Step 3: 重新处理《世界摄影史》**

Run through the existing `start_ocr` tool for book ID `fd5075847d16a46c75317350`; verify the first-page invalid Vision box either recovers through the native fix or routes to a fallback, and that the job continues if any subsequent page is skipped.

- [ ] **Step 4: 验收报告与检索**

Use `ocr_status` to verify only metadata is returned. Then use `search_books` followed by `get_passages` for one narrow query per newly indexed book; verify physical PDF page locations and no report content appears as evidence.

- [ ] **Step 5: 记录真实验收结果，不在本任务中擅自扩大代码范围**

在任务记录中写明每本书的最终状态、报告路径、缺失物理页码和原书哈希验证结果。若验收发现可稳定复现的新缺陷，停止此验收任务，先建立最小失败测试，再以独立修复任务处理。

## 自检

- Spec coverage: Tasks 1–3 cover page schema, safe rendering and Apple Vision bad boxes; Tasks 4–7 cover quality, three engines, retries, enhancement and tiling; Tasks 8–9 cover performance, checkpoints, reports and retry; Task 10 covers all-in-one offline packaging; Task 11 covers documents and verification; Task 12 is gated by explicit OCR authorization.
- Placeholder scan: no implementation task relies on undefined future work; external payload locations are made explicit by the release manifest in Task 10.
- Type consistency: `OcrPageOutcome` is created in Task 1, used by quality/router in Tasks 4/7, persisted by worker in Task 8 and reported in Task 9. `RenderedPage` is created in Task 2 and is the only image input for all engine adapters.
