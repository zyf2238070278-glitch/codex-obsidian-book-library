# Obsidian Book Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable per-book catalog cards, user-preserved categories, and an Obsidian Bases dashboard for the active library.

**Architecture:** A focused `CatalogService` reads canonical book metadata from SQLite, creates one small Markdown card per book, preserves the two user-owned category properties on later syncs, and atomically writes one `.base` dashboard. Import and OCR completion call best-effort single-book sync; a bounded MCP tool performs explicit full backfill.

**Tech Stack:** Python 3.11+, SQLite metadata, Markdown/YAML frontmatter, Obsidian Bases, pytest.

---

### Task 1: Catalog paths and deterministic classification

**Files:**
- Modify: `book_agent/config.py`
- Create: `book_agent/catalog.py`
- Create: `tests/test_catalog.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for `catalog_cards`, `catalog_base`, known titles, and unknown-title fallback**

```python
def test_classifier_uses_curated_taxonomy_and_safe_fallback() -> None:
    assert classify_book("世界摄影史", None, "") == "摄影艺术与史论"
    assert classify_book("完全未知主题", None, "没有匹配词") == "待分类"
```

- [ ] **Step 2: Run `pytest tests/test_config.py tests/test_catalog.py -q` and confirm failures identify missing paths/service**

- [ ] **Step 3: Add `catalog_cards` and `catalog_base` to `AppPaths`, then implement ordered, deterministic keyword rules in `book_agent/catalog.py`**

```python
def classify_book(title: str, author: str | None, preview: str) -> str:
    haystack = "\n".join((title, author or "", preview[:4000])).casefold()
    for category, terms in CATEGORY_RULES:
        if any(term.casefold() in haystack for term in terms):
            return category
    return "待分类"
```

- [ ] **Step 4: Run the two test files and confirm they pass**

- [ ] **Step 5: Commit only Task 1 files with `feat: add deterministic book catalog classification`**

### Task 2: Safe card creation and user-category preservation

**Files:**
- Modify: `book_agent/catalog.py`
- Modify: `book_agent/vault.py`
- Modify: `tests/test_catalog.py`
- Modify: `tests/test_vault.py`

- [ ] **Step 1: Add failing tests that create a card, edit `primary_category` and `custom_categories`, sync again, and assert both edits remain**

```python
first = service.sync_book(book)
text = first.read_text(encoding="utf-8").replace(
    "primary_category: 摄影艺术与史论",
    "primary_category: 我的摄影研究",
).replace("custom_categories: []", "custom_categories:\n  - 必读")
first.write_text(text, encoding="utf-8")
service.sync_book({**book, "status": "keyword_only"})
updated = first.read_text(encoding="utf-8")
assert "primary_category: 我的摄影研究" in updated
assert "  - 必读" in updated
assert "library_status: keyword_only" in updated
```

- [ ] **Step 2: Run the preservation test and confirm it fails before implementation**

- [ ] **Step 3: Implement confined filenames, YAML scalar/list parsing for the two user fields, Obsidian-relative links, atomic writes, and `CatalogSyncError` for malformed cards**

- [ ] **Step 4: Add `50-书目卡片` to `VaultManager.ensure_layout`, run catalog/vault tests, and confirm pass**

- [ ] **Step 5: Commit with `feat: create durable Obsidian book cards`**

### Task 3: Bases dashboard and idempotent full sync

**Files:**
- Modify: `book_agent/catalog.py`
- Modify: `tests/test_catalog.py`

- [ ] **Step 1: Add failing tests for four exact Base views and duplicate-free repeated sync**

```python
result_one = service.sync_all()
result_two = service.sync_all()
assert result_one.created == 2
assert result_two.created == 0
base = paths.catalog_base.read_text(encoding="utf-8")
for name in ("按主分类", "全部书籍", "待 OCR", "OCR 有警告"):
    assert f"name: {name}" in base
assert len(list(paths.catalog_cards.glob("*.md"))) == 2
```

- [ ] **Step 2: Run the new test and confirm it fails**

- [ ] **Step 3: Implement `sync_all()` over `Database.list_books()`, stable card lookup by `book_id`, and atomically render `书库总览.base` using only `50-书目卡片`**

- [ ] **Step 4: Run `pytest tests/test_catalog.py -q` and confirm pass**

- [ ] **Step 5: Commit with `feat: add Obsidian book catalog dashboard`**

### Task 4: Import/OCR integration and explicit MCP sync

**Files:**
- Modify: `book_agent/importer.py`
- Modify: `book_agent/ocr/worker.py`
- Modify: `book_agent/tools.py`
- Modify: `book_agent/mcp_server.py`
- Modify: `installer/install_macos.py`
- Modify: `tests/test_importer.py`
- Modify: `tests/ocr/test_worker.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_ocr_mcp_tools.py`
- Modify: `tests/installer/test_install_macos.py`
- Modify: `tests/test_project_policy.py`

- [ ] **Step 1: Add failing tests for post-import sync, post-OCR sync, and `sync_book_catalog` returning counts without book text**

- [ ] **Step 2: Run the targeted tests and confirm they fail because integration points/tool are missing**

- [ ] **Step 3: Inject `CatalogService` into import/worker composition, call single-book sync after durable state changes, and expose guarded `sync_book_catalog()` through MCP**

```python
@mcp.tool()
def sync_book_catalog() -> dict[str, Any]:
    """Synchronize Obsidian catalog metadata without reading or returning book text."""
    return library_tools.sync_book_catalog()
```

- [ ] **Step 4: Add the tool to installer/project-policy allowlists, run all targeted tests, and confirm pass**

- [ ] **Step 5: Commit with `feat: synchronize book catalog from library lifecycle`**

### Task 5: Backfill and verify the active Obsidian library

**Files:**
- Create at runtime: `<OBSIDIAN_VAULT>/书库/50-书目卡片/*.md`
- Create at runtime: `<OBSIDIAN_VAULT>/书库/书库总览.base`

- [ ] **Step 1: Snapshot original-file names and SHA-256 values without modifying them**
- [ ] **Step 2: Run the catalog sync against the configured active database/vault**
- [ ] **Step 3: Assert 13 cards exist and category counts are `3,3,3,1,1,1,1`**
- [ ] **Step 4: Validate every existing source/parsed/report link target and confirm original hashes are unchanged**
- [ ] **Step 5: Open `书库总览.base` in Obsidian and visually confirm grouped rows are readable**
