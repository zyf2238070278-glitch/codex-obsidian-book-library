# Cross-Platform Codex Obsidian Book Library Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean public GitHub repository and a model-inclusive ZIP that a macOS Apple Silicon or Windows x64 Codex user can install without manually editing Python, MCP, or Obsidian configuration.

**Architecture:** Preserve the existing RAG domain layer while replacing POSIX-only filesystem operations with a small cross-platform safety layer used by imports, rendering, notes, locks, and SQLite setup. Add a shared Python installer behind thin macOS and Windows launchers, generate machine-local Codex configuration, verify a pinned flattened E5 model, and build a privacy-scanned universal Release ZIP from an allowlisted clean snapshot.

**Tech Stack:** Python 3.12, pytest, SQLite FTS5, FastMCP, `filelock`, `uv` 0.11.26, sentence-transformers, GitHub Actions, GitHub Releases.

---

## File map

New focused units:

- `book_agent/safe_fs.py` — cross-platform path validation, link/reparse detection, identities, locking, atomic publication, and identity-safe cleanup.
- `installer/__init__.py` — installer package marker.
- `installer/codex_config.py` — machine-local `.codex/config.toml` rendering and atomic write.
- `installer/obsidian.py` — macOS/Windows Obsidian Vault discovery and selection data.
- `installer/model_bundle.py` — model manifest validation, safe extraction/copy, and offline inference smoke test.
- `installer/setup.py` — preflight, dependency-independent install orchestration, logging, rollback, and final health check.
- `installer/bootstrap-macos.sh` and `installer/bootstrap-windows.ps1` — obtain or use bundled `uv`, run locked dependency setup, then invoke `installer.setup`.
- `install-macos.command` and `install-windows.bat` — double-click entry points.
- `distribution/release.json` — version, model ID/revision, expected asset names, and pinned uv version.
- `scripts/build_release.py` — clean public export, model flattening, uv asset inclusion, ZIP creation, manifests, and checksums.
- `scripts/scan_release.py` — allowlist and privacy scan for source trees and archives.
- `.github/workflows/ci.yml` — macOS/Windows test matrix.
- `.github/workflows/release-smoke.yml` — manual full-model clean-install smoke tests for a candidate Release.
- `README.md`, `docs/安装说明.md`, `docs/使用说明.md`, `docs/常见问题.md`, `docs/隐私与数据存放.md` — end-user documentation.
- `LICENSE` and `THIRD_PARTY_NOTICES.md` — project and bundled-model licensing.

Existing units to modify:

- `book_agent/importer.py` — remove direct `fcntl`/directory-FD locking and use `safe_fs`.
- `book_agent/vault.py` — retain `VaultManager` domain API while delegating portable operations to `safe_fs`.
- `book_agent/rendering.py` — use shared atomic replacement.
- `book_agent/notes.py` — use shared collision-safe publication.
- `book_agent/storage.py` — use shared managed-directory and safe-leaf validation.
- `book_agent/tools.py` — validate external Vault roots with shared link/reparse rules.
- `book_agent/embeddings.py` — accept and report a flattened pinned model directory.
- `pyproject.toml` and `uv.lock` — add direct cross-platform locking dependency and distribution metadata.
- `.gitignore` — ignore generated local config, install state, and release output.
- `.codex/config.toml` — remove from Git tracking while retaining the local ignored copy.
- `.codex/config.toml.template` — public non-personal template.
- `AGENTS.md` — keep evidence rules, remove wording tied to one person's Vault.
- existing tests — preserve current behavioral coverage while replacing assertions tied to POSIX implementation details or personal paths.

## Fixed distribution values

Use these values consistently:

```json
{
  "release_version": "0.1.0-beta.1",
  "release_tag": "v0.1.0-beta.1",
  "project_name": "codex-obsidian-book-library",
  "model_id": "intfloat/multilingual-e5-small",
  "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
  "model_asset": "codex-book-library-model-v0.1.0-beta.1.zip",
  "all_in_one_asset": "codex-obsidian-book-library-v0.1.0-beta.1-all-in-one.zip",
  "uv_version": "0.11.26",
  "python_version": "3.12"
}
```

### Task 1: Make project metadata and configuration publish-safe

**Files:**
- Modify: `tests/test_project_policy.py`
- Modify: `tests/test_user_guide.py`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Create: `.python-version`
- Create: `.codex/config.toml.template`
- Modify tracking only: `.codex/config.toml`

- [ ] **Step 1: Write failing portability-policy tests**

Replace the path-bound config test with tests that require a template and forbid personal material:

```python
def test_public_tree_has_no_personal_runtime_config() -> None:
    template = (PROJECT_ROOT / ".codex" / "config.toml.template").read_text(
        encoding="utf-8"
    )
    assert ("/" + "Users" + "/") not in template
    assert ("\\\\" + "Users" + "\\\\") not in template


def test_config_template_declares_only_book_library_tools() -> None:
    template = (PROJECT_ROOT / ".codex" / "config.toml.template").read_text(
        encoding="utf-8"
    )
    for marker in ("{{PYTHON}}", "{{PROJECT_ROOT}}", "{{OBSIDIAN_ENV}}"):
        assert marker in template
    for tool in TOOL_ALLOWLIST:
        assert f'  "{tool}",' in template
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_project_policy.py tests/test_user_guide.py -q`  
Expected: FAIL because the template is missing and the guides still require the current user's absolute path.

- [ ] **Step 3: Add portable metadata and template**

Add `.python-version`:

```text
3.12
```

Add direct dependency and version metadata in `pyproject.toml`:

```toml
[project]
name = "codex-obsidian-book-library"
version = "0.1.0b1"
requires-python = ">=3.12,<3.13"

dependencies = [
    "beautifulsoup4>=4.12",
    "ebooklib>=0.18",
    "filelock>=3.16,<4",
    "mcp>=1.9,<2",
    "numpy>=1.26",
    "pydantic>=2,<3",
    "pymupdf>=1.24",
]
```

Create `.codex/config.toml.template` with `{{PYTHON}}`, `{{PROJECT_ROOT}}`, and `{{OBSIDIAN_ENV}}` markers. Add `.codex/config.toml`, `data/`, `dist/`, `.installer-cache/`, and `install.log` to `.gitignore`, then run `git rm --cached .codex/config.toml` without deleting the working copy.

- [ ] **Step 4: Lock dependencies and rerun tests**

Run: `uv lock && uv sync --extra dev --extra semantic`  
Expected: exit 0 with a Python 3.12 environment and `filelock` in `uv.lock`.

Run: `uv run pytest tests/test_project_policy.py tests/test_user_guide.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore .python-version .codex/config.toml.template pyproject.toml uv.lock tests/test_project_policy.py tests/test_user_guide.py
git commit -m "build: make project configuration portable"
```

### Task 2: Add cross-platform filesystem safety primitives

**Files:**
- Create: `book_agent/safe_fs.py`
- Create: `tests/test_safe_fs.py`

- [ ] **Step 1: Write failing primitive tests**

Cover root confinement, symlinks, Windows reparse flags, identity checks, locking, replacement, and rollback:

```python
def test_link_like_recognizes_windows_reparse_attribute() -> None:
    info = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
    assert is_link_like(info) is True


def test_managed_directory_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="beneath|inside|受管"):
        ensure_managed_directory(tmp_path / "root", tmp_path / "outside", create=True)


def test_atomic_replace_rolls_back_after_root_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    destination = root / "note.md"
    root.mkdir()
    destination.write_text("old", encoding="utf-8")
    identity = path_identity(root.lstat())
    real_validate = validate_identity
    calls = 0

    def fail_second(path: Path, expected: FileIdentity, label: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("root identity changed")
        real_validate(path, expected, label)

    monkeypatch.setattr(safe_fs, "validate_identity", fail_second)
    with pytest.raises(ValueError, match="identity changed"):
        atomic_write_bytes(destination, b"new", root=root, overwrite=True)
    assert destination.read_text(encoding="utf-8") == "old"
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_safe_fs.py -q`  
Expected: FAIL with `ModuleNotFoundError: book_agent.safe_fs`.

- [ ] **Step 3: Implement the primitive API**

Implement these public types and functions in `book_agent/safe_fs.py`; the bodies below define the required validation and locking behavior, while `atomic_write_bytes` additionally uses a same-directory backup/temp pair and the rollback sequence described immediately after the block:

```python
@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int


def path_identity(info: os.stat_result) -> FileIdentity:
    return FileIdentity(int(info.st_dev), int(info.st_ino))


def is_link_like(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0))
    return stat.S_ISLNK(info.st_mode) or bool(attributes & 0x400)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def validate_identity(path: Path, expected: FileIdentity, label: str) -> None:
    info = _absolute(path).lstat()
    if is_link_like(info) or path_identity(info) != expected:
        raise ValueError(f"{label} identity changed")


def validate_existing_directory(path: Path, label: str) -> FileIdentity:
    info = _absolute(path).lstat()
    if is_link_like(info):
        raise ValueError(f"{label} must not be a symlink or reparse point")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a directory")
    return path_identity(info)


def ensure_managed_directory(
    root: Path, target: Path, *, create: bool,
    expected_root: FileIdentity | None = None,
) -> FileIdentity:
    root = _absolute(root)
    target = _absolute(target)
    relative = target.relative_to(root)
    root_identity = validate_existing_directory(root, "managed root")
    if expected_root is not None and root_identity != expected_root:
        raise ValueError("managed root identity changed")
    current = root
    for component in relative.parts:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise ValueError(f"managed directory does not exist: {current}") from None
            current.mkdir()
            info = current.lstat()
        if is_link_like(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"unsafe managed directory: {current}")
    validate_identity(root, root_identity, "managed root")
    return path_identity(target.lstat())


def inspect_regular_file(
    path: Path, *, allow_missing: bool,
    require_single_link: bool = False,
) -> os.stat_result | None:
    try:
        info = _absolute(path).lstat()
    except FileNotFoundError:
        if allow_missing:
            return None
        raise ValueError(f"required file does not exist: {path}") from None
    if is_link_like(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"path must be a regular non-link file: {path}")
    if require_single_link and int(info.st_nlink) != 1:
        raise ValueError(f"file must not have hard-link aliases: {path}")
    return info


@contextmanager
def exclusive_lock(lock_path: Path) -> Iterator[None]:
    lock = FileLock(str(_absolute(lock_path)))
    with lock:
        yield


def atomic_write_bytes(
    destination: Path, payload: bytes, *, root: Path,
    overwrite: bool, expected_root: FileIdentity | None = None,
) -> Path:
    destination = _absolute(destination)
    root = _absolute(root)
    root_identity = validate_existing_directory(root, "managed root")
    if expected_root is not None and root_identity != expected_root:
        raise ValueError("managed root identity changed")
    parent_identity = ensure_managed_directory(
        root, destination.parent, create=True, expected_root=root_identity
    )
    with exclusive_lock(destination.parent / f".{destination.name}.write.lock"):
        previous = inspect_regular_file(destination, allow_missing=True)
        if previous is not None and not overwrite:
            raise FileExistsError(destination)
        return _atomic_write_locked(
            destination, payload, root=root, root_identity=root_identity,
            parent_identity=parent_identity, previous=previous,
        )


def remove_if_identity(path: Path, identity: FileIdentity) -> bool:
    path = _absolute(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if is_link_like(info) or path_identity(info) != identity:
        return False
    path.unlink()
    return True
```

Implement `_atomic_write_locked` with `tempfile.mkstemp` for both the new payload and, when overwriting, a byte-for-byte backup in the destination directory. Flush and `fsync` both; retry `os.replace` only for `PermissionError`; revalidate the root and parent identities after publication; if validation fails, restore the backup with `os.replace`, or remove the just-published file only when its recorded identity still matches. Always remove leftover temp/backup files in `finally`.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_safe_fs.py -q`  
Expected: PASS.

Run: `uv run pytest tests/test_config.py tests/test_storage.py -q`  
Expected: existing tests still PASS before migration.

- [ ] **Step 5: Commit**

```bash
git add book_agent/safe_fs.py tests/test_safe_fs.py
git commit -m "feat: add cross-platform filesystem primitives"
```

### Task 3: Migrate import locking away from `fcntl`

**Files:**
- Modify: `book_agent/importer.py`
- Modify: `tests/test_importer.py`
- Create: `tests/test_windows_importability.py`

- [ ] **Step 1: Write failing Windows-importability and lock tests**

```python
def test_runtime_modules_do_not_import_fcntl() -> None:
    for relative in ("book_agent/importer.py", "book_agent/vault.py"):
        tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "fcntl" not in imported


def test_book_lock_is_exclusive_across_instances(tmp_path: Path) -> None:
    first = _service(tmp_path)
    second = _service(tmp_path)
    entered: list[str] = []
    def take_second_lock() -> None:
        with second._book_lock("a" * 24):
            entered.append("second")
    with first._book_lock("a" * 24):
        worker = Thread(target=take_second_lock)
        worker.start()
        time.sleep(0.1)
        assert entered == []
    worker.join(timeout=2)
    assert entered == ["second"]
```

Implement the concurrency test with a helper that always exits the context in `finally`, so no lock remains after assertion failures.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_windows_importability.py tests/test_importer.py -q`  
Expected: FAIL because `importer.py` imports `fcntl` and uses directory-FD-only locking.

- [ ] **Step 3: Replace `_book_lock` implementation**

Retain book ID validation, then use shared confinement and locking:

```python
@contextmanager
def _book_lock(self, book_id: str) -> Iterator[None]:
    if len(book_id) != 24 or any(c not in _LOWER_HEX_DIGITS for c in book_id):
        raise ValueError("书籍导入锁标识必须是 24 位小写十六进制字符。")
    lock_directory = self.paths.database.parent / ".import-locks"
    ensure_managed_directory(self.paths.root, lock_directory, create=True)
    with exclusive_lock(lock_directory / f"{book_id}.lock"):
        yield
```

Delete direct `fcntl`, `_required_open_flag`, `_secure_directory_open_flags`, and `_open_lock_directory` code from `importer.py`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_windows_importability.py tests/test_importer.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add book_agent/importer.py tests/test_importer.py tests/test_windows_importability.py
git commit -m "refactor: make import locking cross-platform"
```

### Task 4: Migrate VaultManager to portable publication

**Files:**
- Modify: `book_agent/vault.py`
- Modify: `tests/test_vault.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Add failing portable-vault tests**

Add tests that do not require hard links or `dir_fd`:

```python
def test_import_succeeds_when_hard_links_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    source = tmp_path / "book.txt"
    source.write_text("portable publication", encoding="utf-8")
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    imported = manager.import_original(source)
    assert imported.read_text(encoding="utf-8") == "portable publication"


def test_import_rejects_reparse_point_managed_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    real_lstat = Path.lstat
    def reparse_originals(path: Path):
        info = real_lstat(path)
        if path == manager.paths.originals:
            return _with_file_attributes(info, 0x400)
        return info
    monkeypatch.setattr(Path, "lstat", reparse_originals)
    with pytest.raises(ValueError, match="link|reparse|链接"):
        manager.ensure_layout()
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_vault.py tests/test_tools.py -q`  
Expected: FAIL because current publication requires hard links and secure directory FDs.

- [ ] **Step 3: Refactor VaultManager around `safe_fs`**

Keep the existing `VaultManager.ensure_layout`, `import_original`, `_inspect_original`, `_validate_original_identity`, and `_remove_original` method names and return types unchanged so `ImportService` and current tests retain a stable boundary.

Implement import as: validate source with `lstat`; capture source identity; copy through an opened source stream to a same-filesystem temp under `00-待导入`; flush and `fsync`; revalidate source identity; choose a collision-free final name while holding a directory lock; atomically replace the temp into `10-原始书籍`; validate final bytes and identity; clean only objects whose recorded identity still matches. Remove required hard-link publication and directory-FD helper functions.

- [ ] **Step 4: Run vault, tools, importer, and end-to-end tests**

Run: `uv run pytest tests/test_vault.py tests/test_tools.py tests/test_importer.py tests/test_end_to_end.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add book_agent/vault.py tests/test_vault.py tests/test_tools.py
git commit -m "refactor: publish imported books portably"
```

### Task 5: Migrate parsed Markdown and AI note publication

**Files:**
- Modify: `book_agent/rendering.py`
- Modify: `book_agent/notes.py`
- Modify: `tests/test_rendering.py`
- Modify: `tests/test_notes.py`

- [ ] **Step 1: Add failing no-hard-link and Windows-retry tests**

```python
def test_render_and_note_publish_without_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    parsed_path = _render_fixture(tmp_path)
    note_path = _save_note_fixture(tmp_path)
    assert parsed_path.is_file()
    assert note_path.is_file()


def test_atomic_replace_retries_windows_sharing_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0
    real_replace = os.replace
    def flaky(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "sharing violation")
        real_replace(source, destination)
    monkeypatch.setattr(os, "replace", flaky)
    atomic_write_bytes(tmp_path / "out.md", b"ok", root=tmp_path, overwrite=True)
    assert attempts == 3
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_rendering.py tests/test_notes.py -q`  
Expected: FAIL because both modules use hard links and directory-FD operations.

- [ ] **Step 3: Use shared atomic writers**

Render parsed books with:

```python
return atomic_write_bytes(
    destination,
    _render(book_id, parsed, source_file, passages).encode("utf-8"),
    root=managed_root_path,
    overwrite=True,
    expected_root=(
        None
        if expected_root_identity is None
        else FileIdentity(*expected_root_identity)
    ),
)
```

Publish notes with `overwrite=False` while retaining existing collision suffix behavior and citation validation. Extend `safe_fs.atomic_write_bytes` with three bounded `PermissionError` retries of 0.05, 0.10, and 0.20 seconds around `os.replace`.

- [ ] **Step 4: Run focused and integration tests**

Run: `uv run pytest tests/test_rendering.py tests/test_notes.py tests/test_tools.py tests/test_end_to_end.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add book_agent/safe_fs.py book_agent/rendering.py book_agent/notes.py tests/test_safe_fs.py tests/test_rendering.py tests/test_notes.py
git commit -m "refactor: publish parsed text and notes cross-platform"
```

### Task 6: Migrate SQLite and external Vault validation

**Files:**
- Modify: `book_agent/storage.py`
- Modify: `book_agent/tools.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Add failing reparse and portable-open tests**

```python
def test_database_rejects_windows_reparse_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "root" / "data" / "library.sqlite3", root=tmp_path / "root")
    _mark_lstat_as_reparse(monkeypatch, tmp_path / "root" / "data")
    with pytest.raises(ValueError, match="link|reparse|链接"):
        database.initialize()


def test_build_tools_rejects_reparse_obsidian_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    _mark_lstat_as_reparse(monkeypatch, vault)
    with pytest.raises(ValueError, match="link|reparse|链接"):
        build_tools(tmp_path / "project", vault_root=vault)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_storage.py tests/test_tools.py -q`  
Expected: FAIL because validation only recognizes POSIX symlinks and storage uses directory FDs.

- [ ] **Step 3: Replace directory-FD helpers**

In `Database._safe_parent`, call `ensure_managed_directory(self.root, self.path.parent, create=create)`, record the returned parent identity, inspect the leaf with `inspect_regular_file(self.path, allow_missing=True, require_single_link=True)`, open SQLite by absolute path, then revalidate parent and leaf identity. In `build_tools`, call `validate_existing_directory(explicit_vault, label="Explicit Obsidian vault")`, which rejects symlinks and Windows reparse points.

- [ ] **Step 4: Run full suite**

Run: `uv run pytest -q`  
Expected: all existing and new tests PASS on macOS.

- [ ] **Step 5: Commit**

```bash
git add book_agent/storage.py book_agent/tools.py tests/test_storage.py tests/test_tools.py
git commit -m "refactor: validate database and vault paths portably"
```

### Task 7: Generate machine-local Codex configuration

**Files:**
- Create: `installer/__init__.py`
- Create: `installer/codex_config.py`
- Create: `tests/installer/test_codex_config.py`

- [ ] **Step 1: Write failing rendering tests**

```python
def test_render_config_escapes_windows_paths_and_omits_vault_when_internal() -> None:
    config = render_codex_config(
        project_root=PureWindowsPath(r"C:\\Users\\朋友\\Book Library"),
        python=PureWindowsPath(r"C:\\Users\\朋友\\Book Library\\.venv\\Scripts\\python.exe"),
        obsidian_vault=None,
    )
    parsed = tomllib.loads(config)
    server = parsed["mcp_servers"]["book_library"]
    assert server["command"].endswith("python.exe")
    assert "BOOK_LIBRARY_OBSIDIAN_VAULT" not in server["env"]
    assert server["enabled_tools"] == TOOL_ALLOWLIST


def test_write_config_is_atomic_and_never_contains_template_markers(tmp_path: Path) -> None:
    path = write_codex_config(tmp_path, tmp_path / ".venv" / "bin" / "python", tmp_path / "vault")
    assert "{{" not in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/installer/test_codex_config.py -q`  
Expected: FAIL because the installer package does not exist.

- [ ] **Step 3: Implement config rendering and atomic write**

Use `json.dumps(str(value), ensure_ascii=False)` for TOML basic strings, render the exact six-tool allowlist, use the virtual environment Python as `command`, args `['-m', 'book_agent.mcp_server']`, set offline environment variables, and write `.codex/config.toml` only through `atomic_write_bytes`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/installer/test_codex_config.py tests/test_project_policy.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add installer tests/installer .codex/config.toml.template tests/test_project_policy.py
git commit -m "feat: generate local Codex MCP configuration"
```

### Task 8: Discover Obsidian and orchestrate installation

**Files:**
- Create: `installer/obsidian.py`
- Create: `installer/setup.py`
- Create: `tests/installer/test_obsidian.py`
- Create: `tests/installer/test_setup.py`

- [ ] **Step 1: Write failing discovery and rollback tests**

```python
def test_discovers_open_windows_obsidian_vault(tmp_path: Path) -> None:
    config = tmp_path / "obsidian.json"
    config.write_text(json.dumps({"vaults": {"a": {"path": r"C:\\Notes", "open": True}}}), encoding="utf-8")
    assert discover_vaults(config) == [Path(r"C:\\Notes")]


def test_failed_smoke_does_not_publish_codex_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup, "run_health_check", lambda plan: (_ for _ in ()).throw(RuntimeError("smoke failed")))
    with pytest.raises(RuntimeError, match="smoke failed"):
        setup.install(_plan(tmp_path))
    assert not (tmp_path / ".codex" / "config.toml").exists()


def test_install_is_idempotent(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    first = install(plan)
    second = install(plan)
    assert first.status == second.status == "ready"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/installer/test_obsidian.py tests/installer/test_setup.py -q`  
Expected: FAIL because discovery and setup modules do not exist.

- [ ] **Step 3: Implement installer orchestration**

Define:

```python
@dataclass(frozen=True)
class InstallPlan:
    project_root: Path
    python: Path
    vault: Path | None
    model_source: Path | None
    non_interactive: bool = False


@dataclass(frozen=True)
class InstallResult:
    status: str
    vault: Path
    config: Path
    log: Path
```

`install()` must validate supported OS/architecture, require at least 2 GiB free space, validate or create only the project-internal Vault root, create the four library subdirectories, install/verify the model through Task 9, build tools, require six registered MCP tools, require `library_status()['ok'] is True`, then publish config last. Log details to `data/install.log`; return one Chinese success/failure summary from CLI `main()`.

- [ ] **Step 4: Run installer tests**

Run: `uv run pytest tests/installer/test_obsidian.py tests/installer/test_setup.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add installer/obsidian.py installer/setup.py tests/installer/test_obsidian.py tests/installer/test_setup.py
git commit -m "feat: add guided cross-platform installer"
```

### Task 9: Pin, flatten, verify, and install the E5 model

**Files:**
- Create: `distribution/release.json`
- Create: `installer/model_bundle.py`
- Modify: `book_agent/embeddings.py`
- Create: `tests/installer/test_model_bundle.py`
- Modify: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing manifest and corrupt-bundle tests**

```python
def test_release_metadata_is_fully_pinned() -> None:
    metadata = json.loads((PROJECT_ROOT / "distribution/release.json").read_text(encoding="utf-8"))
    assert metadata["model_revision"] == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert metadata["uv_version"] == "0.11.26"
    assert metadata["release_tag"] == "v0.1.0-beta.1"


def test_install_model_rejects_checksum_mismatch(tmp_path: Path) -> None:
    bundle = _fake_complete_model_bundle(tmp_path)
    (bundle / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256|checksum|校验"):
        install_model_bundle(bundle, tmp_path / "data" / "models")
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/installer/test_model_bundle.py tests/test_embeddings.py -q`  
Expected: FAIL because pinned metadata and model-bundle module do not exist.

- [ ] **Step 3: Implement manifest and safe model installation**

Create `distribution/release.json` using the fixed values at the top of this plan. Implement manifest entries as `{path, size, sha256}`; reject absolute paths, `..`, symlinks, duplicate normalized paths, missing files, unexpected files, size differences, and digest differences. Copy verified flat files to a temp directory under `data/`, reverify, atomically rename to `data/models`, and run `E5EmbeddingProvider(project_root / "data" / "models").embed_query("离线验证")`, requiring shape `(384,)`.

Update `E5EmbeddingProvider` to expose the loaded local path and model revision in `library_status`, without performing a network lookup.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/installer/test_model_bundle.py tests/test_embeddings.py tests/test_tools.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add distribution/release.json installer/model_bundle.py book_agent/embeddings.py tests/installer/test_model_bundle.py tests/test_embeddings.py tests/test_tools.py
git commit -m "feat: install a pinned offline E5 model bundle"
```

### Task 10: Add double-click macOS and Windows launchers

**Files:**
- Create: `installer/bootstrap-macos.sh`
- Create: `installer/bootstrap-windows.ps1`
- Create: `install-macos.command`
- Create: `install-windows.bat`
- Create: `tests/installer/test_launchers.py`

- [ ] **Step 1: Write failing launcher contract tests**

```python
def test_macos_launcher_uses_bundled_uv_then_pinned_official_fallback() -> None:
    text = (PROJECT_ROOT / "installer/bootstrap-macos.sh").read_text(encoding="utf-8")
    assert "vendor/uv/macos-arm64/uv" in text
    assert "https://astral.sh/uv/0.11.26/install.sh" in text
    assert "uv sync --frozen --extra semantic" in text
    assert "python -m installer.setup" in text


def test_windows_launcher_uses_bundled_uv_and_bypasses_only_child_process_policy() -> None:
    batch = (PROJECT_ROOT / "install-windows.bat").read_text(encoding="utf-8")
    ps = (PROJECT_ROOT / "installer/bootstrap-windows.ps1").read_text(encoding="utf-8")
    assert "-ExecutionPolicy Bypass" in batch
    assert "vendor\\uv\\windows-x64\\uv.exe" in ps
    assert "https://astral.sh/uv/0.11.26/install.ps1" in ps
    assert "uv sync --frozen --extra semantic" in ps
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/installer/test_launchers.py -q`  
Expected: FAIL because launcher files do not exist.

- [ ] **Step 3: Implement thin launchers**

`install-macos.command` resolves its own directory and executes `bash installer/bootstrap-macos.sh`; `install-windows.bat` resolves `%~dp0` and invokes only the bundled `bootstrap-windows.ps1` with process-scoped bypass. Both bootstraps prefer the packaged uv binary, fall back to the pinned official uv installer, run `uv sync --frozen --extra semantic`, locate `.venv` Python, and invoke `python -m installer.setup --project-root "$PROJECT_ROOT"` on macOS or `python -m installer.setup --project-root $ProjectRoot` in PowerShell.

- [ ] **Step 4: Execute launcher checks**

Run: `chmod +x install-macos.command installer/bootstrap-macos.sh`  
Run: `bash -n install-macos.command installer/bootstrap-macos.sh`  
Expected: exit 0.

Run: `uv run pytest tests/installer/test_launchers.py -q`  
Expected: PASS; Windows execution is verified later by CI.

- [ ] **Step 5: Commit**

```bash
git add install-macos.command install-windows.bat installer/bootstrap-macos.sh installer/bootstrap-windows.ps1 tests/installer/test_launchers.py
git commit -m "feat: add one-click macOS and Windows installers"
```

### Task 11: Write public documentation and licenses

**Files:**
- Create: `README.md`
- Create: `docs/安装说明.md`
- Create: `docs/使用说明.md`
- Create: `docs/常见问题.md`
- Create: `docs/隐私与数据存放.md`
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`
- Modify or remove: `outputs/书库RAG快速开始.md`
- Modify: `docs/USER_GUIDE.md`
- Modify: `vault/首页.md`
- Modify: `vault/书库/说明.md`
- Modify: `AGENTS.md`
- Modify: `tests/test_user_guide.py`
- Modify: `tests/test_project_policy.py`

- [ ] **Step 1: Write failing documentation tests**

```python
def test_public_docs_cover_both_one_click_installers() -> None:
    install = (PROJECT_ROOT / "docs/安装说明.md").read_text(encoding="utf-8")
    for phrase in ("install-macos.command", "install-windows.bat", "信任", "重新加载", "Obsidian"):
        assert phrase in install


def test_tracked_text_has_no_personal_paths() -> None:
    forbidden = (
        "/" + "Users" + "/" + "zhao" + "yunfei",
        "Obsidian" + "_workspace",
        "Documents" + "/Codex/" + "2026-07-12",
    )
    for path in tracked_text_files(PROJECT_ROOT):
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(value in text for value in forbidden), path
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_user_guide.py tests/test_project_policy.py -q`  
Expected: FAIL because the new docs do not exist and current docs contain personal paths.

- [ ] **Step 3: Write the documents**

The install guide must use five numbered user actions: download ZIP, extract, double-click the platform installer, confirm/select Vault, open/trust/reload the folder in Codex. The usage guide must include exact prompts for import, quote, explain, compare, and save. The FAQ must cover missing MCP UI listing despite successful source calls, `keyword_only`, `needs_ocr`, duplicate import, timeouts, Windows policy prompts, and macOS Gatekeeper. The privacy guide must distinguish local files from selected evidence entering Codex context.

Add an MIT `LICENSE`. In `THIRD_PARTY_NOTICES.md`, identify `intfloat/multilingual-e5-small`, its fixed revision, MIT license, and source URL; also identify bundled uv 0.11.26 and its upstream license/source.

- [ ] **Step 4: Run documentation tests**

Run: `uv run pytest tests/test_user_guide.py tests/test_project_policy.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs outputs vault AGENTS.md LICENSE THIRD_PARTY_NOTICES.md tests/test_user_guide.py tests/test_project_policy.py
git commit -m "docs: add public installation and usage guides"
```

### Task 12: Build and privacy-scan the clean public repository and ZIP

**Files:**
- Create: `scripts/build_release.py`
- Create: `scripts/scan_release.py`
- Create: `tests/test_release_packaging.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing allowlist and leak tests**

```python
def test_release_builder_excludes_runtime_and_private_content(tmp_path: Path) -> None:
    output = build_source_tree(PROJECT_ROOT, tmp_path / "public")
    assert not (output / ".git").exists()
    assert not (output / ".venv").exists()
    assert not (output / "data").exists()
    assert not (output / ".codex" / "config.toml").exists()
    assert (output / ".codex" / "config.toml.template").is_file()


def test_scanner_rejects_personal_path_inside_zip(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    private_path = "/" + "Users" + "/" + "private-person/private"
    _zip_text(archive, "README.md", private_path)
    with pytest.raises(ReleaseScanError, match="personal|private|路径"):
        scan_archive(archive)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_release_packaging.py -q`  
Expected: FAIL because packaging modules do not exist.

- [ ] **Step 3: Implement deterministic allowlisted packaging**

The source allowlist must include only `.github`, `.codex/config.toml.template`, `book_agent`, `installer`, `scripts`, `distribution`, `tests`, the five end-user documents (`README.md`, `docs/安装说明.md`, `docs/使用说明.md`, `docs/常见问题.md`, `docs/隐私与数据存放.md`, plus the generic `docs/USER_GUIDE.md`), empty `vault` templates, `AGENTS.md`, `.gitignore`, `.python-version`, `pyproject.toml`, `uv.lock`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, and the two launchers. Explicitly exclude `docs/superpowers/`, `docs/plans/`, `outputs/`, private development history, and every other unlisted path. Follow model-cache symlinks only while copying the single pinned snapshot into a flat `bundled-model/`; never copy `.no_exist`, `blobs`, `refs`, `.locks`, or unrelated snapshots.

Download and verify uv 0.11.26 release assets for `aarch64-apple-darwin` and `x86_64-pc-windows-msvc`, placing only executables and licenses under `vendor/uv/`. Create the universal ZIP, a model-only ZIP for Git clones, and `.sha256` files. Normalize ZIP timestamps and sort entries for reproducibility.

The scanner must reject macOS home prefixes assembled as `"/" + "Users" + "/"`, Windows home prefixes assembled without embedding a literal personal path in scanner source, the current username, `.sqlite3`, database headers, original book extensions under managed content folders, non-empty AI-note folders, symlinks, files over the declared allowlist, and any `.codex/config.toml` other than the template.

- [ ] **Step 4: Run packaging tests and a dry build**

Run: `uv run pytest tests/test_release_packaging.py -q`  
Expected: PASS.

Run: `uv run python scripts/build_release.py --dry-run --model-cache data/models`  
Expected: exit 0 and a manifest summary without writing public assets.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_release.py scripts/scan_release.py tests/test_release_packaging.py .gitignore
git commit -m "build: create privacy-scanned release packages"
```

### Task 13: Add macOS and Windows CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/release-smoke.yml`
- Create: `tests/test_ci_contract.py`

- [ ] **Step 1: Write failing workflow-contract tests**

```python
def test_ci_matrix_covers_supported_platforms() -> None:
    workflow = yaml.safe_load((PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["os"]
    assert "macos-14" in matrix
    assert "windows-latest" in matrix
    assert "uv run pytest -q" in json.dumps(workflow)
```

Use `yaml.BaseLoader` or inspect text if YAML 1.1 boolean coercion would make `on` ambiguous; do not add PyYAML only for this test.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_ci_contract.py -q`  
Expected: FAIL because workflows are missing.

- [ ] **Step 3: Add workflow matrix and release smoke**

`ci.yml` must use `astral-sh/setup-uv`, Python 3.12, `uv sync --frozen --extra dev --extra semantic`, and `uv run pytest -q` on both platforms. It must also run `python scripts/scan_release.py --tree .` and the launcher contract tests.

`release-smoke.yml` must be manually dispatchable with a Release tag, download the all-in-one ZIP on macOS 14 and Windows latest, extract it, invoke the bootstrap non-interactively with the project-internal Vault, run offline model inference, import a small fixture book, search it, and verify `library_status`.

- [ ] **Step 4: Validate workflows locally**

Run: `uv run pytest tests/test_ci_contract.py -q`  
Expected: PASS.

Run: `uv run pytest -q`  
Expected: all tests PASS on macOS before publishing.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows tests/test_ci_contract.py
git commit -m "ci: test macOS and Windows distributions"
```

### Task 14: Build candidate assets and perform a clean macOS install

**Files:**
- Generated, ignored: `dist/codex-obsidian-book-library-v0.1.0-beta.1-all-in-one.zip`
- Generated, ignored: `dist/codex-book-library-model-v0.1.0-beta.1.zip`
- Generated, ignored: `dist/*.sha256`

- [ ] **Step 1: Run the complete suite before packaging**

Run: `uv run pytest --cov=book_agent --cov=installer --cov-report=term-missing -q`  
Expected: PASS with zero failures.

- [ ] **Step 2: Build real assets from the pinned local model cache**

Run: `uv run python scripts/build_release.py --model-cache data/models --output dist`  
Expected: exit 0; both ZIP assets and SHA-256 files exist; scanner reports zero privacy findings.

- [ ] **Step 3: Verify archive and checksum**

Run: `uv run python scripts/scan_release.py --archive dist/codex-obsidian-book-library-v0.1.0-beta.1-all-in-one.zip`  
Expected: `0 findings`.

Run: `shasum -a 256 -c dist/codex-obsidian-book-library-v0.1.0-beta.1-all-in-one.zip.sha256`  
Expected: `OK`.

- [ ] **Step 4: Perform a clean non-interactive install from the ZIP**

Extract to a new directory under `/private/tmp`, run the macOS bootstrap with project-internal Vault and bundled model, start the MCP server far enough to list all six tools, import a generated TXT fixture, search its unique phrase, and save a cited note. Expected: all operations return `ok: true`, model query shape is `(384,)`, and no files appear outside the extracted project/Vault.

- [ ] **Step 5: Commit any fixes, then tag the candidate locally**

Run the full suite again after clean-install fixes. Create the tag only after tests pass:

```bash
git tag -a v0.1.0-beta.1 -m "Codex Obsidian Book Library v0.1.0-beta.1"
```

### Task 15: Publish a clean public repository, verify Windows CI, and create Release

**Files:**
- External clean repository generated under `/private/tmp/codex-obsidian-book-library-public`
- Public GitHub repository: `codex-obsidian-book-library`

- [ ] **Step 1: Verify GitHub identity and repository absence**

Run: `gh auth status`  
Expected: authenticated GitHub account with repository creation permission.

Run: `gh repo view codex-obsidian-book-library`  
Expected: not found. If it exists, inspect it and update it only if it is owned by the authenticated user and clearly belongs to this project; never overwrite an unrelated repository.

- [ ] **Step 2: Generate the clean public Git snapshot**

Use `scripts/build_release.py --public-tree /private/tmp/codex-obsidian-book-library-public`, run the privacy scanner, initialize a new Git repository there, create a single initial commit, and verify `git log` contains none of the private development history.

- [ ] **Step 3: Create and push the public repository**

Run `gh repo create codex-obsidian-book-library --public --source /private/tmp/codex-obsidian-book-library-public --remote origin --push`.  
Expected: public repository URL returned and `main` pushed successfully.

- [ ] **Step 4: Wait for and inspect GitHub Actions**

Resolve the owner with `gh api user --jq .login`, then use `gh run list --repo "$OWNER/codex-obsidian-book-library" --limit 5` to identify the commit run and `gh run watch "$RUN_ID" --repo "$OWNER/codex-obsidian-book-library" --exit-status`.  
Expected: macOS and Windows jobs both succeed. If either fails, inspect logs, fix in the private development repository with a test first, regenerate the clean snapshot, push the corrected public commit, and repeat until green.

- [ ] **Step 5: Create the public prerelease**

Create `v0.1.0-beta.1` as a prerelease with the all-in-one ZIP, model-only ZIP, both SHA-256 files, and concise installation links. Then manually dispatch `release-smoke.yml` for the tag and require both macOS and Windows smoke jobs to succeed.

- [ ] **Step 6: Final acceptance and handoff**

Verify:

```text
[ ] Public repository opens without authentication
[ ] Release assets download without authentication
[ ] macOS CI passes
[ ] Windows CI passes
[ ] Release smoke passes on both platforms
[ ] All-in-one ZIP includes model and both launchers
[ ] Source repository includes installation and usage guides
[ ] Privacy scanner reports zero findings
[ ] No user books, database, notes, model cache symlinks, or personal paths are public
```

Return the public repository URL, direct all-in-one ZIP Release page, local ZIP path, version, SHA-256, supported platforms, and the two documentation links to the user.
