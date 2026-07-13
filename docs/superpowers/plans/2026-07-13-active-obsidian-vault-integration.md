# Active Obsidian Vault Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Codex book RAG write originals, parsed Markdown, and AI notes into the user's currently open Obsidian vault while keeping SQLite data and embedding models inside the Codex project.

**Architecture:** Add an optional external vault root to `AppPaths`, then maintain two explicit filesystem trust roots: the Obsidian vault for user-visible book files and the project root for database/model data. Thread the vault root through `build_tools` and the MCP environment, preserve the project-local vault as the no-argument fallback, and verify the real active Obsidian vault after the automated suite passes.

**Tech Stack:** Python 3.11+, pathlib, secure directory-FD filesystem operations, FastMCP, SQLite, pytest, TOML, Obsidian.

---

## File map

- `book_agent/config.py` — calculate project-data paths and an optional independent Obsidian vault path.
- `book_agent/vault.py` — enforce separate Obsidian and project filesystem trust roots.
- `book_agent/tools.py` — validate the explicit vault and construct all services with split paths.
- `book_agent/rendering.py` — publish parsed Markdown beneath a generic managed root.
- `book_agent/importer.py` — pass the Obsidian vault as the parsed-Markdown managed root.
- `book_agent/mcp_server.py` — read `BOOK_LIBRARY_OBSIDIAN_VAULT` and pass it to the tool factory.
- `.codex/config.toml` — point the project MCP at the user's active Obsidian vault.
- `tests/test_config.py` — cover default and external path calculation without following vault symlinks.
- `tests/test_vault.py` — cover split layout and external-vault symlink rejection.
- `tests/test_tools.py` — cover explicit-vault validation and the complete external-vault workflow.
- `tests/test_mcp_server.py` — cover environment propagation and startup layout.
- `tests/test_project_policy.py` — lock the real project MCP configuration.
- `tests/test_user_guide.py` — prevent the guide from reverting to the old “open another vault” workflow.
- `docs/USER_GUIDE.md` — document current-vault browsing and local-data placement.
- `outputs/书库RAG快速开始.md` — update the short user-facing workflow.
- `AGENTS.md` — describe the AI-note evidence boundary without the obsolete project-vault prefix.

## Task 1: Support an independent vault path in `AppPaths`

**Files:**
- Modify: `tests/test_config.py:1-22`
- Modify: `book_agent/config.py:1-39`

- [ ] **Step 1: Write the failing external-path tests**

Append these tests to `tests/test_config.py`:

```python
def test_app_paths_keep_external_obsidian_files_separate_from_project_data(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    obsidian_vault = tmp_path / "current-obsidian"

    paths = AppPaths.from_root(project_root, vault_root=obsidian_vault)

    assert paths.root == project_root.resolve()
    assert paths.vault == obsidian_vault.absolute()
    assert paths.library == obsidian_vault.absolute() / "书库"
    assert paths.inbox == obsidian_vault.absolute() / "书库" / "00-待导入"
    assert paths.originals == obsidian_vault.absolute() / "书库" / "10-原始书籍"
    assert paths.parsed == obsidian_vault.absolute() / "书库" / "20-解析文本"
    assert paths.notes == obsidian_vault.absolute() / "书库" / "30-AI读书笔记"
    assert paths.database == project_root.resolve() / "data" / "library.sqlite3"
    assert paths.models == project_root.resolve() / "data" / "models"


def test_app_paths_do_not_follow_an_external_vault_symlink(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    target = tmp_path / "real-vault"
    target.mkdir()
    alias = tmp_path / "vault-alias"
    alias.symlink_to(target, target_is_directory=True)

    paths = AppPaths.from_root(project_root, vault_root=alias)

    assert paths.vault == alias.absolute()
    assert paths.vault != target.resolve()
```

- [ ] **Step 2: Run the tests and verify the new API is absent**

Run:

```bash
uv run pytest tests/test_config.py -v
```

Expected: the existing default-path test passes and both new tests fail with `TypeError: AppPaths.from_root() got an unexpected keyword argument 'vault_root'`.

- [ ] **Step 3: Implement path separation without following the configured vault symlink**

Change `book_agent/config.py` to:

```python
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self


MAX_PREVIEWS = 10
MAX_FULL_PASSAGES = 6
MAX_EVIDENCE_TOKENS = 8000


def _absolute_without_following_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


@dataclass(frozen=True)
class AppPaths:
    root: Path
    vault: Path
    library: Path
    inbox: Path
    originals: Path
    parsed: Path
    notes: Path
    database: Path
    models: Path

    @classmethod
    def from_root(cls, root: Path, vault_root: Path | None = None) -> Self:
        resolved_root = root.expanduser().resolve()
        vault = (
            resolved_root / "vault"
            if vault_root is None
            else _absolute_without_following_symlinks(vault_root)
        )
        library = vault / "书库"

        return cls(
            root=resolved_root,
            vault=vault,
            library=library,
            inbox=library / "00-待导入",
            originals=library / "10-原始书籍",
            parsed=library / "20-解析文本",
            notes=library / "30-AI读书笔记",
            database=resolved_root / "data" / "library.sqlite3",
            models=resolved_root / "data" / "models",
        )
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
uv run pytest tests/test_config.py -v
```

Expected: all tests in `tests/test_config.py` pass.

- [ ] **Step 5: Commit the path model**

```bash
git add book_agent/config.py tests/test_config.py
git commit -m "feat: separate Obsidian and project paths"
```

## Task 2: Enforce two independent secure filesystem roots

**Files:**
- Modify: `tests/test_vault.py:27-48`
- Modify: `book_agent/vault.py:54-72,159-205,266-280`

- [ ] **Step 1: Write failing layout and symlink tests**

Add to `tests/test_vault.py`:

```python
def test_ensure_layout_splits_external_vault_from_project_data(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    obsidian_vault = tmp_path / "current-obsidian"
    obsidian_vault.mkdir()
    paths = AppPaths.from_root(project_root, vault_root=obsidian_vault)

    VaultManager(paths).ensure_layout()

    for directory in (paths.inbox, paths.originals, paths.parsed, paths.notes):
        assert directory.is_dir()
        assert directory.is_relative_to(obsidian_vault)
    assert paths.models.is_dir()
    assert paths.models.is_relative_to(project_root)
    assert paths.database.parent.is_dir()
    assert paths.database.parent.is_relative_to(project_root)
    assert not (project_root / "vault").exists()


def test_external_vault_root_symlink_is_rejected_before_any_write(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "current-obsidian"
    alias.symlink_to(outside, target_is_directory=True)
    paths = AppPaths.from_root(project_root, vault_root=alias)

    with pytest.raises(ValueError, match="symlink"):
        VaultManager(paths).ensure_layout()

    assert list(outside.iterdir()) == []
```

- [ ] **Step 2: Run the tests and verify the old single-root confinement fails**

Run:

```bash
uv run pytest tests/test_vault.py::test_ensure_layout_splits_external_vault_from_project_data tests/test_vault.py::test_external_vault_root_symlink_is_rejected_before_any_write -v
```

Expected: the split-layout test fails because the external book directories are rejected as outside the project root.

- [ ] **Step 3: Generalize the low-level root label**

Update the confinement helper signatures and messages in `book_agent/vault.py`:

```python
def _confined_path(
    configured: Path,
    label: str,
    root: Path,
    root_label: str,
) -> Path:
    try:
        expanded = configured.expanduser()
        if not expanded.is_absolute():
            expanded = root / expanded
        directory = Path(os.path.abspath(os.fspath(expanded)))
        directory.relative_to(root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Managed directory '{label}' is not beneath {root_label}: {configured}"
        ) from exc
    return directory
```

Replace `_managed_directory_beneath` with:

```python
@contextmanager
def _managed_directory_beneath(
    root_path: Path,
    configured: Path,
    label: str,
    *,
    create: bool,
    root_label: str = "project root",
) -> Iterator[tuple[Path, int]]:
    root = _absolute_path(root_path, root_label)
    directory = _confined_path(configured, label, root, root_label)
    root_fd = _open_absolute_directory(root, root_label, create=create)
    current_fd = root_fd
    current_path = root
    try:
        for component in directory.relative_to(root).parts:
            next_fd = _open_directory_component(
                current_fd,
                current_path,
                component,
                label,
                create=create,
            )
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)
            current_path /= component
        yield directory, current_fd
    finally:
        os.close(current_fd)
```

- [ ] **Step 4: Split `VaultManager` layout and private directory helpers**

Replace `ensure_layout` and add the project-only helper:

```python
def ensure_layout(self) -> None:
    vault_directories = (
        ("inbox", self.paths.inbox),
        ("originals", self.paths.originals),
        ("parsed", self.paths.parsed),
        ("notes", self.paths.notes),
    )
    for label, directory in vault_directories:
        with self._managed_directory(directory, label, create=True):
            pass

    project_directories = (
        ("models", self.paths.models),
        ("database.parent", self.paths.database.parent),
    )
    for label, directory in project_directories:
        with self._project_directory(directory, label, create=True):
            pass
```

Keep `_managed_directory` private and make it vault-only:

```python
@contextmanager
def _managed_directory(
    self,
    configured: Path,
    label: str,
    *,
    create: bool,
) -> Iterator[tuple[Path, int]]:
    with _managed_directory_beneath(
        self.paths.vault,
        configured,
        label,
        create=create,
        root_label="Obsidian vault",
    ) as managed:
        yield managed


@contextmanager
def _project_directory(
    self,
    configured: Path,
    label: str,
    *,
    create: bool,
) -> Iterator[tuple[Path, int]]:
    with _managed_directory_beneath(
        self.paths.root,
        configured,
        label,
        create=create,
        root_label="project root",
    ) as managed:
        yield managed
```

- [ ] **Step 5: Run the vault and storage safety suites**

Run:

```bash
uv run pytest tests/test_vault.py tests/test_storage.py tests/test_notes.py -v
```

Expected: all tests pass, including the unchanged assertion that `VaultManager` exposes only `ensure_layout` and `import_original` as public methods.

- [ ] **Step 6: Commit the trust-root split**

```bash
git add book_agent/vault.py tests/test_vault.py
git commit -m "feat: split vault and project trust roots"
```

## Task 3: Validate and construct an explicit Obsidian vault

**Files:**
- Modify: `tests/test_tools.py:28-34`
- Modify: `book_agent/tools.py:1-15,288-314`

- [ ] **Step 1: Write failing tool-factory tests**

Add to `tests/test_tools.py`:

```python
def test_build_tools_uses_external_vault_for_layout_and_project_for_data(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    obsidian_vault = tmp_path / "current-obsidian"
    obsidian_vault.mkdir()

    library = build_tools(
        project_root,
        embedding_provider=NullEmbeddingProvider(),
        vault_root=obsidian_vault,
    )

    assert library.paths.vault == obsidian_vault.absolute()
    assert library.paths.library == obsidian_vault.absolute() / "书库"
    assert library.paths.database == project_root.resolve() / "data" / "library.sqlite3"
    assert library.paths.database.is_file()
    assert library.paths.models == project_root.resolve() / "data" / "models"
    assert library.paths.models.is_dir()
    assert not (project_root / "vault").exists()


def test_build_tools_rejects_a_missing_explicit_obsidian_vault(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    missing_vault = tmp_path / "missing-obsidian"

    with pytest.raises(ValueError, match="does not exist"):
        build_tools(
            project_root,
            embedding_provider=NullEmbeddingProvider(),
            vault_root=missing_vault,
        )

    assert not missing_vault.exists()
```

- [ ] **Step 2: Run the tests and confirm the factory lacks `vault_root`**

Run:

```bash
uv run pytest tests/test_tools.py::test_build_tools_uses_external_vault_for_layout_and_project_for_data tests/test_tools.py::test_build_tools_rejects_a_missing_explicit_obsidian_vault -v
```

Expected: both tests fail because `build_tools` does not accept `vault_root`.

- [ ] **Step 3: Add explicit-vault validation and thread it into `AppPaths`**

Add `import stat` near the imports in `book_agent/tools.py`, then add:

```python
def _require_existing_obsidian_vault(vault: Path) -> None:
    try:
        info = vault.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Configured Obsidian vault does not exist: {vault}") from exc
    except OSError as exc:
        raise ValueError(f"Configured Obsidian vault is unavailable: {vault}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"Configured Obsidian vault must not be a symlink: {vault}")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Configured Obsidian vault is not a directory: {vault}")
```

Replace the `build_tools` signature and path initialization with:

```python
def build_tools(
    project_root: str | Path,
    embedding_provider: object | None = None,
    *,
    vault_root: str | Path | None = None,
) -> LibraryTools:
    """Initialize the managed local library without downloading a model."""

    configured_vault = None if vault_root is None else Path(vault_root)
    paths = AppPaths.from_root(
        Path(project_root).expanduser(),
        vault_root=configured_vault,
    )
    if configured_vault is not None:
        _require_existing_obsidian_vault(paths.vault)
    VaultManager(paths).ensure_layout()
    database = Database(paths.database, root=paths.root)
    database.initialize()

    provider = embedding_provider
    if provider is None:
        local_e5 = E5EmbeddingProvider(paths.models)
        provider = local_e5 if local_e5.available else NullEmbeddingProvider()

    importer = ImportService(paths, database, provider)
    retriever = Retriever(database, provider)
    notes = NoteService(paths, database)
    return LibraryTools(
        paths=paths,
        database=database,
        importer=importer,
        retriever=retriever,
        notes=notes,
        embedding_provider=provider,
    )
```

- [ ] **Step 4: Run the factory and existing tools tests**

Run:

```bash
uv run pytest tests/test_tools.py -v
```

Expected: all `tests/test_tools.py` tests pass.

- [ ] **Step 5: Commit explicit-vault construction**

```bash
git add book_agent/tools.py tests/test_tools.py
git commit -m "feat: construct tools with explicit Obsidian vault"
```

## Task 4: Route originals, parsed Markdown, and notes to the external vault

**Files:**
- Modify: `tests/test_tools.py:36-115`
- Modify: `book_agent/rendering.py:100-118`
- Modify: `book_agent/importer.py:283-297`

- [ ] **Step 1: Write the failing end-to-end split-path test**

Add to `tests/test_tools.py`:

```python
def test_external_vault_receives_all_user_files_while_project_keeps_data(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    obsidian_vault = tmp_path / "current-obsidian"
    obsidian_vault.mkdir()
    library = build_tools(
        project_root,
        embedding_provider=NullEmbeddingProvider(),
        vault_root=obsidian_vault,
    )
    source = _write_chinese_book(tmp_path / "外部书库测试.txt")

    imported = library.import_book(str(source), author="测试作者")
    searched = library.search_books("库存周期", mode="quote", limit=3)
    saved = library.save_reading_note(
        "外部书库路径验证",
        "这是一条带原文依据的验证笔记。",
        [searched["results"][0]["passage_id"]],
    )

    assert imported["status"] == "keyword_only"
    assert Path(imported["original_path"]).is_file()
    assert Path(imported["original_path"]).is_relative_to(
        obsidian_vault / "书库" / "10-原始书籍"
    )
    assert imported["parsed_path"] is not None
    assert Path(imported["parsed_path"]).is_file()
    assert Path(imported["parsed_path"]).is_relative_to(
        obsidian_vault / "书库" / "20-解析文本"
    )
    assert Path(saved["path"]).is_file()
    assert Path(saved["path"]).is_relative_to(
        obsidian_vault / "书库" / "30-AI读书笔记"
    )
    assert saved["wiki_link"].startswith("[[书库/30-AI读书笔记/")
    assert library.paths.database.is_file()
    assert library.paths.database.is_relative_to(project_root)
    assert library.paths.models.is_dir()
    assert library.paths.models.is_relative_to(project_root)
    assert not (project_root / "vault").exists()
```

- [ ] **Step 2: Run the test and verify parsed rendering is still project-confined**

Run:

```bash
uv run pytest tests/test_tools.py::test_external_vault_receives_all_user_files_while_project_keeps_data -v
```

Expected: the import ends in `failed` because `render_parsed_book` still receives the project root while its destination is in the external vault.

- [ ] **Step 3: Rename the rendering boundary to a generic managed root**

Change the keyword and local variable in `book_agent/rendering.py`:

```python
def render_parsed_book(
    destination: str | Path,
    book_id: str,
    parsed: ParsedBook,
    source_file: str | Path,
    passages: Iterable[Passage],
    *,
    managed_root: str | Path | None = None,
) -> Path:
    destination = Path(os.path.abspath(os.fspath(Path(destination).expanduser())))
    root = Path(destination.anchor) if managed_root is None else Path(managed_root)
    content = _render(book_id, parsed, source_file, passages)

    with _managed_directory_beneath(
        root,
        destination.parent,
        "parsed book",
        create=True,
        root_label="Obsidian vault",
    ) as (_, directory_fd):
        try:
            destination_info = os.stat(
                destination.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(destination_info.st_mode):
                raise ValueError("Parsed Markdown destination must not be a symlink")

        temp_name = f".render-{secrets.token_hex(12)}"
        temp_fd = os.open(
            temp_name,
            _secure_create_open_flags(),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            try:
                _write_complete(temp_fd, content.encode("utf-8"))
            finally:
                os.close(temp_fd)
            try:
                os.replace(
                    temp_name,
                    destination.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            except (TypeError, NotImplementedError) as exc:
                raise RuntimeError(
                    "This platform cannot atomically publish parsed Markdown by directory FD"
                ) from exc
            temp_name = None
        finally:
            _remove_temp(directory_fd, temp_name)
    return destination
```

- [ ] **Step 4: Pass the Obsidian root from the importer**

Change the call in `book_agent/importer.py` to:

```python
render_parsed_book(
    destination,
    book_id,
    parsed,
    original,
    passages,
    managed_root=self.paths.vault,
)
```

`NoteService` needs no new parameter: after Task 2, its existing `VaultManager._managed_directory` call is already confined to `paths.vault`.

- [ ] **Step 5: Run the end-to-end and safety suites**

Run:

```bash
uv run pytest tests/test_tools.py::test_external_vault_receives_all_user_files_while_project_keeps_data tests/test_importer.py tests/test_rendering.py tests/test_notes.py -v
```

Expected: all selected tests pass; the end-to-end test reports `keyword_only` because it deliberately uses `NullEmbeddingProvider`.

- [ ] **Step 6: Commit external-vault file routing**

```bash
git add book_agent/rendering.py book_agent/importer.py tests/test_tools.py
git commit -m "feat: route book artifacts to external vault"
```

## Task 5: Wire the active vault through MCP and project configuration

**Files:**
- Modify: `tests/test_mcp_server.py:23-90`
- Modify: `tests/test_project_policy.py:5-67`
- Modify: `book_agent/mcp_server.py:23-25`
- Modify: `.codex/config.toml:1-5`

- [ ] **Step 1: Write failing MCP environment assertions**

Update the `server_module` fixture in `tests/test_mcp_server.py`:

```python
@pytest.fixture
def server_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    root = tmp_path / "mcp-library"
    vault = tmp_path / "current-obsidian"
    vault.mkdir()
    monkeypatch.setenv("BOOK_LIBRARY_ROOT", str(root))
    monkeypatch.setenv("BOOK_LIBRARY_OBSIDIAN_VAULT", str(vault))
    sys.modules.pop("book_agent.mcp_server", None)
    module = importlib.import_module("book_agent.mcp_server")
    yield module
    sys.modules.pop("book_agent.mcp_server", None)
```

Add these assertions to `test_tool_names_and_actual_fastmcp_registration_are_exact`:

```python
assert server_module.OBSIDIAN_VAULT == Path(
    os.environ["BOOK_LIBRARY_OBSIDIAN_VAULT"]
).absolute()
assert server_module.library_tools.paths.vault == server_module.OBSIDIAN_VAULT
assert server_module.library_tools.paths.database.is_relative_to(server_module.ROOT)
```

Update the subprocess test setup and final assertions:

```python
vault = tmp_path / "subprocess-obsidian"
vault.mkdir()
environment["BOOK_LIBRARY_OBSIDIAN_VAULT"] = str(vault)
```

```python
assert (root / "data" / "library.sqlite3").is_file()
assert (vault / "书库" / "10-原始书籍").is_dir()
assert (vault / "书库" / "20-解析文本").is_dir()
assert (vault / "书库" / "30-AI读书笔记").is_dir()
assert not (root / "vault").exists()
```

- [ ] **Step 2: Lock the real project vault path in the policy test**

Add near the constants in `tests/test_project_policy.py`:

```python
ACTIVE_OBSIDIAN_VAULT = "/Users/zhaoyunfei/Documents/Obsidian_workspace"
```

Change the expected environment mapping to:

```python
assert server["env"] == {
    "BOOK_LIBRARY_ROOT": FINAL_PROJECT_ROOT,
    "BOOK_LIBRARY_OBSIDIAN_VAULT": ACTIVE_OBSIDIAN_VAULT,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
```

- [ ] **Step 3: Run the tests and confirm the environment is not consumed yet**

Run:

```bash
uv run pytest tests/test_mcp_server.py tests/test_project_policy.py -v
```

Expected: failures mention missing `OBSIDIAN_VAULT`, the project-local vault being created, and the absent TOML environment key.

- [ ] **Step 4: Read and pass the optional environment path**

Replace the startup path block in `book_agent/mcp_server.py` with:

```python
ROOT = Path(os.environ.get("BOOK_LIBRARY_ROOT", os.getcwd())).expanduser().resolve()
_RAW_OBSIDIAN_VAULT = os.environ.get("BOOK_LIBRARY_OBSIDIAN_VAULT")
OBSIDIAN_VAULT = (
    None
    if not _RAW_OBSIDIAN_VAULT
    else Path(_RAW_OBSIDIAN_VAULT).expanduser().absolute()
)
library_tools = build_tools(ROOT, vault_root=OBSIDIAN_VAULT)
mcp = FastMCP("local-book-library")
```

- [ ] **Step 5: Add the active vault to the real MCP configuration**

Set the `env` line in `.codex/config.toml` to:

```toml
env = { BOOK_LIBRARY_ROOT = "/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo", BOOK_LIBRARY_OBSIDIAN_VAULT = "/Users/zhaoyunfei/Documents/Obsidian_workspace", HF_HUB_OFFLINE = "1", TRANSFORMERS_OFFLINE = "1" }
```

- [ ] **Step 6: Run the MCP and policy tests**

Run:

```bash
uv run pytest tests/test_mcp_server.py tests/test_project_policy.py -v
```

Expected: all selected tests pass and subprocess import remains silent on stdout.

- [ ] **Step 7: Commit MCP wiring**

```bash
git add book_agent/mcp_server.py .codex/config.toml tests/test_mcp_server.py tests/test_project_policy.py
git commit -m "feat: connect MCP to active Obsidian vault"
```

## Task 6: Update user guidance and evidence-policy wording

**Files:**
- Modify: `tests/test_user_guide.py:143-174`
- Modify: `docs/USER_GUIDE.md:102-110,122-134`
- Modify: `outputs/书库RAG快速开始.md:39-53`
- Modify: `AGENTS.md:13`

- [ ] **Step 1: Write failing guide assertions for the current-vault workflow**

Replace the Obsidian-specific assertions at the end of `tests/test_user_guide.py` with:

```python
def test_user_guide_sets_privacy_token_and_current_obsidian_boundaries() -> None:
    guide = _guide()

    for phrase in (
        "完整书籍",
        "索引",
        "向量",
        "保留在本机",
        "选中少量段落",
        "Codex 上下文",
        "当前 Obsidian 仓库",
        "Obsidian_workspace",
        "data/",
    ):
        assert phrase in guide
    assert "作为 Obsidian vault" not in guide
    assert "Open folder as vault" not in guide


def test_user_guide_uses_current_vault_paths_and_readable_spacing() -> None:
    guide = _guide()

    for path in (
        "书库/30-AI读书笔记/",
        "书库/10-原始书籍/",
        "书库/20-解析文本/",
    ):
        assert f"`{path}`" in guide

    assert "/Users/zhaoyunfei/Documents/Obsidian_workspace" in guide
    assert "`vault/书库/" not in guide
    assert "Codex依据" not in guide
    assert "Codex 依据" in guide
```

- [ ] **Step 2: Run the guide tests and verify they expose the obsolete instructions**

Run:

```bash
uv run pytest tests/test_user_guide.py -v
```

Expected: the two changed tests fail because the guide still instructs users to open the project `vault/`.

- [ ] **Step 3: Replace the Obsidian browsing section in the full guide**

Use this text in `docs/USER_GUIDE.md` under “保存到 Obsidian”:

```markdown
笔记会写入当前 Obsidian 仓库的 `书库/30-AI读书笔记/`，并标明它是 AI 生成内容。AI 笔记方便阅读，但不是原始证据；后续引用书籍时，Codex 仍应回到原书段落核对。

本项目已经连接到当前 Obsidian 仓库 `/Users/zhaoyunfei/Documents/Obsidian_workspace`。不需要再选择 “Open folder as vault”，也不需要切换仓库；`书库` 会直接出现在当前左侧栏。原书位于 `书库/10-原始书籍/`，解析文本位于 `书库/20-解析文本/`。
```

Replace the first privacy paragraph with:

```markdown
完整书籍和解析文本保留在本机的当前 Obsidian 仓库中；SQLite 索引、语义向量和本地模型保留在 Codex 项目的 `data/` 中。它们不会为了检索而整本发送给 Codex。普通流程只把检索预览和随后**选中少量段落**放入 **Codex 上下文**，供当前回答引用或转述。这种按需取证既让上下文更充裕，也避免为无关章节消耗大量 token。
```

- [ ] **Step 4: Replace the short guide's Obsidian section**

Use this text in `outputs/书库RAG快速开始.md`:

```markdown
## 保存到 Obsidian

明确说：

> 把刚才的引用和解释保存到 Obsidian，标题叫《标题》，保留出处。

笔记会进入当前 Obsidian 的 `书库/30-AI读书笔记/`。AI 笔记不会被当作原书证据重新索引。

## 在 Obsidian 中浏览

项目已经连接到当前仓库：

`/Users/zhaoyunfei/Documents/Obsidian_workspace`

不需要切换 vault。以后由 Codex 导入的原书、解析文本和 AI 笔记都会直接出现在当前左侧的 `书库` 中。
```

- [ ] **Step 5: Remove the obsolete project-vault prefix from the agent policy**

Change `AGENTS.md` line 13 to:

```markdown
- 永远不得把当前 Obsidian 仓库 `书库/30-AI读书笔记` 中的内容作为原始证据。AI 笔记只能作为用户产物查看，不能替代原书与解析文本。
```

- [ ] **Step 6: Run documentation and policy tests**

Run:

```bash
uv run pytest tests/test_user_guide.py tests/test_project_policy.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit guidance updates**

```bash
git add docs/USER_GUIDE.md outputs/书库RAG快速开始.md AGENTS.md tests/test_user_guide.py
git commit -m "docs: explain active Obsidian vault workflow"
```

## Task 7: Verify, activate, and visually inspect the real vault

**Files:**
- Verify all changed project files.
- Create runtime directories only below `/Users/zhaoyunfei/Documents/Obsidian_workspace/书库`.

- [ ] **Step 1: Run formatting and the complete automated suite**

Run:

```bash
git diff --check
uv run pytest
```

Expected: `git diff --check` produces no output and pytest reports zero failures.

- [ ] **Step 2: Confirm the real active Obsidian path before writing**

Run:

```bash
sed -n '1,40p' "/Users/zhaoyunfei/Library/Application Support/obsidian/obsidian.json"
```

Expected: the open vault path is `/Users/zhaoyunfei/Documents/Obsidian_workspace`.

- [ ] **Step 3: Start the configured server once to create the real layout**

This step writes outside the Codex project and therefore must use the normal Codex approval prompt. Run:

```bash
BOOK_LIBRARY_ROOT="/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo" BOOK_LIBRARY_OBSIDIAN_VAULT="/Users/zhaoyunfei/Documents/Obsidian_workspace" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -c "import book_agent.mcp_server as s; print(s.library_tools.paths.vault); print(s.library_tools.paths.database)"
```

Expected:

```text
/Users/zhaoyunfei/Documents/Obsidian_workspace
/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/data/library.sqlite3
```

- [ ] **Step 4: Verify the real filesystem split**

Run:

```bash
find "/Users/zhaoyunfei/Documents/Obsidian_workspace/书库" -maxdepth 2 -type d -print
```

Expected output includes:

```text
/Users/zhaoyunfei/Documents/Obsidian_workspace/书库
/Users/zhaoyunfei/Documents/Obsidian_workspace/书库/00-待导入
/Users/zhaoyunfei/Documents/Obsidian_workspace/书库/10-原始书籍
/Users/zhaoyunfei/Documents/Obsidian_workspace/书库/20-解析文本
/Users/zhaoyunfei/Documents/Obsidian_workspace/书库/30-AI读书笔记
```

- [ ] **Step 5: Verify the Obsidian UI**

Use the `computer-use` skill to inspect the already-open Obsidian window. Confirm that the current vault remains `Obsidian_workspace` and that a top-level `书库` folder is visible in the left file explorer. Do not switch vaults and do not create files through the UI.

- [ ] **Step 6: Run a final repository check**

Run:

```bash
git status --short --branch
git log -8 --oneline
```

Expected: the branch is `main`, the worktree is clean, and the six implementation commits from Tasks 1–6 appear above the approved design and plan commits.

- [ ] **Step 7: Report the one-time reload requirement**

Tell the user that the real folder now exists and is visible. Explain that Codex must be reloaded once so the current project task picks up the changed MCP environment; after that, uploading, querying, quoting, paraphrasing, and saving remain entirely inside Codex.
