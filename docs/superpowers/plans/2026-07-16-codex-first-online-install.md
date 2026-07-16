# Codex-First Online Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an ordinary Git clone self-installing through Codex, including Python, locked dependencies, the fixed E5 semantic model, RapidOCR models, Apple Vision OCR, generated MCP configuration, and a final health check.

**Architecture:** Git carries the source, Codex instructions, pinned arm64 `uv`, and the small arm64 Vision helper. `install-macos.command` always uses project-local uv to obtain Python 3.12 and run the Python installer; the installer syncs the lock, downloads and verifies the fixed semantic-model snapshot, prepares RapidOCR, validates Vision, runs self-tests, and only then publishes `.codex/config.toml`.

**Tech Stack:** Bash, Python 3.12, uv 0.11.26, pytest, Hugging Face Hub, Sentence Transformers, RapidOCR, Apple Vision, MCP stdio, Git/GitHub.

---

## Execution prerequisites

- Execute in a new isolated worktree from commit `8d26781` on branch `codex/codex-first-online-install`; do not reuse the paused `codex/offline-macos-v0.3` worktree.
- Preserve the original checkout's unrelated `AGENTS.md`, `photography-aesthetics-timeline.html`, and `scripts/vision_ocr_pdf.swift` changes. Reconcile the install section with the user's `AGENTS.md` edit only at final integration.
- Follow test-driven development for every behavior change and commit after each task.
- Do not implement the discontinued offline wheelhouse/Python-mirror/ZIP plan.
- Use the existing local assets only as pinned Git inputs:
  - `<UV_BINARY>`, SHA-256 `c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554`;
  - `<PROJECT_ROOT>/bin/book-vision-ocr`, existing arm64 helper validated by the current installer tests.

## File responsibility map

- `AGENTS.md`: tells Codex when and how to run the single installer and how to verify after restart.
- `install-macos.command`: platform gate plus project-local uv/Python bootstrap; no model or MCP logic.
- `install-from-github.command`: compatibility wrapper only; delegates to `install-macos.command` and no longer downloads the obsolete v0.2 ZIP.
- `installer/model_assets.py`: fixed E5 download, manifest validation, and reuse.
- `installer/runtime_selftest.py`: dependency, model, OCR, and MCP smoke checks.
- `installer/install_macos.py`: orders sync, asset preparation, self-test, and config publication.
- `README.md`, `docs/安装说明.md`: the four user actions: clone, open in Codex and ask to install, restart, check status.

### Task 1: Make the Git clone self-describing and complete

**Files:**
- Modify: `.gitignore`
- Delete from Git: `.codex/config.toml`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Create: `tests/test_codex_install_contract.py`
- Add binary: `bin/uv`
- Add binary: `bin/book-vision-ocr`

- [ ] **Step 1: Write the failing Git/Codex contract tests**

Create `tests/test_codex_install_contract.py`:

```python
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UV_SHA256 = "c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554"


def _tracked(path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def test_git_clone_contains_both_bootstrap_executables() -> None:
    uv = ROOT / "bin" / "uv"
    helper = ROOT / "bin" / "book-vision-ocr"
    assert _tracked("bin/uv")
    assert _tracked("bin/book-vision-ocr")
    assert uv.stat().st_mode & 0o111
    assert helper.stat().st_mode & 0o111
    assert hashlib.sha256(uv.read_bytes()).hexdigest() == UV_SHA256
    assert helper.read_bytes()[:4] in {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"}


def test_machine_specific_codex_config_is_generated_not_tracked() -> None:
    assert not _tracked(".codex/config.toml")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/.codex/config.toml" in ignore
    assert "/Obsidian书库/" in ignore


def test_agents_gives_codex_one_install_route() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    required = (
        "首次安装与修复",
        "./install-macos.command",
        "完整退出并重启 Codex",
        "library_status",
        "不要自行拼接另一套 Python、pip、uv 或模型下载命令",
    )
    assert all(item in text for item in required)


def test_readme_has_the_four_step_codex_flow() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    ordered = (
        "git clone https://github.com/zyf2238070278-glitch/codex-obsidian-book-library.git",
        "请安装并检查这个书库",
        "完整退出并重启 Codex",
        "检查书库状态",
    )
    positions = [text.index(item) for item in ordered]
    assert positions == sorted(positions)
```

- [ ] **Step 2: Run the tests and verify the expected failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_codex_install_contract.py -v
```

Expected: failures because `bin/uv` and the helper are not Git-tracked and the new Codex install wording is absent.

- [ ] **Step 3: Track only the two pinned executables**

Change the binary section of `.gitignore` to:

```gitignore
/bin/*
!/bin/uv
!/bin/book-vision-ocr
/.codex/config.toml
/Obsidian书库/
```

Copy the already verified uv binary without changing its bytes, retain mode `0755`, and add both binaries:

```bash
UV_BINARY="${UV_BINARY:?Set UV_BINARY to the verified uv executable first}"
cp "$UV_BINARY" bin/uv
chmod 0755 bin/uv bin/book-vision-ocr
/usr/bin/shasum -a 256 bin/uv
git add -f bin/uv bin/book-vision-ocr
```

Expected uv SHA-256: `c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554`.

Remove the currently tracked `.codex/config.toml` from Git because it contains
the publishing machine's absolute paths. The installer recreates it after clone:

```bash
git rm --cached .codex/config.toml
```

- [ ] **Step 4: Add the Codex install section and short README flow**

Add this section before the existing book-evidence rules in `AGENTS.md`:

```markdown
## 首次安装与修复

- 用户要求安装、初始化、修复环境，或在刚克隆后要求检查书库时，只运行项目根目录的 `./install-macos.command`。
- 不要自行拼接另一套 Python、pip、uv 或模型下载命令；安装入口负责固定版本、目录和校验。
- 安装失败时读取真实退出码和错误输出，修复明确问题后重试同一入口，不得假装完成。
- 安装成功后明确要求用户完整退出并重启 Codex，再重新打开并信任整个项目。
- 重启后的最终确认必须调用 `library_status`；脚本退出 0 不等于当前任务已经加载 MCP。
```

Replace the README installation opening with these exact user actions, followed by the existing usage/privacy explanation:

```markdown
## Git 安装（推荐）

1. 在终端运行：

   ```bash
   git clone https://github.com/zyf2238070278-glitch/codex-obsidian-book-library.git
   ```

2. 在 Codex 中打开并信任整个 `codex-obsidian-book-library` 目录，新建任务并说：“请安装并检查这个书库”。
3. 等安装器显示成功后，完整退出并重启 Codex，再打开同一项目。
4. 新建任务并说：“检查书库状态”。正常后即可导入书籍和按需启动 OCR。

Codex 会运行项目唯一安装入口，联网准备项目专用 Python、锁定依赖、语义模型和 OCR 模型。用户不需要手动配置 Python、uv、pip、MCP 或模型路径。
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_codex_install_contract.py tests/test_project_policy.py -q
git diff --check
```

Expected: all selected tests pass.

Commit:

```bash
git add .gitignore AGENTS.md README.md tests/test_codex_install_contract.py bin/uv bin/book-vision-ocr
git commit -m "build: make git clone self-installing"
```

### Task 2: Download and verify the fixed semantic model

**Files:**
- Create: `installer/model_assets.py`
- Create: `tests/installer/test_model_assets.py`
- Modify: `installer/install_macos.py`
- Modify: `tests/installer/test_install_macos.py`

- [ ] **Step 1: Write failing model reuse/download/tamper tests**

Create fixture manifests with small files and test this public interface:

```python
import hashlib
import json
from pathlib import Path

import pytest

from installer import model_assets


REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
FIXTURE_BYTES = b'{"model":"fixture"}\n'


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def populate_model_fixture(snapshot: Path) -> None:
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_bytes(FIXTURE_BYTES)


def make_model_fixture(
    tmp_path: Path, *, create_files: bool
) -> tuple[Path, Path, Path]:
    model_root = tmp_path / "models"
    snapshot = (
        model_root
        / "models--intfloat--multilingual-e5-small"
        / "snapshots"
        / REVISION
    )
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_id": "intfloat/multilingual-e5-small",
                "model_revision": REVISION,
                "files": [{
                    "path": "config.json",
                    "size": len(FIXTURE_BYTES),
                    "sha256": _digest(FIXTURE_BYTES),
                }],
            }
        ),
        encoding="utf-8",
    )
    if create_files:
        populate_model_fixture(snapshot)
    return model_root, manifest, snapshot


def mutate_fixture(snapshot: Path, manifest: Path, mutation: str) -> None:
    target = snapshot / "config.json"
    if mutation == "missing":
        target.unlink()
    elif mutation == "size":
        target.write_bytes(FIXTURE_BYTES + b"x")
    elif mutation == "hash":
        target.write_bytes(b'x' * len(FIXTURE_BYTES))
    elif mutation == "escape":
        target.unlink()
        outside = snapshot.parents[2].parent / "outside.json"
        outside.write_bytes(FIXTURE_BYTES)
        target.symlink_to(outside)
    else:
        raise AssertionError(mutation)


def test_ensure_model_downloads_exact_revision_and_validates(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=False)

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        populate_model_fixture(snapshot)
        return str(snapshot)

    result = model_assets.ensure_model(
        model_root=model_root,
        manifest_path=manifest,
        snapshot_download=download,
    )

    assert result == snapshot
    assert calls == [{
        "repo_id": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "cache_dir": str(model_root),
    }]


def test_ensure_model_reuses_a_valid_snapshot_without_network(tmp_path: Path) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    result = model_assets.ensure_model(
        model_root=model_root,
        manifest_path=manifest,
        snapshot_download=lambda **_: (_ for _ in ()).throw(AssertionError("network")),
    )
    assert result == snapshot


@pytest.mark.parametrize("mutation", ["missing", "size", "hash", "escape"])
def test_validate_model_rejects_corruption(tmp_path: Path, mutation: str) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    mutate_fixture(snapshot, manifest, mutation)
    with pytest.raises(model_assets.ModelAssetError):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)
```

Fixture helpers create the Hugging Face cache layout
`models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3` and a manifest with exact `path`, `size`, and lowercase SHA-256 records.

- [ ] **Step 2: Run and observe the missing-module error**

Run:

```bash
.venv/bin/python -m pytest tests/installer/test_model_assets.py -v
```

Expected: collection ERROR because `installer.model_assets` does not exist.

- [ ] **Step 3: Implement the focused model asset module**

Create `installer/model_assets.py` with these public constants and functions:

```python
MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class ModelAssetError(RuntimeError):
    pass


def validate_model(*, model_root: Path, manifest_path: Path) -> Path:
    """Return the fixed valid snapshot or raise ModelAssetError."""


def ensure_model(
    *,
    model_root: Path,
    manifest_path: Path,
    snapshot_download: Callable[..., str] | None = None,
) -> Path:
    """Reuse a valid snapshot or download the fixed revision and validate it."""
```

Implementation rules:

- Parse JSON strictly: top-level keys exactly `model_id`, `model_revision`, `files`; each record exactly `path`, `size`, `sha256`.
- Require the fixed ID/revision, a non-empty file list, safe NFC POSIX relative paths, positive integer sizes, and lowercase 64-character hashes.
- The expected snapshot is exactly `model_root/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- Permit Hugging Face's relative snapshot symlinks only when their resolved target remains beneath `model_root`; reject absolute or escaping links and special files.
- Stream size/hash validation for every manifest entry and reject missing or extra regular snapshot files other than `.cache` metadata.
- `ensure_model` first calls `validate_model`; only a validation failure triggers `huggingface_hub.snapshot_download` with the exact kwargs shown in the test.
- Require the returned path to resolve to the expected fixed snapshot and validate again. Convert download/IO/JSON failures to `ModelAssetError` with concise Chinese context.

- [ ] **Step 4: Integrate model preparation before config publication**

In `installer/install_macos.py`, import `ensure_model` and, after `_sync_environment` has produced `.venv` but before RapidOCR/config publication, load the downloader from the new environment using a subprocess module entry:

```python
def _prepare_semantic_model(
    *, project_root: Path, python: Path, run_command: Callable[..., Any]
) -> None:
    command = [
        str(python),
        "-m",
        "installer.model_assets",
        "--model-root",
        str(project_root / "data" / "models"),
        "--manifest",
        str(project_root / "distribution" / "model-manifest.json"),
    ]
    try:
        run_command(command, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise InstallError(
            f"语义模型下载或校验失败（退出码 {exc.returncode}）。请检查网络后重试。"
        ) from exc
```

Add a CLI to `model_assets.py` that calls `ensure_model` and returns `0` only after validation. Extend installer tests so the recorded order is `uv sync`, model module, RapidOCR copy, Vision validation, then config write; a model command failure must leave an existing config byte-for-byte unchanged.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest \
  tests/installer/test_model_assets.py tests/installer/test_install_macos.py -q
```

Expected: all selected tests pass.

Commit:

```bash
git add installer/model_assets.py installer/install_macos.py \
  tests/installer/test_model_assets.py tests/installer/test_install_macos.py
git commit -m "feat: install fixed semantic model online"
```

### Task 3: Make the launcher deterministic and add final self-tests

**Files:**
- Modify: `install-macos.command`
- Modify: `install-from-github.command`
- Modify: `tests/installer/test_install_macos_launcher.py`
- Modify: `tests/installer/test_install_from_github_launcher.py`
- Create: `installer/runtime_selftest.py`
- Create: `tests/installer/test_runtime_selftest.py`
- Modify: `installer/install_macos.py`

- [ ] **Step 1: Write failing launcher contract tests**

Replace the legacy “prefer system python3” expectation with:

```python
def test_launcher_always_uses_project_uv_and_fixed_python(tmp_path: Path) -> None:
    launcher = _copy_launcher(tmp_path)
    release = launcher.parent
    capture = tmp_path / "uv arguments.txt"
    _write_executable(release / "bin" / "uv", _capture_script())
    completed = subprocess.run(
        [str(launcher), "--vault", str(tmp_path / "Vault With Spaces")],
        text=True,
        capture_output=True,
        env={**os.environ, "CAPTURE_FILE": str(capture)},
        check=False,
    )
    assert completed.returncode == 0
    assert _read_arguments(capture) == [
        "run", "--no-project", "--python", "3.12",
        str(release / "installer" / "install_macos.py"),
        "--project-root", str(release),
        "--vault", str(tmp_path / "Vault With Spaces"),
    ]


def test_launcher_rejects_non_arm64_and_old_macos_in_source_contract() -> None:
    text = SOURCE_LAUNCHER.read_text(encoding="utf-8")
    assert "/usr/bin/uname -s" in text
    assert "/usr/bin/uname -m" in text
    assert "/usr/bin/sw_vers -productVersion" in text
    assert "arm64" in text
    assert "macOS 16" in text
```

Change `tests/installer/test_install_from_github_launcher.py` so the compatibility command is required to execute only the local `install-macos.command`; reject any `TAG=`, GitHub Release URL, ZIP download, `curl`, or `ditto` text.

- [ ] **Step 2: Run launcher tests and observe legacy failures**

Run:

```bash
.venv/bin/python -m pytest \
  tests/installer/test_install_macos_launcher.py \
  tests/installer/test_install_from_github_launcher.py -q
```

Expected: failures because the launcher prefers system Python and the compatibility script still downloads v0.2.

- [ ] **Step 3: Simplify both shell entry points**

Change `install-macos.command` to:

- use `#!/bin/bash`, `set -u`, physical `PROJECT_ROOT`, and absolute macOS base utilities;
- require `/usr/bin/uname -s` equal `Darwin`, `/usr/bin/uname -m` equal `arm64`, and numeric major from `/usr/bin/sw_vers -productVersion` at least `16`;
- require executable `bin/uv` and verify its bytes with `/usr/bin/shasum -a 256` against `c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554` before execution;
- never choose system `python3`, PATH `uv`, pip, Homebrew, Xcode, or Command Line Tools;
- run the exact project uv argv asserted in Step 1 and preserve its exit status;
- retain the interactive-only “按回车关闭” behavior.

Replace `install-from-github.command` with a compatibility wrapper:

```bash
#!/bin/bash
set -u
case "$0" in */*) script_directory=${0%/*} ;; *) script_directory=. ;; esac
PROJECT_ROOT="$(CDPATH= cd -- "$script_directory" && pwd -P)" || exit 1
exec "$PROJECT_ROOT/install-macos.command" "$@"
```

- [ ] **Step 4: Write failing runtime self-test tests**

Create `tests/installer/test_runtime_selftest.py` around this interface:

```python
from installer.runtime_selftest import SelfTestError, run_selftest


def test_selftest_requires_embedding_ocr_vision_and_mcp(tmp_path: Path) -> None:
    events: list[str] = []
    result = run_selftest(
        project_root=tmp_path,
        import_probe=lambda: events.append("imports"),
        embedding_probe=lambda root: (events.append("embedding") or 384),
        rapidocr_probe=lambda root: events.append("rapidocr"),
        vision_probe=lambda helper: events.append("vision"),
        mcp_probe=lambda root: events.append("mcp"),
    )
    assert result.embedding_dimensions == 384
    assert events == ["imports", "embedding", "rapidocr", "vision", "mcp"]


def test_selftest_rejects_wrong_embedding_dimensions(tmp_path: Path) -> None:
    with pytest.raises(SelfTestError, match="384"):
        run_selftest(
            project_root=tmp_path,
            import_probe=lambda: None,
            embedding_probe=lambda _: 12,
            rapidocr_probe=lambda _: None,
            vision_probe=lambda _: None,
            mcp_probe=lambda _: None,
        )
```

- [ ] **Step 5: Implement and integrate the self-test**

Create `installer/runtime_selftest.py` with:

```python
@dataclass(frozen=True)
class SelfTestResult:
    embedding_dimensions: int


class SelfTestError(RuntimeError):
    pass


def run_selftest(
    *,
    project_root: Path,
    import_probe: Callable[[], None] = _probe_imports,
    embedding_probe: Callable[[Path], int] = _probe_embedding,
    rapidocr_probe: Callable[[Path], None] = _probe_rapidocr,
    vision_probe: Callable[[Path], None] = _probe_vision,
    mcp_probe: Callable[[Path], None] = _probe_mcp,
) -> SelfTestResult:
    try:
        import_probe()
        dimensions = embedding_probe(project_root / "data" / "models")
        if dimensions != 384:
            raise SelfTestError(
                f"语义模型维度错误：预期 384，实际 {dimensions}"
            )
        rapidocr_probe(project_root / "data" / "ocr-models" / "rapidocr")
        vision_probe(project_root / "bin" / "book-vision-ocr")
        mcp_probe(project_root)
    except SelfTestError:
        raise
    except Exception as exc:
        raise SelfTestError(f"安装自检失败：{exc}") from exc
    return SelfTestResult(embedding_dimensions=dimensions)
```

The concrete probes must:

- import `fitz`, `ebooklib`, `mcp`, `numpy`, `onnxruntime`, `rapidocr`, `sentence_transformers`;
- instantiate `E5EmbeddingProvider(project_root / "data/models")` with offline environment variables, require availability, embed `"安装自检"`, and return vector dimension `384`;
- require the three `book_agent.ocr.rapid.REQUIRED_MODEL_FILES` under `data/ocr-models/rapidocr`;
- execute `bin/book-vision-ocr --capabilities` and require schema 2 plus `zh-Hans` and `en-US`;
- use `mcp.client.stdio.stdio_client` with `StdioServerParameters(command=sys.executable, args=["-m", "book_agent.mcp_server"], cwd=str(project_root), env={...})`, create a real `ClientSession`, await `initialize()` and `list_tools()`, and require `library_status` in the returned tool names; run the async probe through `anyio.run` so the library owns bounded child cleanup.

Expose `python -m installer.runtime_selftest --project-root PATH`, print one concise success line, and exit nonzero on `SelfTestError`.

In `installer/install_macos.py`, run this module with the final `.venv/bin/python` after model/RapidOCR/Vision preparation and before `_write_text_atomically`. A self-test failure must preserve any previous config and return a Chinese `InstallError`.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest \
  tests/installer/test_install_macos_launcher.py \
  tests/installer/test_install_from_github_launcher.py \
  tests/installer/test_runtime_selftest.py \
  tests/installer/test_install_macos.py -q
```

Expected: all selected tests pass.

Commit:

```bash
git add install-macos.command install-from-github.command installer/install_macos.py \
  installer/runtime_selftest.py tests/installer/test_install_macos_launcher.py \
  tests/installer/test_install_from_github_launcher.py \
  tests/installer/test_runtime_selftest.py tests/installer/test_install_macos.py
git commit -m "feat: verify codex online installation"
```

### Task 4: Align docs and prove a fresh Git clone installs

**Files:**
- Modify: `docs/安装说明.md`
- Modify: `docs/USER_GUIDE.md`
- Modify: `docs/常见问题.md`
- Modify: `tests/test_release_docs.py`
- Modify: `tests/test_user_guide.py`

- [ ] **Step 1: Write failing documentation contract tests**

Require the public docs to contain, in order:

```python
CODEX_FIRST_FLOW = (
    "git clone https://github.com/zyf2238070278-glitch/codex-obsidian-book-library.git",
    "请安装并检查这个书库",
    "完整退出并重启 Codex",
    "检查书库状态",
)


@pytest.mark.parametrize("path", PUBLIC_INSTALL_DOCS)
def test_public_install_docs_use_codex_first_flow(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    positions = [text.index(item) for item in CODEX_FIRST_FLOW]
    assert positions == sorted(positions)
    for stale in (
        "v0.2.0-beta.1",
        "约 292 MB",
        "下载全量 ZIP",
        "uv sync --extra dev --extra semantic",
        "/Users/" + "zhaoyunfei/",
    ):
        assert stale not in text
```

Also require the docs to state that setup is online, versions are lock-pinned, runtime data stays under the clone, OCR remains local, and a rerun repairs/moves configuration without deleting books or notes.

- [ ] **Step 2: Run docs tests and observe stale-flow failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_release_docs.py tests/test_user_guide.py -q
```

Expected: failures for v0.2/ZIP/manual uv text and the publishing user's absolute path.

- [ ] **Step 3: Rewrite only installation/recovery sections**

Update the three docs to use the exact four-step Codex-first flow. Keep the existing book import, evidence, OCR authorization, privacy, and note-saving rules. Explicitly state:

- setup downloads about 500 MB for the semantic model plus Python packages and needs stable internet;
- the project uses its own Python/uv environment and does not modify system Python;
- Apple Vision and RapidOCR execute locally after setup;
- moving the clone requires rerunning “请安装并检查这个书库” and one Codex restart;
- deleting the clone can delete its default Vault, so books/notes must be backed up first;
- no example contains a publishing-machine absolute path.

- [ ] **Step 4: Run the full automated suite**

Run:

```bash
.venv/bin/python -m pytest -m 'not macos_vision' -q
```

Expected: all selected tests pass with zero failures.

- [ ] **Step 5: Perform the fresh-clone online acceptance test**

From outside the implementation worktree:

```bash
TEST_ROOT="$(/usr/bin/mktemp -d /private/tmp/codex-book-git-install.XXXXXX)"
SOURCE_WORKTREE="${SOURCE_WORKTREE:?Set SOURCE_WORKTREE to the implementation worktree first}"
git clone --no-local \
  "$SOURCE_WORKTREE" \
  "$TEST_ROOT/codex-obsidian-book-library"
cd "$TEST_ROOT/codex-obsidian-book-library"
./install-macos.command
./.venv/bin/python -m installer.runtime_selftest --project-root "$PWD"
git status --short
```

Expected:

- installation succeeds using only tracked pre-bootstrap files plus network downloads;
- the self-test reports imports, 384-dimensional E5, RapidOCR, Vision, and MCP success;
- `git status --short` is empty because `.venv`, `data`, generated `.codex/config.toml`, and the default Vault are ignored runtime files;
- no user books or publishing-machine absolute path appears in the clone.

Identity-check the `/private/tmp/codex-book-git-install.*` directory before removing only that disposable test tree.

- [ ] **Step 6: Commit documentation and prepare publication**

Run:

```bash
git diff --check
git status --short
git add docs/安装说明.md docs/USER_GUIDE.md docs/常见问题.md \
  tests/test_release_docs.py tests/test_user_guide.py
git commit -m "docs: explain codex-first git installation"
```

After the final code review, use the `finishing-a-development-branch` skill to integrate the branch, reconcile only the user's current `AGENTS.md` change, push the verified latest commit to GitHub, and provide the clone command. Do not publish if the fresh-clone installation or runtime self-test fails.
