# Offline All-in-One macOS Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a transferable Apple Silicon ZIP that installs CPython 3.12 and every locked runtime dependency without network access or preinstalled developer tools.

**Non-negotiable install gate:** The recipient installs no Homebrew, Python, uv,
pip, compiler, Xcode, Command Line Tools, model, OCR runtime, or package manager.
After extraction, the only project action is one run of `install-macos.command`;
any prompt or failure that asks for an extra technical dependency means the
candidate is not deliverable.

**Architecture:** Keep the existing `v0.2.0-beta.1` online-bootstrap release intact while adding an offline `v0.3.0-beta.1` contract. The ZIP carries one pinned python-build-standalone archive plus a hash-locked macOS arm64 wheelhouse; the launcher keeps an isolated bootstrap runtime alive until its child exits, while the installer clones a separate publishable runtime, repairs uv's remaining absolute venv links, validates the paired runtime/venv inside a transaction, and then atomically publishes them with the generated Codex config.

**Tech Stack:** Python 3.12, uv 0.11.26, Bash, pytest, ZIP/tar/wheel metadata, Mach-O inspection (`file`, `lipo`, `otool`, `vtool`, `codesign`), macOS `sandbox-exec`, MCP, Sentence Transformers, RapidOCR, Apple Vision.

---

## Execution prerequisites

- Execute this plan in an isolated worktree created with the `using-git-worktrees` skill from commit `67c6897` or a descendant containing the approved design.
- Treat the separate `.bootstrap-runtime` and venv sealing steps below as the implementation correction to the design's single-runtime shorthand: a running PBS interpreter cannot have its own stdlib tree moved, and uv 0.11.26 leaves the base venv link absolute even with `--relocatable`.
- Preserve the user's unrelated `AGENTS.md`, `photography-aesthetics-timeline.html`, and `scripts/vision_ocr_pdf.swift` changes in the original checkout.
- Follow test-driven development for every behavior change: add one failing test, run it and observe the expected failure, implement only that behavior, then rerun the focused and related suites.
- Keep downloaded Python/wheel assets under ignored `dist/offline-assets/`; commit only source, tests, manifests, requirements, notices, and documentation.
- Reuse the existing ignored model snapshot read-only from `<PROJECT_ROOT>/data/models/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3`; the isolated worktree will not contain ignored model cache files, so verify this source against `distribution/model-manifest.json` before the final build.
- Do not change `install-from-github.command` to the new version until the offline ZIP has been uploaded and its public URL has been verified. This plan produces a local release candidate only.

## File responsibility map

- `distribution/offline-release.json`: immutable offline release/version/platform contract.
- `distribution/python-manifest.json`: pinned Python archive path, size, and SHA-256.
- `distribution/wheelhouse-manifest.json`: exact wheel filename, distribution, version, size, SHA-256, and wheel tags.
- `distribution/requirements-macos-arm64-py312.txt`: hash-locked install input for the wheelhouse.
- `distribution/rapidocr-model-manifest.json`: exact three ONNX member paths, sizes, and SHA-256 values extracted from the pinned RapidOCR wheel.
- `scripts/offline_assets.py`: strict manifest loading, asset-tree validation, hashing, and safe copying for release builds.
- `scripts/prepare_offline_assets.py`: online build-machine preparation of the Python archive, deterministic `antlr4` wheel, wheelhouse, requirements, and license inventory.
- `scripts/macho_audit.py`: safe archive-member inspection plus reusable Apple Silicon architecture, deployment-target, dependency, and signing-status audit.
- `scripts/build_macos_release.py`: assemble and verify either the legacy release or the new offline payload.
- `installer/offline_runtime.py`: construct offline uv commands, validate a staged runtime, publish/rollback `.runtime` and `.venv`, and persist an installation fingerprint.
- `installer/runtime_selftest.py`: run imports, E5, RapidOCR, Vision/MCP smoke checks inside the staged environment.
- `installer/install_macos.py`: orchestrate environment installation, Vault/runtime directory creation, OCR model publication, and config writing.
- `install-macos.command`: no-system-Python bootstrap using bundled uv and the local Python mirror.
- `scripts/verify_offline_release.py`: repeatable current-Mac hard-network-denied installation and smoke-test gate.
- `tests/test_offline_assets.py`, `tests/test_prepare_offline_assets.py`, `tests/test_build_macos_release.py`: build-side tests.
- `tests/installer/test_install_macos_launcher.py`, `tests/installer/test_offline_runtime.py`, `tests/installer/test_runtime_selftest.py`, `tests/installer/test_install_macos.py`: target-side tests.
- `tests/test_verify_offline_release.py`, `tests/test_release_docs.py`: release-gate and public-contract tests.

### Task 1: Freeze the offline release metadata contract

**Files:**
- Create: `distribution/offline-release.json`
- Modify: `tests/test_build_macos_release.py`

- [ ] **Step 1: Write the failing metadata test**

Add this exact test without changing `distribution/release.json` or the existing `FIXED_METADATA` assertion:

```python
OFFLINE_METADATA = {
    "version": "0.3.0-beta.1",
    "tag": "v0.3.0-beta.1",
    "distribution_kind": "offline",
    "project": "codex-obsidian-book-library",
    "platform": "macos-arm64",
    "minimum_macos": "14.0",
    "model_id": "intfloat/multilingual-e5-small",
    "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "uv_version": "0.11.26",
    "uv_sha256": "c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554",
    "python_version": "3.12.11",
    "python_build": "20251007",
    "python_asset": "cpython-3.12.11+20251007-aarch64-apple-darwin-install_only_stripped.tar.gz",
    "python_sha256": "407fa242942a7ba5d91899abc562fc9897f7a0376f8d2060285e8c0560323f19",
    "archive": "codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline.zip",
    "top_level_directory": "codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline",
    "vision_helper_version": "0.2.0",
    "vision_schema_version": "2",
}


def test_offline_release_metadata_is_fully_pinned() -> None:
    path = PROJECT_ROOT / "distribution" / "offline-release.json"
    assert json.loads(path.read_text(encoding="utf-8")) == OFFLINE_METADATA
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run:

```bash
uv run pytest tests/test_build_macos_release.py::test_offline_release_metadata_is_fully_pinned -v
```

Expected: FAIL with `FileNotFoundError` for `distribution/offline-release.json`.

- [ ] **Step 3: Add the exact offline metadata file**

Create `distribution/offline-release.json` with the same keys and values as `OFFLINE_METADATA`, formatted as UTF-8 JSON with two-space indentation and one trailing newline.

- [ ] **Step 4: Verify legacy and offline metadata together**

Run:

```bash
uv run pytest \
  tests/test_build_macos_release.py::test_release_metadata_is_pinned_for_macos_arm64 \
  tests/test_build_macos_release.py::test_offline_release_metadata_is_fully_pinned -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit the metadata contract**

```bash
git add distribution/offline-release.json tests/test_build_macos_release.py
git commit -m "build: pin offline macos release metadata"
```

### Task 2: Add strict offline asset manifests

**Files:**
- Create: `scripts/offline_assets.py`
- Create: `tests/test_offline_assets.py`

- [ ] **Step 1: Write failing manifest and tamper tests**

Create fixtures using ordinary files and assert the public API below:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import offline_assets


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(path: Path, kind: str, relative: str, data: bytes) -> Path:
    payload = {
        "schema_version": 1,
        "kind": kind,
        "files": [
            {
                "path": relative,
                "size": len(data),
                "sha256": _sha(data),
                "mode": "0644",
            }
        ],
    }
    if kind == "wheelhouse":
        payload["requirements_sha256"] = _sha(b"requirements")
        payload["uv_lock_sha256"] = _sha(b"uv.lock")
        payload["target"] = {
            "python_version": "3.12.11",
            "implementation": "cp",
            "abi": "cp312",
            "platform": "macosx_14_0_arm64",
            "extras": ["ocr", "semantic"],
            "include_dev": False,
        }
        payload["files"][0].update(
            {
                "distribution": "demo",
                "version": "1.0",
                "tags": ["py3-none-any"],
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validate_asset_tree_returns_exact_trusted_records(tmp_path: Path) -> None:
    root = tmp_path / "wheelhouse"
    target = root / "demo-1.0-py3-none-any.whl"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wheel")
    manifest = _manifest(
        tmp_path / "manifest.json",
        "wheelhouse",
        "demo-1.0-py3-none-any.whl",
        b"wheel",
    )

    records = offline_assets.validate_asset_tree(
        root=root,
        manifest_path=manifest,
        expected_kind="wheelhouse",
    )

    assert records[0].path == Path("demo-1.0-py3-none-any.whl")
    assert records[0].sha256 == _sha(b"wheel")


@pytest.mark.parametrize("mutation", ["tamper", "extra", "symlink"])
def test_validate_asset_tree_rejects_untrusted_content(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "python-mirror"
    target = root / "20251007" / "runtime.tar.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"runtime")
    manifest = _manifest(
        tmp_path / "manifest.json", "python", "20251007/runtime.tar.gz", b"runtime"
    )
    if mutation == "tamper":
        target.write_bytes(b"runtimf")
    elif mutation == "extra":
        (root / "20251007" / "extra.txt").write_text("extra", encoding="utf-8")
    else:
        target.unlink()
        target.symlink_to(tmp_path / "outside")

    with pytest.raises(offline_assets.OfflineAssetError):
        offline_assets.validate_asset_tree(
            root=root,
            manifest_path=manifest,
            expected_kind="python",
        )
```

- [ ] **Step 2: Run the focused tests and observe import failure**

Run:

```bash
uv run pytest tests/test_offline_assets.py -v
```

Expected: collection ERROR because `scripts.offline_assets` does not exist.

- [ ] **Step 3: Implement the strict manifest loader and validator**

Create `scripts/offline_assets.py` with these concrete types and behaviors:

```python
from __future__ import annotations

import hashlib
import json
import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class OfflineAssetError(ValueError):
    pass


@dataclass(frozen=True)
class AssetRecord:
    path: Path
    size: int
    sha256: str
    mode: int
    distribution: str | None = None
    version: str | None = None
    tags: tuple[str, ...] = ()


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    with os.fdopen(os.dup(descriptor), "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
        info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def _safe_relative(raw: object) -> Path:
    if type(raw) is not str or not raw:
        raise OfflineAssetError("asset path must be a non-empty string")
    if (
        "\\" in raw
        or unicodedata.normalize("NFC", raw) != raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise OfflineAssetError(f"unsafe asset path: {raw}")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise OfflineAssetError(f"unsafe asset path: {raw}")
    return Path(*pure.parts)


def _load_manifest(manifest_path: Path, expected_kind: str) -> dict[str, object]:
    parent_descriptor = os.open(
        manifest_path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        descriptor = os.open(
            manifest_path.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o644:
                raise OfflineAssetError("offline manifest must be a 0644 regular file")
            with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as source:
                raw = json.load(source)
            if _stable_identity(os.fstat(descriptor)) != _stable_identity(before):
                raise OfflineAssetError("offline manifest changed while being read")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    expected_keys = {"schema_version", "kind", "files"}
    if expected_kind == "wheelhouse":
        expected_keys.update({"requirements_sha256", "uv_lock_sha256", "target"})
    if type(raw) is not dict or set(raw) != expected_keys:
        raise OfflineAssetError("offline asset manifest schema is invalid")
    if raw["schema_version"] != 1 or raw["kind"] != expected_kind:
        raise OfflineAssetError("offline asset manifest identity is invalid")
    if expected_kind == "wheelhouse":
        target = raw["target"]
        expected_target = {
            "python_version": "3.12.11",
            "implementation": "cp",
            "abi": "cp312",
            "platform": "macosx_14_0_arm64",
            "extras": ["ocr", "semantic"],
            "include_dev": False,
        }
        for key in ("requirements_sha256", "uv_lock_sha256"):
            value = raw[key]
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise OfflineAssetError(f"wheelhouse {key} is invalid")
        if target != expected_target:
            raise OfflineAssetError("wheelhouse target is invalid")
    if type(raw["files"]) is not list or not raw["files"]:
        raise OfflineAssetError("offline asset manifest must contain files")
    return raw


def _load_records(manifest_path: Path, expected_kind: str) -> tuple[AssetRecord, ...]:
    raw = _load_manifest(manifest_path, expected_kind)
    records: list[AssetRecord] = []
    seen: set[Path] = set()
    seen_portable: set[str] = set()
    for item in raw["files"]:
        record_keys = {"path", "size", "sha256", "mode"}
        if expected_kind == "wheelhouse":
            record_keys.update({"distribution", "version", "tags"})
        if type(item) is not dict or set(item) != record_keys:
            raise OfflineAssetError("offline asset record schema is invalid")
        relative = _safe_relative(item["path"])
        if relative in seen:
            raise OfflineAssetError(f"duplicate asset path: {relative}")
        portable = relative.as_posix().casefold()
        if portable in seen_portable:
            raise OfflineAssetError(f"portable asset path collision: {relative}")
        if type(item["size"]) is not int or item["size"] <= 0:
            raise OfflineAssetError(f"invalid asset size: {relative}")
        if (
            type(item["sha256"]) is not str
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
        ):
            raise OfflineAssetError(f"invalid asset SHA-256: {relative}")
        if item["mode"] != "0644":
            raise OfflineAssetError(f"invalid asset mode: {relative}")
        distribution = None
        version = None
        tags: tuple[str, ...] = ()
        if expected_kind == "wheelhouse":
            distribution = item["distribution"]
            version = item["version"]
            raw_tags = item["tags"]
            if (
                type(distribution) is not str
                or not distribution
                or type(version) is not str
                or not version
                or type(raw_tags) is not list
                or not raw_tags
                or any(type(tag) is not str or not tag for tag in raw_tags)
                or raw_tags != sorted(set(raw_tags))
            ):
                raise OfflineAssetError(f"invalid wheel metadata: {relative}")
            tags = tuple(raw_tags)
        seen.add(relative)
        seen_portable.add(portable)
        records.append(
            AssetRecord(
                relative,
                item["size"],
                item["sha256"],
                0o644,
                distribution,
                version,
                tags,
            )
        )
    return tuple(sorted(records, key=lambda record: record.path.as_posix()))


def wheelhouse_requirements_sha256(manifest_path: Path) -> str:
    raw = _load_manifest(manifest_path, "wheelhouse")
    return str(raw["requirements_sha256"])


def wheelhouse_uv_lock_sha256(manifest_path: Path) -> str:
    raw = _load_manifest(manifest_path, "wheelhouse")
    return str(raw["uv_lock_sha256"])


def validate_asset_tree(
    *, root: Path, manifest_path: Path, expected_kind: str
) -> tuple[AssetRecord, ...]:
    # `_load_records` opens the manifest with O_NOFOLLOW, checks a 0644 regular
    # file before and after reading, and rejects identity/size changes.
    records = _load_records(manifest_path, expected_kind)
    expected = {record.path for record in records}
    # `_open_real_directory` rejects a symlink root. `_walk_beneath` traverses
    # each component with dir_fd + O_NOFOLLOW, rejects links/special files and
    # portable path collisions, and returns open descriptors for 0644 files.
    with _open_real_directory(root) as root_fd:
        opened = _walk_beneath(root_fd)
        try:
            actual = set(opened)
            if actual != expected:
                raise OfflineAssetError("asset tree does not match its manifest")
            for record in records:
                descriptor, info = opened[record.path]
                if info.st_size != record.size or _sha256_fd(descriptor) != record.sha256:
                    raise OfflineAssetError(f"asset hash or size mismatch: {record.path}")
                if _stable_identity(os.fstat(descriptor)) != _stable_identity(info):
                    raise OfflineAssetError(f"asset changed during validation: {record.path}")
        finally:
            for descriptor, _info in opened.values():
                os.close(descriptor)
    return records
```

The Python manifest paths are relative to the Python mirror root, for example
`20251007/cpython-3.12.11+20251007-aarch64-apple-darwin-install_only_stripped.tar.gz`.
Wheelhouse manifest paths are wheel basenames relative to the wheelhouse root.
The release builder, not either manifest, adds the destination prefixes
`offline/python-mirror/` and `offline/wheelhouse/`. Add a focused test for the
wheelhouse requirements fingerprint and reject an invalid or unexpected
top-level manifest key. Add root/manifest symlink, backslash, control character,
non-NFC, case-fold collision, same-size race, and intermediate-directory-swap
tests for the descriptor-based helpers shown above.

- [ ] **Step 4: Run focused tests and the release-builder tests**

Run:

```bash
uv run pytest tests/test_offline_assets.py tests/test_build_macos_release.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the reusable validator**

```bash
git add scripts/offline_assets.py tests/test_offline_assets.py
git commit -m "build: validate pinned offline assets"
```

### Task 3: Prepare a complete hash-locked wheelhouse

**Files:**
- Create: `scripts/prepare_offline_assets.py`
- Create: `tests/test_prepare_offline_assets.py`
- Create during the production run: `distribution/python-manifest.json`
- Create during the production run: `distribution/wheelhouse-manifest.json`
- Create during the production run: `distribution/requirements-macos-arm64-py312.txt`
- Create during the production run: `distribution/rapidocr-model-manifest.json`
- Create during the production run: `third_party/python-dependencies/manifest.json`

- [ ] **Step 1: Write failing command-construction and manifest tests**

The tests must inject a command runner and assert these exact invariants:

```python
def test_export_command_selects_runtime_extras_without_dev(tmp_path: Path) -> None:
    command = prepare_offline_assets.export_command(
        uv=Path("/tools/uv"),
        project_root=tmp_path / "project",
        destination=tmp_path / "raw-requirements.txt",
    )
    assert command == [
        "/tools/uv",
        "export",
        "--project",
        str(tmp_path / "project"),
        "--frozen",
        "--no-dev",
        "--extra",
        "semantic",
        "--extra",
        "ocr",
        "--format",
        "requirements-txt",
        "--no-header",
        "--no-emit-project",
        "--output-file",
        str(tmp_path / "raw-requirements.txt"),
    ]


def test_wheelhouse_manifest_is_sorted_and_rejects_sdists(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "demo-1.0-py3-none-any.whl").write_bytes(b"wheel")
    manifest = prepare_offline_assets.build_wheelhouse_manifest(
        wheelhouse,
        requirements_sha256="0" * 64,
        uv_lock_sha256="1" * 64,
    )
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "wheelhouse"
    assert manifest["requirements_sha256"] == "0" * 64
    assert manifest["uv_lock_sha256"] == "1" * 64
    assert manifest["target"] == {
        "python_version": "3.12.11",
        "implementation": "cp",
        "abi": "cp312",
        "platform": "macosx_14_0_arm64",
        "extras": ["ocr", "semantic"],
        "include_dev": False,
    }
    assert manifest["files"][0]["path"] == "demo-1.0-py3-none-any.whl"
    (wheelhouse / "bad-1.0.tar.gz").write_bytes(b"sdist")
    with pytest.raises(prepare_offline_assets.AssetPreparationError, match="wheel"):
        prepare_offline_assets.build_wheelhouse_manifest(
            wheelhouse,
            requirements_sha256="0" * 64,
            uv_lock_sha256="1" * 64,
        )
```

Add tests that prove the supplied uv binary's version and SHA-256 must match offline release metadata, `SOURCE_DATE_EPOCH=315532800` is supplied when building `antlr4-python3-runtime==4.9.3`, `uv export` uses `--no-header` so neither the project nor temporary output path enters the requirements bytes, every final requirement is an exact `name==version` pin with an optional environment marker plus at least one `--hash=sha256:`, every selected artifact is a wheel, the manifest file count equals the derived target closure count, and wheel filenames are unique by normalized distribution/version. Inject a download failure while an old wheelhouse and all old contract files exist; require every old byte to remain unchanged and no temporary directory to remain.

- [ ] **Step 2: Run the focused tests and observe import failure**

```bash
uv run pytest tests/test_prepare_offline_assets.py -v
```

Expected: collection ERROR because `scripts.prepare_offline_assets` does not exist.

- [ ] **Step 3: Implement the preparation CLI**

Implement these exact stages in `prepare_assets` and expose them through argparse:

```python
def export_command(*, uv: Path, project_root: Path, destination: Path) -> list[str]:
    return [
        str(uv), "export", "--project", str(project_root), "--frozen",
        "--no-dev", "--extra", "semantic", "--extra", "ocr",
        "--format", "requirements-txt", "--no-header",
        "--no-emit-project",
        "--output-file", str(destination),
    ]


def prepare_assets(
    *,
    project_root: Path,
    uv: Path,
    python_archive: Path,
    output_dir: Path,
    contract_dir: Path,
    dependency_license_manifest: Path,
    verify_only: bool = False,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> PreparedAssets:
    """Prepare one macOS-arm64/CPython-3.12 offline dependency closure."""
```

Define the immutable return type before `prepare_assets`:

```python
@dataclass(frozen=True)
class PreparedAssets:
    python_manifest: Path
    wheelhouse_manifest: Path
    requirements: Path
    rapidocr_model_manifest: Path
    dependency_license_manifest: Path
    wheel_count: int
```

Use one strict, sorted dependency-license schema so Task 4 can validate it:

```json
{
  "schema_version": 1,
  "kind": "python-wheel-license-inventory",
  "wheels": [
    {
      "filename": "demo-1.0-py3-none-any.whl",
      "distribution": "demo",
      "version": "1.0",
      "wheel_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "license_expression": null,
      "license": "MIT",
      "license_classifiers": ["License :: OSI Approved :: MIT License"],
      "license_files": [
        {"path": "demo-1.0.dist-info/licenses/LICENSE", "sha256": "0000000000000000000000000000000000000000000000000000000000000000"}
      ]
    }
  ]
}
```

Require one inventory entry per wheel manifest record, exact matching filename,
normalized distribution, version, and wheel SHA-256, sorted unique classifiers,
and sorted safe in-wheel license paths. `license_expression` and legacy `license`
may each be either a string or JSON null; all other keys and types are exact.

Generate `distribution/rapidocr-model-manifest.json` with the exact schema below
from the already hash-validated `rapidocr` wheel:

```json
{
  "schema_version": 1,
  "kind": "rapidocr-models",
  "source_distribution": "rapidocr",
  "source_version": "3.9.1",
  "source_wheel_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "files": [
    {
      "filename": "PP-OCRv6_det_small.onnx",
      "wheel_member": "rapidocr/models/PP-OCRv6_det_small.onnx",
      "size": 1,
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  ]
}
```

The production manifest must contain exactly the three names in
`RAPIDOCR_MODEL_FILES`, positive real sizes, their real lowercase hashes, safe
unique wheel member paths, and the actual pinned RapidOCR version/wheel hash.
Tests must reject a missing, extra, renamed, duplicate, or same-size-mutated ONNX
member.

The body must perform, in order:

1. Verify the uv version/SHA-256 and Python archive filename/SHA-256 against `distribution/offline-release.json`; require the archive to reside below `output_dir/python-mirror/20251007` as the exact pinned regular file.
2. Run `uv export` using `export_command`.
3. Create a temporary seeded build venv with `uv venv --seed --python 3.12`.
4. Create the same-filesystem temporary candidate wheelhouse that will later be atomically published. Build only `antlr4-python3-runtime==4.9.3` with `SOURCE_DATE_EPOCH=315532800` and `--no-deps` directly into that candidate wheelhouse; do not use a separate disposable wheel directory.
5. Replace the exported antlr hash block with the locally built wheel hash while retaining `antlr4-python3-runtime==4.9.3`.
6. Run the build venv's `python -m pip download` with `--require-hashes`, `--only-binary=:all:`, `--platform macosx_14_0_arm64`, `--implementation cp`, `--python-version 312`, and `--abi cp312`, passing that exact candidate wheelhouse directory to both `--dest` and `--find-links` and the concrete generated requirements path to `-r`. Add an assertion that the prebuilt antlr wheel is present there before download and remains the selected artifact afterward.
7. Reject every non-`.whl` file, duplicate normalized distribution/version, missing requirement distribution, or wheel unsupported by CPython 3.12 macOS arm64.
8. Write sorted JSON manifests with size, lowercase SHA-256, mode `0644`, normalized distribution/version, and parsed wheel tags. Python record paths are relative to `output_dir/python-mirror`; wheel record paths are basenames relative to `output_dir/wheelhouse`; the wheelhouse manifest records the final requirements and `uv.lock` SHA-256 values plus the fixed target/extras object; the RapidOCR manifest records the three trusted ONNX members.
9. Extract each wheel's `.dist-info/METADATA` license fields and `.dist-info/licenses/*` names into the explicit `dependency_license_manifest` destination without unpacking wheel payloads into the repository.
10. Build the wheelhouse and all five contract files in same-filesystem temporary siblings, then atomically publish them with backups and reverse-order rollback. Do not alter an existing wheelhouse or any contract on preparation, validation, or publication failure; delete backups only after all six final paths are revalidated.

CLI arguments must be `--project-root`, `--uv`, `--python-archive`, `--output-dir`, `--contract-dir`, `--dependency-license-manifest`, and `--verify-only`. In `--verify-only` mode, do not download, build, or modify anything: run the pinned uv `export_command` into a temporary file, resolve it against the existing locally built antlr wheel, regenerate the exact target requirements, recompute the Python, wheelhouse, RapidOCR, and dependency-license manifests, and byte-compare all five contract files with their canonical UTF-8 serialization. This must also prove the current `uv.lock` hash, extras, platform, ABI, Python version, and dev-exclusion object still match the wheelhouse contract. Any difference is `AssetPreparationError`.

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/test_prepare_offline_assets.py tests/test_offline_assets.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the preparer**

```bash
git add scripts/prepare_offline_assets.py tests/test_prepare_offline_assets.py
git commit -m "build: prepare locked offline wheelhouse"
```

### Task 4: Extend the deterministic release builder with offline assets

**Files:**
- Create: `scripts/macho_audit.py`
- Create: `tests/test_macho_audit.py`
- Modify: `scripts/build_macos_release.py`
- Modify: `tests/test_build_macos_release.py`

- [ ] **Step 1: Add a failing offline archive fixture test**

Extend `_make_release_inputs` with tiny fixture Python/wheel trees and manifests, then add:

```python
def test_offline_release_contains_exact_runtime_contract(tmp_path: Path) -> None:
    inputs = _make_offline_release_inputs(tmp_path)
    result = build_macos_release.build_release(**inputs)
    metadata = json.loads(inputs["metadata_path"].read_text(encoding="utf-8"))
    with zipfile.ZipFile(result.archive) as archive:
        names = set(archive.namelist())
    prefix = metadata["top_level_directory"] + "/"
    assert prefix + "offline/python-manifest.json" in names
    assert prefix + "offline/wheelhouse-manifest.json" in names
    assert prefix + "offline/requirements-macos-arm64-py312.txt" in names
    assert prefix + "offline/rapidocr-model-manifest.json" in names
    assert prefix + "distribution/offline-release.json" in names
    assert prefix + "distribution/model-manifest.json" in names
    assert any(name.startswith(prefix + "offline/python-mirror/") for name in names)
    assert any(name.startswith(prefix + "offline/wheelhouse/") for name in names)
```

Add separate tests that mutate a wheel without changing size, add an extra wheel, substitute an sdist, include a symlink, change `uv.lock` after contract generation, alter the fixed target/extras object, inject the local account name, place a credential or SQLite header only inside a compressed wheel/tar member, and rebuild identical inputs twice. Expected results are `ReleaseBuildError` for mutations and equal SHA-256 for repeated builds. Add `tests/test_macho_audit.py` cases for an unsafe tar member, an escaping tar symlink, a non-arm64 Mach-O, minimum deployment target 15.0, a Homebrew absolute load dependency, a publishing-machine `LC_RPATH`, an unresolved `@rpath` dependency that would require `/opt/homebrew`, an inert upstream `/opt/homebrew` RPATH whose libraries all resolve inside the trusted closure, and allowed `/usr/lib`, `/System/Library`, `@rpath`, or `@loader_path` dependencies. Inject command output so these tests do not depend on fixture binaries.

Add a flat-model fixture whose root is named `models` and contains the exact
regular-file set from `distribution/model-manifest.json`. Prove offline metadata
accepts either that already materialized release layout or the pinned Hugging
Face `snapshots/{model_revision}` cache layout with safe blob symlinks. Keep the
legacy fixture restricted to the Hugging Face layout.

- [ ] **Step 2: Run one fixture test and observe the missing API failure**

```bash
uv run pytest tests/test_build_macos_release.py::test_offline_release_contains_exact_runtime_contract -v
```

Expected: FAIL because `build_release` has no offline input parameters.

- [ ] **Step 3: Add the offline build interface**

Add this immutable input type:

```python
@dataclass(frozen=True)
class OfflineReleaseInputs:
    python_root: Path
    python_manifest: Path
    wheelhouse_root: Path
    wheelhouse_manifest: Path
    requirements: Path
    rapidocr_model_manifest: Path
    dependency_license_manifest: Path
```

Add `offline_inputs: OfflineReleaseInputs | None = None` to `build_release`. When metadata contains `"distribution_kind": "offline"`, require this object, validate both asset trees with `scripts.offline_assets.validate_asset_tree`, require the requirements SHA-256 to equal `wheelhouse_requirements_sha256`, require `wheelhouse_uv_lock_sha256` to equal the current project `uv.lock` bytes, and strictly validate both the RapidOCR and dependency-license manifests against the normalized wheel distribution/version/hash set. Prefix Python record paths with `offline/python-mirror/` and wheel record paths with `offline/wheelhouse/`, then add every copied file to `trusted_payload` using its manifest size/SHA-256. Copy both asset manifests, requirements, and the RapidOCR manifest to the exact `offline/` paths asserted by the fixture. Copy the already validated `metadata_path` bytes to `distribution/offline-release.json` and validated `model_manifest_path` bytes to `distribution/model-manifest.json`, adding both to the offline-only trusted payload; never add them to the legacy payload.

For offline metadata, extend model copying to accept an already materialized
flat model directory only when every manifest path is a non-symlink regular
file with the exact size and SHA-256, while also accepting the existing pinned
snapshot/cache-blob layout. Keep the legacy `_copy_model_payload`
snapshot-name, cache-blob, and safe-symlink behavior unchanged.

Keep the legacy metadata schema and build path unchanged. Define a separate strict `OFFLINE_REQUIRED_METADATA_KEYS` set matching Task 1. Treat `.whl`, `.tar.gz`, `.onnx`, `.dylib`, `.so`, `.a`, `.docx`, and known Mach-O files as binary for UTF-8 scanning; exact-hash third-party archives still receive local-home, project-root, credential, file-type, and path traversal checks.

Implement `scripts/macho_audit.py` with `audit_python_archive` and `audit_wheel`. Safely inspect archive members in a private temporary directory: reject absolute/traversing names, devices, sockets, and escaping links; do not use `extractall` without a validating filter. For every Mach-O member, use absolute `/usr/bin/file`, `/usr/bin/lipo`, `/usr/bin/otool`, `/usr/bin/vtool`, and `/usr/bin/codesign` through injected argv-only subprocess calls. Parse both linked libraries and every `LC_RPATH`; require an `arm64` slice, reject minimum macOS above 14.0, reject publishing-machine RPATHs and all Homebrew/non-system absolute load dependencies, and allow `/usr/lib`, `/System/Library`, `@rpath`, `@loader_path`, and `@executable_path` only when non-system targets resolve inside the trusted Python-plus-wheel closure. A pinned upstream `/opt/homebrew` RPATH may be reported as inert only when no dependency resolution uses it and the exact outer asset hash is trusted; otherwise reject it. Record codesign status but do not require every upstream extension to carry a Developer ID signature.

The same safe archive walker must recursively stream-decompress wheel/tar/embedded-archive members with a depth limit of 3 and a total expanded-byte limit of 3 GiB, then apply path normalization, local-home/project-path, credential, SQLite-magic, forbidden file type, and first-party text scans to member names and contents. `/Users/runner` or other upstream build markers may be exempted only inside an outer asset whose filename, size, and SHA-256 already match its trusted manifest; the real publishing account/path and all credential/SQLite patterns remain forbidden even there. The release builder must run the archive, recursive privacy, and Mach-O audits after hash validation and before staging publication.

Add CLI arguments:

```text
--python-root
--python-manifest
--wheelhouse-root
--wheelhouse-manifest
--requirements
--rapidocr-model-manifest
--dependency-license-manifest
```

Require all seven together for offline metadata and reject them for legacy metadata. Preserve deterministic ZIP timestamps and `0644` modes for archives/wheels; only the launcher, uv, and Vision helper remain `0755` inside the ZIP.

- [ ] **Step 4: Run offline and legacy builder suites**

```bash
uv run pytest \
  tests/test_offline_assets.py tests/test_macho_audit.py \
  tests/test_build_macos_release.py -q
```

Expected: all tests pass, including all pre-existing legacy release tests.

- [ ] **Step 5: Commit the builder integration**

```bash
git add \
  scripts/macho_audit.py scripts/build_macos_release.py \
  tests/test_macho_audit.py tests/test_build_macos_release.py
git commit -m "build: package trusted offline runtime assets"
```

### Task 5: Bootstrap pinned Python without system Python or network

**Files:**
- Modify: `install-macos.command`
- Modify: `tests/installer/test_install_macos_launcher.py`

- [ ] **Step 1: Replace launcher expectations with failing offline-bootstrap tests**

Build the launcher fixture under `Book Library 离线 Release With Spaces`. Its
fake bundled uv must record arguments and, when called with `python install`,
create an executable at
`transaction / ".bootstrap-runtime/cpython-3.12.11-macos-aarch64-none/bin/python3.12"`.

Assert the first child command has this exact semantic argument set:

```python
assert uv_arguments == [
    "python", "install", "3.12.11",
    "--install-dir", str(transaction / ".bootstrap-runtime"),
    "--mirror", (release / "offline" / "python-mirror").as_uri(),
    "--no-bin", "--no-registry", "--offline",
    "--cache-dir", str(transaction / ".cache/uv"), "--no-config",
]
```

Assert the second child process is the isolated bootstrap Python running:

```python
assert python_executable == (
    transaction
    / ".bootstrap-runtime/cpython-3.12.11-macos-aarch64-none/bin/python3.12"
)
assert python_arguments == [
    "-S", "-m", "installer.install_macos",
    "--project-root", str(release),
    "--transaction-root", str(transaction),
]
```

Add tests that invoke the launcher from an unrelated working directory with `PATH` pointing to an empty directory, malicious inherited `PYTHONHOME`/`PYTHONPATH`/`DYLD_LIBRARY_PATH`/package-cache variables, `BASH_ENV`, exported Bash functions, `PERL5OPT`, `PERL5LIB`, `PERLLIB`, and OpenSSL/config overrides; also cover `uname -m` not equal to `arm64`, `sw_vers -productVersion` equal to `13.7`, missing/not-executable bundled uv, a same-size byte mutation of bundled uv, missing Python archive, a same-size byte mutation of the Python archive, child exit 23 propagation, and interactive-only close prompting. Require a privileged Bash shebang to suppress startup-file/function imports. Assert the hash tool, uv, and bootstrap Python receive only their exact allowlisted environments with transaction-local `TMPDIR`, and that the bootstrap Python's working directory is the exact physical `PROJECT_ROOT`; the uv Python-install environment must use `UV_PYTHON_DOWNLOADS=manual`, while every later venv/package command uses `UV_PYTHON_DOWNLOADS=never`. Both tamper cases must fail before the fake uv receives any command. Add argument tests that accept each of `--vault PATH` and `--codex-config PATH` once, but reject duplicate options, missing values, `--project-root`, `--transaction-root`, `--python`, `--skip-sync`, option abbreviations, and every unknown option before bootstrap. Add a static launcher command audit that rejects unqualified commands and any invocation of system `python3`, `pip`, `uv`, `brew`, `curl`, `git`, `xcrun`, `clang`, `swift`, `lipo`, `otool`, or `codesign`; the only executable paths permitted before bundled uv starts are the launcher shell built-ins and the explicit macOS base utilities listed in Step 3.

- [ ] **Step 2: Run launcher tests and observe legacy-command failures**

```bash
uv run pytest tests/installer/test_install_macos_launcher.py -v
```

Expected: failures showing the launcher still prefers system `python3` and invokes `uv run --python 3.12`.

- [ ] **Step 3: Implement the offline shell bootstrap**

Rewrite the non-prompt portion of `install-macos.command` to:

1. Change the shebang to `#!/bin/bash -p`. Before any external command, use only shell built-ins to unset `BASH_ENV`, `ENV`, `CDPATH`, imported-function state, `PERL5OPT`, `PERL5LIB`, `PERLLIB`, Python/package/model overrides, and dynamic-loader variables; set a fixed restrictive `umask`. Resolve `PROJECT_ROOT` from the launcher path.
2. Use absolute `/usr/bin/uname` and `/usr/bin/sw_vers`; strictly parse the leading numeric macOS major version and require arm64 plus major version 14 or newer (this naturally accepts the public 14, 15, and 26 naming sequence).
3. Require executable non-symlink `bin/uv`, regular non-symlink `installer/install_macos.py`, the fixed regular non-symlink Python archive, and every required offline/release/model manifest, requirements, and `uv.lock` as regular non-symlink files.
4. Create `.offline-install-stage.XXXXXX` beneath the project using `/usr/bin/mktemp -d`, identity-record it, install a cleanup trap, and create `.bootstrap-runtime`, `.home`, `.tmp`, `.empty-path`, and cache children with absolute `/bin/mkdir`.
5. Convert the Python mirror path to a `file://` URI without depending on `PATH`; test the conversion against spaces and UTF-8.
6. Compute both digests by invoking absolute `/usr/bin/shasum -a 256` through `/usr/bin/env -i` with only fixed locale plus transaction-local `HOME`, `TMPDIR`, and empty `PATH`; this makes its `/usr/bin/perl` child independent of inherited Perl state. Compare uv to `c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554` and the Python archive to `407fa242942a7ba5d91899abc562fc9897f7a0376f8d2060285e8c0560323f19` before invoking uv.
7. Start uv and the bootstrap Python through separate `/usr/bin/env -i` allowlists, never inherited shell state. Both receive transaction-local `HOME`, `TMPDIR`, empty `PATH`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, and `HF_HOME`, plus offline/no-user-site variables. The explicit `uv python install` process alone receives `UV_PYTHON_DOWNLOADS=manual` because `never` rejects even an explicit local-mirror install in uv 0.11.26; it still receives `--offline`, the exact `file://` mirror, `--no-registry`, and an explicit cache. The bootstrap Python and all uv commands it later launches receive `UV_PYTHON_DOWNLOADS=never`. Pass `--cache-dir "$TRANSACTION_ROOT/.cache/uv"` rather than `--no-cache`, so uv cannot choose a temporary directory outside the transaction.
8. Invoke bundled uv with the exact arguments asserted above.
9. Parse launcher arguments into a Bash array before bootstrap. Accept only one `--vault PATH` and one `--codex-config PATH`, reject missing/duplicate/unknown/trusted-path/bypass options, then use a shell-builtin `cd` inside the child subshell to lock its physical cwd to `PROJECT_ROOT` and invoke the fixed bootstrap interpreter as `-S -m installer.install_macos` with exactly one launcher-owned `--project-root "$PROJECT_ROOT"`, exactly one launcher-owned `--transaction-root "$TRANSACTION_ROOT"`, and only those validated optional pairs. `-S` prevents startup-time loading of site/customization packages before the installer validates the release. Never append raw `$@`.
10. Keep `.bootstrap-runtime` in place for the entire installer process. After the bootstrap Python exits on either success or failure, preserve its exit status, recheck that the saved transaction path is the unchanged direct child created by `/usr/bin/mktemp` beneath the unchanged project root, and only then remove that tree with absolute `/bin/rm -rf --`. The Python installer must never delete the runtime from which its own still-running process is executing.

Do not call system `python3`, `pip`, `curl`, `git`, Homebrew, user `uv`,
`xattr`, `spctl`, `xcrun`, `clang`, `swift`, `lipo`, `otool`, `codesign`,
Xcode, or Command Line Tools. The target-side launcher may use only shell
built-ins plus absolute `/usr/bin/env`, `/usr/bin/uname`, `/usr/bin/sw_vers`,
`/usr/bin/shasum` and its system `/usr/bin/perl` interpreter,
`/usr/bin/mktemp`, `/bin/mkdir`, and `/bin/rm` before it starts the verified
bundled uv binary. `/bin/rm` may remove only the launcher-created transaction
directory after its identity and project-root containment are rechecked.

- [ ] **Step 4: Run launcher tests**

```bash
uv run pytest tests/installer/test_install_macos_launcher.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the bootstrap**

```bash
git add install-macos.command tests/installer/test_install_macos_launcher.py
git commit -m "feat: bootstrap bundled python fully offline"
```

### Task 6: Create the relocatable venv transaction

**Files:**
- Create: `installer/offline_runtime.py`
- Create: `tests/installer/test_offline_runtime.py`

- [ ] **Step 1: Write failing exact-command tests**

Add a fixture transaction containing `.bootstrap-runtime/cpython-3.12.11-macos-aarch64-none/bin/python3.12`, an empty `.venv` target, wheelhouse, wheelhouse manifest, and requirements. `build_environment` must first safely clone that fixed bootstrap tree to `.runtime`, build the venv from the clone, and seal the venv for paired relocation. Assert:

```python
commands = offline_runtime.environment_commands(layout)
assert commands == [
    [
        str(layout.uv), "venv", "--relocatable", "--python",
        str(layout.runtime_python), str(layout.staged_venv),
        "--no-python-downloads", "--offline",
        "--cache-dir", str(layout.cache), "--no-config",
    ],
    [
        str(layout.uv), "pip", "sync", "--python", str(layout.staged_python),
        "--offline", "--no-index", "--find-links", str(layout.wheelhouse),
        "--require-hashes", "--no-build", "--no-python-downloads",
        "--link-mode", "copy", str(layout.requirements),
        "--cache-dir", str(layout.cache), "--no-config",
    ],
]
```

Add tests proving `build_environment` supplies transaction-local HOME/cache/empty-PATH directories, removes inherited `PYTHONPATH`, `PYTHONHOME`, `VIRTUAL_ENV`, `CONDA_PREFIX`, `PIP_*`, `UV_*`, `HF_*`, `TRANSFORMERS_*`, and `DYLD_*` values before adding its explicit offline values, rejects any resolved path outside `project_root`/`transaction_root`, cleans partial `.runtime`/`.venv` candidates while leaving the live `.bootstrap-runtime` untouched on failure, and never uses `shell=True`. Parameterize target-side contract corruption as a same-size wheel mutation, an extra wheel, a wheel symlink, a requirements-file mutation, a model/helper/source mutation covered only by `RELEASE-MANIFEST.json`, a symlinked listed release file, and an unlisted executable or root-level `sitecustomize.py`. Each case must raise `OfflineRuntimeError` before the injected command runner records either uv command. Add repeat-install fixtures proving only the exact current transaction plus known mutable install/data locations are tolerated and that a sibling staging tree or an extra file anywhere else is rejected.

Add relocation tests for uv 0.11.26's actual behavior: start with an absolute
`.venv/bin/python` symlink and an absolute transaction `home =` entry in
`pyvenv.cfg`; require the sealing step to replace the interpreter link with the
exact relative sibling-runtime target, remove only the validated `home` entry,
retain `relocatable = true`, and leave no transaction path in launchers,
activation scripts, configuration, or text metadata. Move the paired
`.runtime` and `.venv` directories to a different final parent and execute the
interpreter from an unrelated working directory; require exact final
`sys.executable`, `sys.prefix`, `sys.base_prefix`, and stdlib containment. Also
prove an unexpected symlink target, unexpected `pyvenv.cfg`, remaining
transaction path, or failed post-seal interpreter probe aborts before
publication.

- [ ] **Step 2: Run and observe missing-module failure**

```bash
uv run pytest tests/installer/test_offline_runtime.py -v
```

Expected: collection ERROR because `installer.offline_runtime` does not exist.

- [ ] **Step 3: Implement layout and environment creation**

Create:

```python
@dataclass(frozen=True)
class OfflineLayout:
    project_root: Path
    transaction_root: Path
    uv: Path
    bootstrap_runtime_root: Path
    bootstrap_python: Path
    staged_runtime_root: Path
    runtime_python: Path
    staged_venv: Path
    staged_python: Path
    wheelhouse: Path
    wheelhouse_manifest: Path
    rapidocr_model_manifest: Path
    requirements: Path
    cache: Path
    home: Path
    temporary: Path
    empty_path: Path


def environment_commands(layout: OfflineLayout) -> list[list[str]]:
    return [
        [str(layout.uv), "venv", "--relocatable", "--python",
         str(layout.runtime_python), str(layout.staged_venv),
         "--no-python-downloads", "--offline", "--cache-dir",
         str(layout.cache), "--no-config"],
        [str(layout.uv), "pip", "sync", "--python", str(layout.staged_python),
         "--offline", "--no-index", "--find-links", str(layout.wheelhouse),
         "--require-hashes", "--no-build", "--no-python-downloads",
         "--link-mode", "copy", str(layout.requirements),
         "--cache-dir", str(layout.cache), "--no-config"],
    ]
```

Implement `layout_for(project_root, transaction_root) -> OfflineLayout` with these exact mappings:

```python
return OfflineLayout(
    project_root=project_root,
    transaction_root=transaction_root,
    uv=project_root / "bin/uv",
    bootstrap_runtime_root=transaction_root / ".bootstrap-runtime",
    bootstrap_python=(
        transaction_root
        / ".bootstrap-runtime/cpython-3.12.11-macos-aarch64-none/bin/python3.12"
    ),
    staged_runtime_root=transaction_root / ".runtime",
    runtime_python=(
        transaction_root
        / ".runtime/cpython-3.12.11-macos-aarch64-none/bin/python3.12"
    ),
    staged_venv=transaction_root / ".venv",
    staged_python=transaction_root / ".venv/bin/python",
    wheelhouse=project_root / "offline/wheelhouse",
    wheelhouse_manifest=project_root / "offline/wheelhouse-manifest.json",
    rapidocr_model_manifest=project_root / "offline/rapidocr-model-manifest.json",
    requirements=project_root / "offline/requirements-macos-arm64-py312.txt",
    cache=transaction_root / ".cache/uv",
    home=transaction_root / ".home",
    temporary=transaction_root / ".tmp",
    empty_path=transaction_root / ".empty-path",
)
```

Implement `clone_bootstrap_runtime(layout: OfflineLayout) -> None` with
descriptor-relative traversal. Require the currently executing interpreter to
be the exact resolved `layout.bootstrap_python`, CPython 3.12.11, and arm64.
Require the bootstrap root and every ancestor to be unchanged real
directories; reject absolute/escaping links, devices, sockets, setuid/setgid,
world-writable files, controls, non-NFC names, and portable-name collisions.
Copy regular bytes, modes, and only safe relative internal symlinks into a new
same-filesystem `layout.staged_runtime_root`; never hard-link to the bootstrap
tree or external cache. Rewalk both trees and require identical relative
path/type/mode/size/SHA-256/link-target records before invoking uv.

Implement `seal_relocatable_venv(layout: OfflineLayout) -> None` after `uv pip
sync`. uv 0.11.26's `--relocatable` repairs entrypoint and activation scripts
but deliberately leaves the base interpreter link and `pyvenv.cfg home`
absolute. Require `.venv/bin/python` to be exactly the absolute symlink to
`layout.runtime_python`; atomically replace it with
`../../.runtime/cpython-3.12.11-macos-aarch64-none/bin/python3.12`, while
requiring `python3` and `python3.12` to remain relative links to `python`.
Strictly parse `pyvenv.cfg`, require its `home` value to equal
`layout.runtime_python.parent`, require `uv = 0.11.26`, exact version and
`relocatable = true`, then atomically rewrite it without the `home` line. For
this fixed CPython build, absence of `home` makes base-prefix discovery follow
the now-relative real executable; do not write either the transaction path or
the eventual machine-specific final path into the staged venv. Scan every text
launcher, activation script, `.pth`, `pyvenv.cfg`, and symlink target for the
transaction path, publishing-machine paths, and unapproved absolute paths. uv
0.11.26 intentionally emits relocatable console-script wrappers beginning with
the exact operating-system shebang `#!/bin/sh` and a fixed relative
`dirname`/`realpath` template; accept that one version-pinned template only
after parsing the whole wrapper, proving it embeds no absolute
interpreter/project path, and recording that these console scripts are not used
by the installer or release verifier. Reject any other outside-root shebang,
absolute reference, shell body, or wrapper variant. The Task 11 process
allowlist therefore does not need `/bin/sh`, `dirname`, or `realpath`: all
supported application paths invoke the sealed Python with `-m`. Add fixtures
for the exact accepted uv wrapper, a modified shell command, an absolute
interpreter, a transaction path, and a publishing-user path. Finally execute
the sealed staged interpreter from `layout.empty_path` and require its
executable, prefix, base prefix, and stdlib to resolve only inside the staged
venv/runtime.

Implement `validate_local_contract(layout: OfflineLayout) -> None` inside the shipped installer package rather than importing build-only `scripts/offline_assets.py`. It must strictly parse the wheelhouse manifest, require only `schema_version`, `kind`, `requirements_sha256`, `uv_lock_sha256`, `target`, and `files`, require the fixed target/extras object from Task 2, require every record to contain only `path`, `size`, `sha256`, `mode`, distribution/version/tags metadata, require every path to be a portable `.whl` basename, and reject exact/NFC/case-fold duplicates. Using descriptor-relative `O_NOFOLLOW` traversal, require an exact flat set of `0644` regular wheel files with no symlinks, subdirectories, extras, or missing entries; compare every size and SHA-256; require the requirements, `uv.lock`, and RapidOCR manifest to be non-symlink `0644` regular files; compare requirements and lock hashes to the wheelhouse contract; then strictly validate the RapidOCR source wheel hash and exact three ONNX records by streaming those members from the trusted wheel and comparing member size/SHA-256. This deliberate target-side duplicate validation is required because `scripts/offline_assets.py` is not included in the release source allowlist.

Also implement `validate_release_payload(layout: OfflineLayout) -> None`: strictly parse `RELEASE-MANIFEST.json`, verify its embedded release metadata equals shipped `distribution/offline-release.json`, and descriptor-hash every manifest-listed immutable file with its exact size/mode while rejecting symlink/special/path-normalization collisions. Walk the release root and reject every unlisted path except the identity-checked current `transaction_root` and these exact repeat-install mutable locations: `.runtime/`, `.venv/`, `.codex/config.toml`, `data/library.sqlite3` plus its three SQLite sidecar suffixes, `data/ocr/`, `data/ocr-models/rapidocr/`, the default `Obsidian书库/`, and a regular non-symlink root `.DS_Store`. This is an allowlist, not a blanket exemption for generated-looking names; reject sibling staging trees, root-level Python customization modules, and additions under immutable source/model/offline directories. Call this before `validate_local_contract`, so model/helper/source corruption fails before environment construction and the target never needs `lipo`, `otool`, Xcode Command Line Tools, or other developer utilities.

`build_environment` must call `validate_release_payload` and then
`validate_local_contract`, verify and clone the bootstrap runtime, create the
isolated directories, run both commands with argv lists and `check=True`, seal
the venv, and use one sanitized environment
for every uv/Python validation subprocess. Preserve only safe locale values,
set transaction-local `HOME`, `TMPDIR`, `PATH`, `XDG_CACHE_HOME`,
`UV_CACHE_DIR`, `PIP_CACHE_DIR`, and `HF_HOME`, then set `UV_OFFLINE=1`,
`UV_NO_CONFIG=1`, `UV_PYTHON_DOWNLOADS=never`, `PIP_NO_INDEX=1`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONNOUSERSITE=1`, and
`PYTHONDONTWRITEBYTECODE=1`; do not pass through user Python, package-manager,
model-cache, or dynamic-loader overrides. Do not use `--no-cache`: every uv
command receives the explicit transaction cache directory.

Also implement `installation_fingerprint(layout: OfflineLayout) -> str` as lowercase SHA-256 over canonical JSON containing the byte-level SHA-256 of the shipped `distribution/offline-release.json`, `distribution/model-manifest.json`, `offline/python-manifest.json`, wheelhouse manifest, requirements, RapidOCR manifest, `uv.lock`, `bin/uv`, and `bin/book-vision-ocr`. Add tests proving stable inputs yield the same digest and a same-size contract mutation changes it. The digest will be written into the staged runtime and used for safe idempotent reuse in Tasks 7–8.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/installer/test_offline_runtime.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit environment construction**

```bash
git add installer/offline_runtime.py tests/installer/test_offline_runtime.py
git commit -m "feat: build relocatable environment from wheelhouse"
```

### Task 7: Add staged runtime self-tests and rollback-safe publication

**Files:**
- Create: `installer/runtime_selftest.py`
- Create: `tests/installer/test_runtime_selftest.py`
- Modify: `installer/offline_runtime.py`
- Modify: `tests/installer/test_offline_runtime.py`

- [ ] **Step 1: Write failing self-test and rollback tests**

The self-test unit suite must call
`run_checks(project_root, expected_runtime_root, forbidden_path)` and assert it
returns a JSON-serializable report containing:

```python
assert report["python"] == {
    "implementation": "CPython", "major": 3, "minor": 12, "micro": 11
}
assert report["architecture"] == "arm64"
assert report["paths"] == {
    "executable": str(expected_python),
    "prefix": str(expected_venv),
    "base_prefix": str(expected_runtime),
    "stdlib": str(expected_stdlib),
}
assert report["imports"] == sorted(
    ["cv2", "fitz", "mcp", "numpy", "onnxruntime", "rapidocr",
     "sentence_transformers", "torch"]
)
assert report["embedding_dimensions"] == 384
assert report["rapidocr"] == "ok"
assert report["mcp"] == "ok"
```

Use dependency injection for embedding/OCR/MCP probes in unit tests so the real heavy checks remain in the integration gate. Add tree-validation cases for an absolute symlink other than the pre-seal exact uv interpreter link, an escaping relative symlink, a text launcher/shebang containing the transaction path, and an interpreter/prefix/stdlib outside the project. Add target-ancestor cases where `.codex`, `data`, `data/ocr-models`, the custom config parent, or Vault parent is a symlink. Add `publish_environment` tests that begin with valid old `.runtime`, `.venv`, managed RapidOCR model directory, config, Vault layout, and database plus a live bootstrap interpreter tree that must never move; inject failure after the first candidate move, during final-path validation, during tracked directory/database initialization, and during config writing; then assert a recursive before/after tree snapshot and all old bytes are identical while `.bootstrap-runtime` remains readable. Add a no-prior-install case proving rollback removes every newly published file and only still-empty directories created by this attempt.

- [ ] **Step 2: Run and observe missing behavior**

```bash
uv run pytest \
  tests/installer/test_runtime_selftest.py \
  tests/installer/test_offline_runtime.py -v
```

Expected: failures for missing `runtime_selftest` and `publish_environment`.

- [ ] **Step 3: Implement self-test and transactional publication**

`runtime_selftest.py` must expose `run_checks(project_root: Path, expected_runtime_root: Path, forbidden_path: Path | None = None) -> dict[str, object]` and corresponding required `--project-root`/`--expected-runtime-root` plus optional `--forbid-path` CLI arguments that print exactly one JSON object. Invoke it as a module with subprocess `cwd=project_root` explicitly, even when the parent was started elsewhere, so local `book_agent` imports do not depend on installing the project into the venv. Every nested `-m book_agent...` or `-m installer...` subprocess in Tasks 7–8 must likewise receive the exact project cwd rather than inherit one. The production probes must:

- import the eight critical packages;
- confirm `platform.machine() == "arm64"` and exactly CPython 3.12.11;
- confirm resolved `sys.executable`, `sys.prefix`, `sys.base_prefix`, and `sysconfig.get_path("stdlib")` all match the expected venv/runtime closure inside the project; reject `forbidden_path` in any reported or text runtime path after final publication;
- validate the exact non-symlink model file set, sizes, and SHA-256 values against `distribution/model-manifest.json` before loading it;
- load `data/models` with `local_files_only=True` and return a finite 384-value embedding;
- locate the three installed RapidOCR wheel models, verify their exact member size/SHA-256 against `offline/rapidocr-model-manifest.json`, then recognize a generated in-memory `OFFLINE OCR 123` image;
- use `mcp.client.stdio.stdio_client`, `ClientSession`, and `StdioServerParameters` to start `[sys.executable, "-m", "book_agent.mcp_server"]` with the project cwd and an empty temporary library/Vault root; complete `initialize`, require the exact tools from `TOOL_NAMES` through protocol `tools/list`, and call `library_status` through protocol `tools/call`; an in-process function call is not sufficient;
- invoke `bin/book-vision-ocr --capabilities` and require schema 2 with `zh-Hans` and `en-US`.

Add to `offline_runtime.py`:

```python
def publish_environment(
    *,
    layout: OfflineLayout,
    staged_ocr_models: Path,
    final_ocr_models: Path,
    config_path: Path,
    config_text: str,
    validate_final: Callable[[Path], None],
    finalize_install: Callable[[CreationJournal], None],
    write_config: Callable[[Path, str], None],
    validate_config: Callable[[Path], None],
) -> None:
    """Publish all managed install targets or restore every old byte."""
```

Implement a tree validator in `offline_runtime.py` that permits only relative symlinks resolving inside the validated runtime/venv closure and scans text launchers, shebangs, and `pyvenv.cfg` for the transaction path or publishing-machine path. Call it both before and after publication. Add `CreationJournal`, which records device/inode/type for each file or directory created during finalization and rolls back only unchanged new files and unchanged empty directories in reverse order.

Before staged checks, write `installation_fingerprint(layout) + "\n"` as mode `0644` to `transaction_root/.runtime/INSTALLATION-FINGERPRINT`. Before any mutation, safely walk every existing ancestor of `.runtime`, `.venv`, `data/ocr-models/rapidocr`, config, database, and Vault with descriptor-relative `O_NOFOLLOW`; reject symlink or non-directory ancestors and recheck identities during publication. The publication function must use same-filesystem `os.replace` for staged `.runtime`, staged `.venv`, and the staged RapidOCR directory; unique same-parent backups for existing managed directories and the config; `validate_final(layout.project_root)`; `finalize_install(journal)`; atomic config writing; `validate_config(config_path)`; rollback in reverse order; and backup cleanup only after every step succeeds. Keep all backups live through finalization and config validation, so any failure restores the previous runtime, venv, OCR models, exact config bytes, and removes journaled new runtime/Vault/database artifacts. If no old managed target existed, rollback removes the newly published target. The parent installer continues executing only from `transaction_root/.bootstrap-runtime` throughout these moves; never move, rename, rewrite, or delete that tree from the Python process. Recheck `sys.executable`, `sys.base_prefix`, and stdlib remain inside the unchanged bootstrap tree immediately before and after publication, and let the shell remove it only after the installer process exits.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest \
  tests/installer/test_runtime_selftest.py \
  tests/installer/test_offline_runtime.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit self-test and publication**

```bash
git add \
  installer/runtime_selftest.py installer/offline_runtime.py \
  tests/installer/test_runtime_selftest.py tests/installer/test_offline_runtime.py
git commit -m "feat: verify and atomically publish offline runtime"
```

### Task 8: Integrate offline runtime installation and Codex config generation

**Files:**
- Create: `installer/initialize_library.py`
- Modify: `installer/install_macos.py`
- Create: `tests/installer/test_initialize_library.py`
- Modify: `tests/installer/test_install_macos.py`

- [ ] **Step 1: Write failing installer contract tests**

Replace the legacy `uv sync` expectations with tests asserting:

```python
result = install_macos.install(
    project_root=project_root,
    transaction_root=transaction_root,
    run_command=fake_runner,
)
assert result.python == project_root / ".venv" / "bin" / "python"
assert not any("sync" in command and "--find-links" not in command for command in calls)
```

Update the config assertion to include:

```python
assert server["env"]["PYTHONNOUSERSITE"] == "1"
assert server["env"]["PYTHONPATH"] == ""
assert server["env"]["PYTHONHOME"] == ""
```

Add failure tests for a missing/mismatched live bootstrap runtime, a failed safe bootstrap-to-staged clone, wrong exact Python version/architecture/path report, self-test nonzero exit, RapidOCR trusted digest mismatch, Vision hash/header/capabilities failure, final-path validation failure, tracked Vault/database initialization failure, and config-write/config-parse failure. Start the installer from an unrelated parent cwd and assert every `-m installer...`, `-m book_agent...`, uv, self-test, and MCP subprocess receives the explicit intended cwd. Prove Vision validation succeeds with an empty PATH and never calls `lipo`, `otool`, `codesign`, Xcode, or Command Line Tools. Compare recursive before/after trees so every failure leaves the old `.runtime`, `.venv`, managed RapidOCR model directory, database, Vault, and `.codex/config.toml` unchanged with no new empty directories, while the live `.bootstrap-runtime` remains available until the installer returns. Add idempotency tests proving a matching `INSTALLATION-FINGERPRINT` plus successful final self-tests reuses the existing environment without cloning or either uv environment command, while a fingerprint mismatch or failed revalidation rebuilds transactionally. Add parser tests requiring exactly one `--project-root` and `--transaction-root`, disabling abbreviations, accepting only optional Vault/config paths, and proving `--skip-sync`, `--python`, duplicate trusted flags, and unknown options are not public.

- [ ] **Step 2: Run focused tests and observe legacy sync failures**

```bash
uv run pytest tests/installer/test_install_macos.py -v
```

Expected: failures because `install()` does not accept `transaction_root` and still runs `uv sync --python 3.12`.

- [ ] **Step 3: Replace network environment setup with the transaction API**

Create `installer.initialize_library` as a final-venv CLI with required
`--project-root` and `--vault`. It may initialize an empty database and Vault
layout only when `data/library.sqlite3` was absent at the start of this install;
if a database already exists, validate it as one non-symlink regular file and
do not migrate or modify it. The parent installer records every newly created
file/directory in `CreationJournal` and verifies the resulting empty database
with `PRAGMA quick_check` before config publication. Track and remove any new
`-journal`, `-wal`, or `-shm` sidecar on failure; require no sidecar to remain
after successful initialization.

Change `install()` to require `transaction_root` for production installation and call, in order:

1. `offline_runtime.layout_for(project_root, transaction_root)`, verify the current process is executing from the exact layout bootstrap interpreter, then call `validate_release_payload(layout)` and `validate_local_contract(layout)`.
2. Compute `installation_fingerprint(layout)`. If the final `.runtime/INSTALLATION-FINGERPRINT` matches, run final-path `uv pip check`, `runtime_selftest.py`, and managed RapidOCR model validation; reuse only if all pass.
3. When reuse is not safe, call `offline_runtime.build_environment` with the layout and injected runner, then run the cache-pinned `uv pip check` command and staged module self-test with `--expected-runtime-root` equal to `transaction_root/.runtime`, using the same sanitized environment.
4. Copy the three RapidOCR models from the staged venv into `transaction_root/.ocr-models/rapidocr`, rejecting symlinks and verifying every source and destination against `offline/rapidocr-model-manifest.json`, not merely against each other.
5. Validate the existing Vision helper against its already verified release-manifest hash, parse its thin 64-bit Mach-O header in Python to require CPU type arm64, and run `--capabilities`; do not invoke target-machine `lipo`, `otool`, `codesign`, Xcode, or Command Line Tools. Validate all target ancestors and the rendered complete config text before any managed target is replaced. Do not create Vault, runtime, database, config-parent, or OCR-parent directories yet.
6. Call `offline_runtime.publish_environment` with staged/final OCR paths and a `validate_final(project_root)` callback that reruns exact CPython 3.12.11 architecture/path checks, critical imports, cache-pinned `uv pip check`, fingerprint, and managed OCR validation from final paths. Its `finalize_install(journal)` callback creates the empty Vault/runtime layout, invokes final `.venv/bin/python -m installer.initialize_library` only for a previously absent database, and records/validates every new inode; after that callback returns, `publish_environment` atomically writes and parses the config while all backups are still live.
7. On the reuse path, call the shared journaled finalization helper so directory/database/config failure is equally reversible without replacing the environment. On either path, return with `.bootstrap-runtime` and the transaction directory intact; the launcher preserves the exit code and removes them only after this Python process has fully exited. Any exception leaves shell cleanup enabled and exits nonzero.

Keep an explicit `skip_environment_install` keyword only for direct unit tests that exercise config rendering; never add it to `_build_parser` or `main`. Configure argparse with `allow_abbrev=False`, pre-scan to reject duplicate singleton flags, make `--project-root` and `--transaction-root` required, and expose only `--vault` and `--codex-config` besides them. Remove public `--skip-sync`, public `--python`, `_find_uv`, `_sync_environment`, PATH fallback, and network-oriented error messages once no test references them.

- [ ] **Step 4: Run installer and launcher suites**

```bash
uv run pytest \
  tests/installer/test_offline_runtime.py \
  tests/installer/test_runtime_selftest.py \
  tests/installer/test_initialize_library.py \
  tests/installer/test_install_macos.py \
  tests/installer/test_install_macos_launcher.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit installer integration**

```bash
git add \
  installer/initialize_library.py installer/install_macos.py \
  tests/installer/test_initialize_library.py tests/installer/test_install_macos.py
git commit -m "feat: install book library from offline runtime"
```

### Task 9: Prepare real assets and pin their contracts

**Files:**
- Generate under ignored path: `dist/offline-assets/python-mirror/20251007/*.tar.gz`
- Generate under ignored path: `dist/offline-assets/wheelhouse/*.whl`
- Generate and commit: `distribution/python-manifest.json`
- Generate and commit: `distribution/wheelhouse-manifest.json`
- Generate and commit: `distribution/requirements-macos-arm64-py312.txt`
- Generate and commit: `distribution/rapidocr-model-manifest.json`
- Generate and commit: `third_party/python-dependencies/manifest.json`

- [ ] **Step 1: Download the fixed Python asset**

Run from the isolated worktree root:

```bash
mkdir -p dist/offline-assets/python-mirror/20251007
curl -fL --retry 3 --connect-timeout 15 \
  -o 'dist/offline-assets/python-mirror/20251007/cpython-3.12.11+20251007-aarch64-apple-darwin-install_only_stripped.tar.gz' \
  'https://github.com/astral-sh/python-build-standalone/releases/download/20251007/cpython-3.12.11%2B20251007-aarch64-apple-darwin-install_only_stripped.tar.gz'
shasum -a 256 'dist/offline-assets/python-mirror/20251007/cpython-3.12.11+20251007-aarch64-apple-darwin-install_only_stripped.tar.gz'
```

Expected SHA-256: `407fa242942a7ba5d91899abc562fc9897f7a0376f8d2060285e8c0560323f19`.

- [ ] **Step 2: Produce wheels and contract files**

```bash
uv run python scripts/prepare_offline_assets.py \
  --project-root "$PWD" \
  --uv "$(command -v uv)" \
  --python-archive 'dist/offline-assets/python-mirror/20251007/cpython-3.12.11+20251007-aarch64-apple-darwin-install_only_stripped.tar.gz' \
  --output-dir dist/offline-assets \
  --contract-dir distribution \
  --dependency-license-manifest third_party/python-dependencies/manifest.json
```

Expected from the current lock: 81 wheel files, zero sdists, and five generated contract/license files. Derive and report the actual wheel count from the manifest rather than asserting a literal count in implementation code.

- [ ] **Step 3: Independently validate the generated assets**

```bash
uv run python -m scripts.prepare_offline_assets \
  --project-root "$PWD" \
  --uv "$(command -v uv)" \
  --python-archive 'dist/offline-assets/python-mirror/20251007/cpython-3.12.11+20251007-aarch64-apple-darwin-install_only_stripped.tar.gz' \
  --output-dir dist/offline-assets \
  --contract-dir distribution \
  --dependency-license-manifest third_party/python-dependencies/manifest.json \
  --verify-only
uv run pytest tests/test_offline_assets.py tests/test_prepare_offline_assets.py -q
```

Expected: asset verification succeeds and all tests pass without modifying the contract files.

- [ ] **Step 4: Review and commit only contracts and the generated inventory**

```bash
git diff --check
git add \
  distribution/python-manifest.json \
  distribution/wheelhouse-manifest.json \
  distribution/requirements-macos-arm64-py312.txt \
  distribution/rapidocr-model-manifest.json \
  third_party/python-dependencies/manifest.json
git commit -m "build: pin offline python dependency closure"
```

### Task 10: Update release privacy, licenses, and user documentation

**Files:**
- Modify: `THIRD_PARTY_NOTICES.md`
- Create: `third_party/python/LICENSE`
- Create: `third_party/python-build-standalone/NOTICE.txt`
- Use generated: `third_party/python-dependencies/manifest.json`
- Modify: `README.md`
- Modify: `docs/安装说明.md`
- Modify: `docs/常见问题.md`
- Modify: `docs/隐私与数据存放.md`
- Modify: `scripts/create_word_guides.py`
- Regenerate: `docs/word/安装说明.docx`
- Regenerate: `docs/word/使用说明.docx`
- Modify: `scripts/build_macos_release.py`
- Modify: `tests/test_release_docs.py`
- Modify: `tests/test_build_macos_release.py`

- [ ] **Step 1: Make the documentation contract fail**

Change `tests/test_release_docs.py` so public documents must contain:

```python
required = (
    "v0.3 离线 ZIP 安装过程不下载 Python、Python 包、语义模型或 OCR 模型",
    "不需要 Homebrew",
    "不需要预先安装 Python",
    "不需要 uv、pip、Xcode 或 Command Line Tools",
    "解压后只需运行一次 install-macos.command",
    "v0.3 安装缓存只存在于解压目录内的临时事务中，结束后自动删除",
    "最低 macOS 14",
    "所有 Apple Silicon M 系列",
    "建议预留 3–5 GB",
    "Codex 对话仍可能需要联网",
    "安装并导入书籍后不要随意移动项目目录",
)
```

Within the v0.3 offline-ZIP sections, reject stale claims `首次依赖安装仍然需要联网`, `uv sync 安装依赖失败`, `安装工具可能在系统用户缓存中保留下载缓存`, and `不提供 OCR`. Separately require a clearly labeled legacy note that the still-published `install-from-github.command` downloads the existing online `v0.2.0-beta.1` asset/dependencies and therefore needs network; do not apply the v0.3 offline promise to that command. Extend `RELEASE_TEXT_PATHS` with both Python notice files and the dependency manifest. Add builder tests requiring the new files in offline ZIPs while leaving the legacy source allowlist unchanged.

- [ ] **Step 2: Run docs tests and observe missing/old-text failures**

```bash
uv run pytest tests/test_release_docs.py tests/test_build_macos_release.py -q
```

Expected: failures for missing notices and stale online-install wording.

- [ ] **Step 3: Update text and licensing artifacts**

- Copy the unmodified CPython 3.12 license from the pinned Python archive into `third_party/python/LICENSE`.
- Write `third_party/python-build-standalone/NOTICE.txt` with the exact upstream project name, asset filename, release `20251007`, source URL, SHA-256, and pointer to licenses embedded in the archive.
- Use the generated `third_party/python-dependencies/manifest.json` from Task 9 and retain the original wheels with their `.dist-info/licenses` entries.
- Update `THIRD_PARTY_NOTICES.md` to enumerate CPython/PSF, python-build-standalone, uv, E5, Apple system frameworks, and the wheel manifest.
- Update all Markdown documents with the approved install/offline/Gatekeeper/data boundaries and keep `install-from-github.command` documented as the existing online `v0.2.0-beta.1` path until the new release is uploaded.
- Update `scripts/create_word_guides.py`, regenerate both DOCX files, then use the `documents` skill render-and-verify workflow to inspect every generated page before accepting them.
- Add the new notice/manifest files only to the offline source allowlist in `scripts/build_macos_release.py`.

- [ ] **Step 4: Run docs, policy, and release tests**

```bash
uv run pytest \
  tests/test_release_docs.py tests/test_user_guide.py tests/test_project_policy.py \
  tests/test_build_macos_release.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit documentation and notices**

```bash
git add \
  THIRD_PARTY_NOTICES.md third_party/python third_party/python-build-standalone \
  third_party/python-dependencies README.md docs scripts/create_word_guides.py \
  scripts/build_macos_release.py tests/test_release_docs.py \
  tests/test_build_macos_release.py
git commit -m "docs: document fully offline macos installation"
```

### Task 11: Add a repeatable hard-offline release verifier

**Files:**
- Create: `scripts/verify_offline_release.py`
- Create: `scripts/check_offline_report.py`
- Create: `tests/test_verify_offline_release.py`
- Create: `tests/test_check_offline_report.py`
- Reuse without changing: `scripts/macho_audit.py`

- [ ] **Step 1: Write failing verifier command and report tests**

Inject a runner and assert the verifier constructs a sandboxed install with a
dynamic allowlist profile. External-dependency proof must not rely on an
enumerated denylist:

```python
assert install_command[:2] == ["/usr/bin/sandbox-exec", "-p"]
profile = install_command[2]
assert "(deny network*)" in profile
assert "(deny process-exec)" in profile
assert "(deny file-read-data)" in profile
assert "(deny file-write*)" in profile
assert '(allow file-read-data (literal "/"))' in profile
for allowed_executable in (
    "/bin/bash", "/bin/mkdir", "/bin/rm", "/usr/bin/env",
    "/usr/bin/uname", "/usr/bin/sw_vers", "/usr/bin/shasum",
    "/usr/bin/perl", "/usr/bin/mktemp", "/usr/bin/codesign",
):
    assert f'(literal "{allowed_executable}")' in profile
assert f'(subpath "{extraction_root}")' in profile
assert f'(subpath "{work_dir}")' in profile
for external_root in (
    str(real_user_home), "/Applications", "/Library", "/Network",
    "/opt", "/usr/local", "/nix", "/sw",
):
    assert f'(allow file-read-data (subpath "{external_root}"))' not in profile
for forbidden_executable in (
    "/usr/bin/python3", "/usr/bin/pip3", "/usr/bin/curl",
    "/usr/bin/git", "/usr/bin/xcrun", "/usr/bin/clang",
    "/usr/bin/swift", "/usr/bin/lipo", "/usr/bin/otool",
):
    assert f'(allow process-exec (literal "{forbidden_executable}"))' not in profile
assert install_command[3] == str(extracted_launcher)
assert install_environment["PATH"] == str(empty_path)
assert install_environment["HOME"] == str(clean_home)
assert install_environment["UV_OFFLINE"] == "1"
```

The report test must require `archive_sha256`, `host_macos`, `host_arch`, `controller_scope`, `sandbox_profile`, `sandbox_profile_sha256`, `network_probes`, `external_dependency_probes`, `network_denied`, `external_dependencies`, `zip_crc`, `release_manifest`, `safe_extract`, `first_install`, `second_install`, `imports`, `embedding_dimensions`, `rapidocr`, `vision`, `mcp_protocol`, `txt_retrieval`, `privacy_scan`, and `macho_audit`. Require both digests to be 64-character lowercase SHA-256 values, require `sandbox_profile_sha256` to equal the SHA-256 of the exact bounded UTF-8 `sandbox_profile` string, `host_arch == "arm64"`, the explicit controller/target scope string, `embedding_dimensions == 384`, and `network_denied`, `external_dependencies`, plus every named pass/fail gate equal to `ok`. The profile is an allowlist and therefore contains only the disposable `/private/tmp` work path plus public macOS system paths, never the publishing user's home/worktree. Each probe record must have an allowlisted category label, whether a target existed, the successful unsandboxed positive-control result where applicable, the sandboxed result, and exact denial errno/status; store only a SHA-256 of any publishing-machine absolute probe path so the deliverable report does not leak the local account.

Add independent checker tests for stale/mismatched `archive_sha256`, a missing or
extra report key, a malformed probe record, a non-`ok` gate, a changed profile
with stale digest, a correctly rehashed profile containing an extra allow rule,
and a report symlink. `scripts/check_offline_report.py` must recompute the
archive SHA-256 from an open descriptor, recompute the exact profile digest,
independently parse the small canonical SBPL subset and reject any operation or
allow path outside Task 11's fixed policy plus its single disposable work root,
strictly parse the complete report schema, require every gate and probe, and
exit nonzero on any mismatch. It must not import or call the verifier
implementation whose output it is checking.

Add malicious archive fixtures for `../` and absolute names, backslashes,
control characters, non-NFC names, Unicode/case-fold collisions, duplicate
members, symlink/special external modes, wrong member mode/size/hash, a missing
or extra manifest member, a ZIP-bomb size limit, and a symlinked/non-empty work
directory. Also record every application target-runtime subprocess and require
the exact same sandbox profile prefix on installs, imports, pip check,
self-test, TXT retrieval, RapidOCR, Vision, and the real MCP session. Add active
forbidden-access probes for system Python, Homebrew/MacPorts, Xcode/Command Line
Tools, the publishing worktree, and the real user's package/cache roots; each
probe must be denied when its target exists or the target must be confirmed
absent, while access to the extracted bundled runtime still succeeds.
Classify those as verifier-owned negative probes and exclude only that explicit
probe class from the application-command audit. Application commands must never
name or successfully execute a forbidden tool, even by absolute path; a probe
may name one but must fail before that executable starts. Treat the macOS base
`/usr/bin/codesign` binary as a narrowly scoped operating-system prerequisite,
not as Xcode or the separately installed Command Line Tools package. Add a
positive control proving it exists on the supported host without invoking
`xcrun`, and permit it in application-command audit records only for the exact
Vision snapshot verification argv `codesign --verify --strict SNAPSHOT`, where
`SNAPSHOT` is a release-hash-bound regular file beneath the disposable work
tree. Reject every other codesign argv, cwd, target, or caller context.

- [ ] **Step 2: Run and observe missing-module failure**

```bash
uv run pytest tests/test_verify_offline_release.py -v
```

Expected: collection ERROR because `scripts.verify_offline_release` does not exist.

- [ ] **Step 3: Implement the verifier**

The CLI accepts `--archive`, `--work-dir`, and `--report`. It must:

1. Require the archive and work-directory ancestors to be non-symlink regular file/directories, then verify the host is arm64 and macOS major 26 or newer for the current-machine candidate gate.
2. Before extraction, normalize every central-directory name as strict NFC POSIX relative text; reject backslashes, controls, absolute/traversing/duplicate/case-fold-colliding names, encryption, symlink/special modes, unexpected modes, multiple top levels, and expanded payload above 3 GiB. Verify ZIP CRC, parse the exact `RELEASE-MANIFEST.json` schema, require the central-directory set to equal its record paths plus the manifest itself, and stream-check every listed path/size/SHA-256/mode. The manifest is CRC-checked and schema-validated but is not self-hashed. Then validate the embedded Python, wheelhouse, requirements, RapidOCR, release, and model contracts against those same bytes.
3. Require a new empty real `work_dir`, create `extraction_root = work_dir / "离线 书库 验收 路径 含 空格 兼容性 测试 目录 长路径 001 002 003 004 005"`, and safely extract each verified regular member with descriptor-relative `O_NOFOLLOW`, exclusive creation, explicit mode, and post-write hash/identity recheck. Never call raw `extractall`.
4. Create clean transaction-local HOME/TMP/cache/empty-PATH directories and one reusable dynamic `sandbox-exec` profile. Keep `(allow default)` only for non-file/non-exec macOS operations, then apply default-deny rules for `network*`, `process-exec`, `file-read-data`, and `file-write*`. After those denies, allow process execution only for `/bin/bash`, `/bin/mkdir`, `/bin/rm`, `/usr/bin/env`, `/usr/bin/uname`, `/usr/bin/sw_vers`, `/usr/bin/shasum`, its `/usr/bin/perl` interpreter, `/usr/bin/mktemp`, the macOS base `/usr/bin/codesign`, and verified executable descendants of the safely extracted root. The verifier's application-command audit must accept codesign only as the real `VisionOcrEngine` child with exact argv `--verify --strict` and a verified private helper snapshot beneath the work tree; it must reject all other codesign uses. This is an already-present macOS system binary and must not be obtained through Xcode, `xcrun`, or a Command Line Tools installation; a missing binary fails the candidate instead of asking the recipient to install anything. Allow exact data read of the root vnode with `(literal "/")`—macOS 26 requires this even to launch an otherwise allowed system binary, and a literal rule does not expose descendants. Allow other file data reads only beneath the verifier work tree and narrowly enumerated immutable macOS base roots needed for the loader, frameworks, locale/timezone data, device files, and system Perl: `/System`, `/bin`, `/sbin`, `/usr/bin`, `/usr/sbin`, `/usr/lib`, `/usr/share`, `/etc`, `/private/etc`, `/private/var/db/timezone`, and `/dev`. Explicitly deny optional Python-framework/library roots after broader system allows if a rule would otherwise cover them. Allow writes only beneath the verifier work tree and required `/dev/null`/terminal devices. Do not allow the real HOME, original worktree/model cache, `/Applications`, third-party `/Library`, `/Network`, `/opt`, `/usr/local`, `/nix`, or `/sw`. Escape every literal path for the sandbox language and reject newline/control injection. Unit tests must compile the profile on macOS and actively prove one allowed executable/read/write plus every forbidden class, including the exact root-vnode rule, and exact allowed-versus-rejected codesign command records, so rule ordering and argv scoping are verified rather than inferred from string matching.
5. Run the launcher twice and every post-install application command beneath that exact wrapper. Include separately tagged verifier-owned socket/DNS and forbidden execution/read probes. Require each denial probe to succeed without the sandbox and fail under the sandbox with the expected denial status, so a missing file, broken tool, or ordinary DNS outage cannot be misreported as enforcement. Set `network_denied = "ok"` only when both network operations are positively controlled and denied. Set `external_dependencies = "ok"` only when the allowlist profile compiles, allowed control probes succeed, every existing forbidden target is denied, absent targets are recorded as absent, no application command names a forbidden executable, and both installs plus all runtime checks succeed. The verifier controller itself may perform only static archive/report/Mach-O auditing outside this target profile; `network_denied` and `external_dependencies` describe the recipient installation and runtime subprocess tree, whose descendants inherit the profile.
6. Parse generated `.codex/config.toml`, reject build-machine/user/transaction paths, require exact final Python/cwd/env values, and use those parsed values for the MCP launch.
7. Under the wrapper, run final exact CPython/path imports, cache-pinned `uv pip check`, `runtime_selftest.py`, and a TXT fixture containing `OFFLINE PASSAGE TOKEN 7319`. Import it through the real library API, run bounded keyword search for that token, pass the returned `passage_id` to `get_passages`, require the verified passage text, and only then set `txt_retrieval = "ok"`.
8. Under the wrapper, establish a real MCP stdio session from the generated config with initialize, protocol `tools/list`, and `tools/call` for `library_status`; do not substitute an in-process call. For the native Vision gate, use PyMuPDF to create a raster-only one-page PDF containing `VISION TEST 123`, then run the actual `book_agent.ocr.vision.VisionOcrEngine` with the final `bin/book-vision-ocr` and transaction-local temp root. Require the recorded child commands to contain exactly one allowed `/usr/bin/codesign --verify --strict` against the engine-created, release-digest-bound private helper snapshot before the helper execution, with no `xcrun` or developer-tool lookup. Require `recognize_page(..., page_index=0).ordered_text()` to contain the phrase. The helper accepts a rendered image and returns schema-2 JSON; it does not accept a PDF or create an OCR PDF, and a capabilities-only check is insufficient.
9. Reuse `scripts.macho_audit` to audit every Mach-O with the absolute macOS tools; require arm64, minimum macOS at most 14.0, no unresolved non-system load/RPATH dependency, and include signing status without misrepresenting ad-hoc or unsigned upstream wheels as Developer ID signed.
10. Reuse the bounded recursive archive scanner for SQLite headers, books, notes, actual publishing account/home/project paths, credential patterns, and forbidden nested names/content. Permit `/Users/runner` only inside an exact-hash trusted third-party member already proven by both release and asset manifests; tests must prove a real local path or nested secret is rejected and the pinned runner marker is allowed.
11. After all evidence is captured, identity-check and safely remove only the verifier-created extraction, HOME, cache, and test-data trees; reject symlink swaps and require `work_dir` to be empty afterward.
12. Atomically write a canonical UTF-8 JSON report only when every gate and cleanup succeeds. Include the exact bounded profile text, its SHA-256, and bounded per-probe evidence described in Step 1, but hash publishing-machine probe paths rather than disclosing them.

Implement `scripts/check_offline_report.py --archive PATH --report PATH` as the
independent strict checker from Step 1. It uses only the Python standard library
and shares no parser, constants, or success function with the verifier.

- [ ] **Step 4: Run verifier tests**

```bash
uv run pytest \
  tests/test_verify_offline_release.py tests/test_check_offline_report.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the verifier**

```bash
git add \
  scripts/verify_offline_release.py scripts/check_offline_report.py \
  tests/test_verify_offline_release.py tests/test_check_offline_report.py
git commit -m "test: verify macos release with network denied"
```

### Task 12: Build, test, and deliver the candidate ZIP

**Files:**
- Generate: `dist/codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline.zip`
- Generate: `dist/SHA256SUMS`
- Generate: `dist/offline-verification-report.json`
- Deliver: `<OUTPUT_DIR>/codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline.zip`
- Deliver: `<OUTPUT_DIR>/codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline-SHA256SUMS.txt`
- Deliver: `<OUTPUT_DIR>/codex-obsidian-book-library-v0.3.0-beta.1-offline-verification-report.json`

- [ ] **Step 1: Run the full clean baseline suite before building**

```bash
uv run pytest -m 'not macos_vision' -q
```

Expected: all selected tests pass with zero failures.

- [ ] **Step 2: Build and validate the current Vision helper**

```bash
mkdir -p dist/offline-assets/bin
uv run python scripts/build_vision_helper.py \
  --output dist/offline-assets/bin/book-vision-ocr
```

Expected: exit 0; `lipo -archs` prints only `arm64`; strict codesign verification and `--capabilities` succeed.

- [ ] **Step 3: Build the deterministic offline ZIP**

```bash
set -e
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:?Set MODEL_SNAPSHOT to the verified model snapshot first}"
uv run python -m scripts.prepare_offline_assets \
  --project-root "$PWD" \
  --uv "$(command -v uv)" \
  --python-archive 'dist/offline-assets/python-mirror/20251007/cpython-3.12.11+20251007-aarch64-apple-darwin-install_only_stripped.tar.gz' \
  --output-dir dist/offline-assets \
  --contract-dir distribution \
  --dependency-license-manifest third_party/python-dependencies/manifest.json \
  --verify-only
uv run python scripts/build_macos_release.py \
  --project-root "$PWD" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --uv-binary "$(command -v uv)" \
  --vision-helper "$PWD/dist/offline-assets/bin/book-vision-ocr" \
  --output-dir "$PWD/dist" \
  --metadata "$PWD/distribution/offline-release.json" \
  --model-manifest "$PWD/distribution/model-manifest.json" \
  --python-root "$PWD/dist/offline-assets/python-mirror" \
  --python-manifest "$PWD/distribution/python-manifest.json" \
  --wheelhouse-root "$PWD/dist/offline-assets/wheelhouse" \
  --wheelhouse-manifest "$PWD/distribution/wheelhouse-manifest.json" \
  --requirements "$PWD/distribution/requirements-macos-arm64-py312.txt" \
  --rapidocr-model-manifest "$PWD/distribution/rapidocr-model-manifest.json" \
  --dependency-license-manifest "$PWD/third_party/python-dependencies/manifest.json"
cp dist/SHA256SUMS dist/.first-offline-SHA256SUMS
uv run python scripts/build_macos_release.py \
  --project-root "$PWD" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --uv-binary "$(command -v uv)" \
  --vision-helper "$PWD/dist/offline-assets/bin/book-vision-ocr" \
  --output-dir "$PWD/dist" \
  --metadata "$PWD/distribution/offline-release.json" \
  --model-manifest "$PWD/distribution/model-manifest.json" \
  --python-root "$PWD/dist/offline-assets/python-mirror" \
  --python-manifest "$PWD/distribution/python-manifest.json" \
  --wheelhouse-root "$PWD/dist/offline-assets/wheelhouse" \
  --wheelhouse-manifest "$PWD/distribution/wheelhouse-manifest.json" \
  --requirements "$PWD/distribution/requirements-macos-arm64-py312.txt" \
  --rapidocr-model-manifest "$PWD/distribution/rapidocr-model-manifest.json" \
  --dependency-license-manifest "$PWD/third_party/python-dependencies/manifest.json"
/usr/bin/cmp dist/.first-offline-SHA256SUMS dist/SHA256SUMS
```

Expected: the offline ZIP and matching `SHA256SUMS` are atomically published; `/usr/bin/cmp` exits 0 after the second identical build.

- [ ] **Step 4: Run the hard-network-denied integration gate**

Run this command with approved host execution outside Codex's outer workspace
sandbox so Apple Vision can access its normal macOS XPC services; the verifier's
own `sandbox-exec` profile must still be the layer that denies all network access.

```bash
set -e
OFFLINE_VERIFY_WORK_DIR="$(/usr/bin/mktemp -d /private/tmp/codex-book-library-offline-verification.XXXXXX)"
/bin/rm -f "$PWD/dist/offline-verification-report.json"
UV_OFFLINE=1 uv run --offline --frozen --no-sync python \
  scripts/verify_offline_release.py \
  --archive "$PWD/dist/codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline.zip" \
  --work-dir "$OFFLINE_VERIFY_WORK_DIR" \
  --report "$PWD/dist/offline-verification-report.json"
/bin/rmdir "$OFFLINE_VERIFY_WORK_DIR"
UV_OFFLINE=1 uv run --offline --frozen --no-sync python \
  scripts/check_offline_report.py \
  --archive "$PWD/dist/codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline.zip" \
  --report "$PWD/dist/offline-verification-report.json"
```

Expected: first and second installs pass while network and every external
developer/package environment are denied; `external_dependencies`, imports,
384-dimensional embedding, RapidOCR, Vision, MCP, TXT retrieval, privacy scan,
and Mach-O audit all report `ok`; the independent checker recomputes the ZIP
hash and accepts the newly created report.

- [ ] **Step 5: Exercise the real quarantine, Finder, and Gatekeeper path**

Use a disposable directory and the exact already-hashed ZIP; do not rebuild or
modify the candidate. Create a copy, attach a browser-style
`com.apple.quarantine` value with build-machine `/usr/bin/xattr`, open that copy
with Archive Utility through Finder, and use the `computer-use` skill to inspect
the resulting UI and first-launch flow. Verify the extracted
`install-macos.command`, `bin/uv`, and `bin/book-vision-ocr` retain quarantine.
Attempt the ordinary Finder launch first. If Gatekeeper blocks it because this
candidate is not Developer ID notarized, use only Finder's context-menu
**Open** followed by the standard macOS confirmation; never delete quarantine,
call `spctl`, or weaken security settings. Require the installer to reach its
success message and the resulting config/runtime to pass the final self-test.
Identity-check and remove only the disposable QA tree afterward.

Expected: the same ZIP installs successfully through the real current-macOS 26
Archive Utility/Gatekeeper flow. Record whether ordinary open succeeded or the
documented one-time context-menu confirmation was required. If the official
Open flow cannot start the installer, stop and do not deliver the artifact.

- [ ] **Step 6: Run final verification after artifact generation**

```bash
set -e
unzip -tq dist/codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline.zip
cd dist
/usr/bin/shasum -a 256 -c SHA256SUMS
cd ..
uv run pytest -m 'not macos_vision' -q
git status --short
```

Expected: ZIP CRC clean, SHA matches `dist/SHA256SUMS`, all selected tests pass, and only ignored build artifacts remain outside committed source changes.

- [ ] **Step 7: Copy the verified deliverables**

Use normal file-copy commands, preserving the ZIP bytes exactly:

```bash
set -e
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/release-output}"
mkdir -p "$OUTPUT_DIR"
cp \
  dist/codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline.zip \
  "$OUTPUT_DIR/"
cp dist/SHA256SUMS \
  "$OUTPUT_DIR/codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline-SHA256SUMS.txt"
cp dist/offline-verification-report.json \
  "$OUTPUT_DIR/codex-obsidian-book-library-v0.3.0-beta.1-offline-verification-report.json"
cd "$OUTPUT_DIR"
/usr/bin/shasum -a 256 -c \
  codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline-SHA256SUMS.txt
```

Require the final checksum command to exit 0 before reporting completion.

- [ ] **Step 8: Record the compatibility limitation in the handoff**

Report the fresh macOS 26 hard-offline evidence and the exact Finder/Gatekeeper
result, including whether the one-time official context-menu confirmation was
needed. State that macOS 14 compatibility is supported by deployment-target and
architecture audits but remains a release-candidate claim until the same ZIP
passes on a clean macOS 14 Apple Silicon machine or VM with no Homebrew,
Python, uv, pip, Xcode, or Command Line Tools and with its virtual NIC detached.
Also state that the macOS 26 verifier's Mach-O controller uses build-host audit
tools and is not itself a no-CLT recipient utility; the recipient application
subtree is the part proven isolated. Do not claim real-device coverage for
every M-series generation without those machines.
