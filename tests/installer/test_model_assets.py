from __future__ import annotations

import hashlib
import json
import os
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
                "files": [
                    {
                        "path": "config.json",
                        "size": len(FIXTURE_BYTES),
                        "sha256": _digest(FIXTURE_BYTES),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if create_files:
        populate_model_fixture(snapshot)
    return model_root, manifest, snapshot


def _manifest_data(manifest: Path) -> dict[str, object]:
    return json.loads(manifest.read_text(encoding="utf-8"))


def _write_manifest(manifest: Path, data: dict[str, object]) -> None:
    manifest.write_text(json.dumps(data), encoding="utf-8")


def _open_test_root(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY)


def test_fd_hash_stays_bound_to_opened_regular_file_after_path_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    original = root / "model.bin"
    original.write_bytes(FIXTURE_BYTES)
    root_fd = _open_test_root(root)
    file_fd = -1
    try:
        file_fd = model_assets._open_regular_file_fd(root_fd, "model.bin")
        original.rename(root / "opened-inode.bin")
        original.write_bytes(b"X" * len(FIXTURE_BYTES))

        model_assets._validate_file_descriptor(
            file_fd,
            "model.bin",
            len(FIXTURE_BYTES),
            _digest(FIXTURE_BYTES),
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)


def test_fd_traversal_rejects_ancestor_swapped_to_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ancestor = root / "weights"
    ancestor.mkdir(parents=True)
    (ancestor / "model.bin").write_bytes(FIXTURE_BYTES)
    root_fd = _open_test_root(root)
    try:
        ancestor.rename(root / "real-weights")
        ancestor.symlink_to("real-weights", target_is_directory=True)

        with pytest.raises(model_assets.ModelAssetError, match="目录|符号链接"):
            model_assets._open_regular_file_fd(root_fd, "weights/model.bin")
    finally:
        os.close(root_fd)


def test_fd_hash_stays_bound_after_relative_symlink_target_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    blobs = root / "blobs"
    snapshot = root / "snapshots" / REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    target = blobs / "original"
    target.write_bytes(FIXTURE_BYTES)
    (snapshot / "config.json").symlink_to("../../blobs/original")
    root_fd = _open_test_root(root)
    file_fd = -1
    try:
        file_fd = model_assets._open_regular_file_fd(
            root_fd, f"snapshots/{REVISION}/config.json"
        )
        target.rename(blobs / "opened-inode")
        target.write_bytes(b"X" * len(FIXTURE_BYTES))

        model_assets._validate_file_descriptor(
            file_fd,
            "config.json",
            len(FIXTURE_BYTES),
            _digest(FIXTURE_BYTES),
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)


def test_fd_traversal_opens_normal_hugging_face_relative_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    blob = root / "models--fixture" / "blobs" / "digest"
    snapshot = root / "models--fixture" / "snapshots" / REVISION
    blob.parent.mkdir(parents=True)
    blob.write_bytes(FIXTURE_BYTES)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").symlink_to("../../blobs/digest")
    root_fd = _open_test_root(root)
    file_fd = -1
    try:
        file_fd = model_assets._open_regular_file_fd(
            root_fd,
            f"models--fixture/snapshots/{REVISION}/config.json",
        )
        model_assets._validate_file_descriptor(
            file_fd,
            "config.json",
            len(FIXTURE_BYTES),
            _digest(FIXTURE_BYTES),
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)


def test_snapshot_validation_does_not_reopen_listed_path_after_fd_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    original = model_assets._validate_snapshot_and_files_fd

    def validate_then_replace(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        listed = snapshot / "config.json"
        listed.rename(snapshot / "validated-inode.json")
        listed.write_bytes(b"X" * len(FIXTURE_BYTES))

    monkeypatch.setattr(
        model_assets, "_validate_snapshot_and_files_fd", validate_then_replace
    )

    assert model_assets.validate_model(
        model_root=model_root, manifest_path=manifest
    ) == snapshot


def test_ensure_model_reuses_valid_snapshot_without_network(tmp_path: Path) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    calls: list[dict[str, object]] = []

    result = model_assets.ensure_model(
        model_root=model_root,
        manifest_path=manifest,
        snapshot_download=lambda **kwargs: calls.append(kwargs) or str(snapshot),
    )

    assert result == snapshot
    assert calls == []


def test_absent_snapshot_downloads_exact_revision_and_validates(
    tmp_path: Path,
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=False)
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        populate_model_fixture(snapshot)
        return str(snapshot)

    assert model_assets.ensure_model(
        model_root=model_root,
        manifest_path=manifest,
        snapshot_download=download,
    ) == snapshot
    assert calls == [
        {
            "repo_id": model_assets.MODEL_ID,
            "revision": model_assets.MODEL_REVISION,
            "cache_dir": str(model_root),
            "allow_patterns": ["config.json"],
        }
    ]


def test_downloader_allow_patterns_are_exact_regular_manifest_paths(
    tmp_path: Path,
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=False)
    data = _manifest_data(manifest)
    data["files"].append(  # type: ignore[union-attr]
        {
            "path": "nested/tokenizer.json",
            "size": len(FIXTURE_BYTES),
            "sha256": _digest(FIXTURE_BYTES),
        }
    )
    _write_manifest(manifest, data)
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        populate_model_fixture(snapshot)
        (snapshot / "nested").mkdir()
        (snapshot / "nested" / "tokenizer.json").write_bytes(FIXTURE_BYTES)
        return str(snapshot)

    model_assets.ensure_model(
        model_root=model_root,
        manifest_path=manifest,
        snapshot_download=download,
    )

    assert calls[0]["allow_patterns"] == [
        "config.json",
        "nested/tokenizer.json",
    ]
    assert all(
        "*" not in pattern and not pattern.endswith("/")
        for pattern in calls[0]["allow_patterns"]
    )
    assert manifest.name not in calls[0]["allow_patterns"]


def test_existing_snapshot_with_extra_directory_is_rebuilt_cleanly(
    tmp_path: Path,
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    extra = snapshot / "onnx" / "model.onnx"
    extra.parent.mkdir()
    extra.write_bytes(b"unlisted")
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        clean_snapshot = model_assets._expected_snapshot(Path(str(kwargs["cache_dir"])))
        populate_model_fixture(clean_snapshot)
        return str(clean_snapshot)

    assert model_assets.ensure_model(
        model_root=model_root,
        manifest_path=manifest,
        snapshot_download=download,
    ) == snapshot
    assert calls[0]["cache_dir"] == str(model_root)
    assert calls[0]["allow_patterns"] == ["config.json"]
    assert not (snapshot / "onnx").exists()


def test_failed_rebuild_restores_existing_model_cache(tmp_path: Path) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    extra = snapshot / "onnx" / "model.onnx"
    extra.parent.mkdir()
    extra.write_bytes(b"keep-this-cache")

    def fail_download(**kwargs: object) -> str:
        replacement = Path(str(kwargs["cache_dir"])) / "partial-download"
        replacement.mkdir(parents=True)
        (replacement / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("network stopped")

    with pytest.raises(model_assets.ModelAssetError, match="下载固定语义模型失败"):
        model_assets.ensure_model(
            model_root=model_root,
            manifest_path=manifest,
            snapshot_download=fail_download,
        )

    assert extra.read_bytes() == b"keep-this-cache"
    assert not (model_root / "partial-download").exists()
    assert not list(model_root.parent.glob(f".{model_root.name}.backup-*"))


def test_restore_failure_preserves_the_only_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    extra = snapshot / "onnx" / "model.onnx"
    extra.parent.mkdir()
    extra.write_bytes(b"only-copy")
    original_replace = model_assets.os.replace
    replace_calls = 0

    def fail_second_replace(source: object, destination: object) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise PermissionError("restore blocked")
        original_replace(source, destination)

    monkeypatch.setattr(model_assets.os, "replace", fail_second_replace)

    with pytest.raises(model_assets.ModelAssetError, match="无法恢复旧缓存"):
        model_assets.ensure_model(
            model_root=model_root,
            manifest_path=manifest,
            snapshot_download=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("network stopped")
            ),
        )

    backups = list(model_root.parent.glob(f".{model_root.name}.backup-*"))
    assert len(backups) == 1
    assert (
        backups[0]
        / "original"
        / snapshot.relative_to(model_root)
        / "onnx"
        / "model.onnx"
    ).read_bytes() == b"only-copy"


def test_directory_identity_swap_is_rejected_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    (snapshot / "unlisted.bin").write_bytes(b"force-rebuild")
    outside = tmp_path / "outside"
    outside.mkdir()
    displaced = tmp_path / "displaced-models"
    calls: list[dict[str, object]] = []
    original_replace = model_assets.os.replace
    swapped = False

    def swap_before_isolation(source: object, destination: object) -> None:
        nonlocal swapped
        if not swapped and Path(source) == model_root:
            swapped = True
            model_root.rename(displaced)
            model_root.symlink_to(outside, target_is_directory=True)
        original_replace(source, destination)

    monkeypatch.setattr(model_assets.os, "replace", swap_before_isolation)

    with pytest.raises(model_assets.ModelAssetError, match="隔离期间发生变化"):
        model_assets.ensure_model(
            model_root=model_root,
            manifest_path=manifest,
            snapshot_download=lambda **kwargs: calls.append(kwargs) or "unused",
        )

    assert calls == []
    assert model_root.is_symlink()
    assert list(outside.iterdir()) == []
    assert (displaced / "models--intfloat--multilingual-e5-small").is_dir()


def test_lstat_failure_after_isolation_restores_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    extra = snapshot / "onnx" / "model.onnx"
    extra.parent.mkdir()
    extra.write_bytes(b"only-copy")
    original_lstat = model_assets.Path.lstat

    def fail_backup_lstat(path: Path) -> os.stat_result:
        if path.name == "original" and path.parent.name.startswith(".models.backup-"):
            raise PermissionError("cannot inspect isolated cache")
        return original_lstat(path)

    monkeypatch.setattr(model_assets.Path, "lstat", fail_backup_lstat)
    calls: list[dict[str, object]] = []

    with pytest.raises(model_assets.ModelAssetError, match="无法检查已隔离"):
        model_assets.ensure_model(
            model_root=model_root,
            manifest_path=manifest,
            snapshot_download=lambda **kwargs: calls.append(kwargs) or "unused",
        )

    assert calls == []
    assert extra.read_bytes() == b"only-copy"
    assert not list(model_root.parent.glob(f".{model_root.name}.backup-*"))


def test_unsafe_symlink_model_root_is_rejected_without_downloading(
    tmp_path: Path,
) -> None:
    model_root, manifest, _ = make_model_fixture(tmp_path, create_files=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    model_root.parent.mkdir(parents=True, exist_ok=True)
    model_root.symlink_to(outside, target_is_directory=True)
    calls: list[dict[str, object]] = []

    with pytest.raises(model_assets.ModelAssetError, match="不安全"):
        model_assets.ensure_model(
            model_root=model_root,
            manifest_path=manifest,
            snapshot_download=lambda **kwargs: calls.append(kwargs) or "unused",
        )

    assert calls == []
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "缺失"),
        ("size", "大小"),
        ("hash", "SHA-256"),
        ("extra", "额外"),
    ],
)
def test_validate_rejects_invalid_snapshot_content(
    tmp_path: Path, mutation: str, match: str
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    target = snapshot / "config.json"
    if mutation == "missing":
        target.unlink()
    elif mutation == "size":
        target.write_bytes(FIXTURE_BYTES + b"x")
    elif mutation == "hash":
        target.write_bytes(b"X" + FIXTURE_BYTES[1:])
    else:
        (snapshot / "unlisted.bin").write_bytes(b"extra")

    with pytest.raises(model_assets.ModelAssetError, match=match):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)


@pytest.mark.parametrize(
    "target", ["/tmp/model", "../../../../outside-model", "config.json"]
)
def test_validate_rejects_absolute_or_escaping_symlink(
    tmp_path: Path, target: str
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=False)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").symlink_to(target)

    with pytest.raises(model_assets.ModelAssetError, match="符号链接"):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)


def test_validate_accepts_hugging_face_relative_symlink_and_cache_metadata(
    tmp_path: Path,
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=False)
    blob = model_root / "models--intfloat--multilingual-e5-small" / "blobs" / "fixture"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(FIXTURE_BYTES)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").symlink_to("../../blobs/fixture")
    cache = snapshot / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "config.json.metadata").write_text("metadata", encoding="utf-8")

    assert model_assets.validate_model(
        model_root=model_root, manifest_path=manifest
    ) == snapshot


def test_validate_rejects_symlink_disguised_as_cache_metadata(tmp_path: Path) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    (snapshot / ".cache").symlink_to(outside, target_is_directory=True)

    with pytest.raises(model_assets.ModelAssetError, match="cache.*不安全"):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.__setitem__("extra", True),
        lambda data: data["files"][0].__setitem__("extra", True),
        lambda data: data.__setitem__("model_id", "intfloat/other"),
        lambda data: data.__setitem__("model_revision", "0" * 40),
        lambda data: data["files"][0].__setitem__("path", "/config.json"),
        lambda data: data["files"][0].__setitem__("path", "a/../config.json"),
        lambda data: data["files"][0].__setitem__("path", "bad\\name"),
        lambda data: data["files"][0].__setitem__("path", "bad\u0000name"),
        lambda data: data["files"][0].__setitem__("sha256", "A" * 64),
        lambda data: data["files"][0].__setitem__("sha256", "a" * 63),
        lambda data: data["files"][0].__setitem__("size", 0),
        lambda data: data["files"][0].__setitem__("size", True),
    ],
)
def test_validate_rejects_invalid_manifest_schema_or_values(
    tmp_path: Path, mutate: object
) -> None:
    model_root, manifest, _ = make_model_fixture(tmp_path, create_files=True)
    data = _manifest_data(manifest)
    mutate(data)  # type: ignore[operator]
    _write_manifest(manifest, data)

    with pytest.raises(model_assets.ModelAssetError, match="清单"):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)


@pytest.mark.parametrize("second_path", ["config.json", "CONFIG.JSON"])
def test_validate_rejects_duplicate_or_casefold_collision(
    tmp_path: Path, second_path: str
) -> None:
    model_root, manifest, _ = make_model_fixture(tmp_path, create_files=True)
    data = _manifest_data(manifest)
    data["files"].append(  # type: ignore[union-attr]
        {
            "path": second_path,
            "size": len(FIXTURE_BYTES),
            "sha256": _digest(FIXTURE_BYTES),
        }
    )
    _write_manifest(manifest, data)

    with pytest.raises(model_assets.ModelAssetError, match="冲突|重复"):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)


def test_validate_rejects_casefold_collision_in_path_ancestors(tmp_path: Path) -> None:
    model_root, manifest, _ = make_model_fixture(tmp_path, create_files=True)
    data = _manifest_data(manifest)
    files = data["files"]
    files[0]["path"] = "Weights/config.json"  # type: ignore[index]
    files.append(  # type: ignore[union-attr]
        {
            "path": "weights/tokenizer.json",
            "size": len(FIXTURE_BYTES),
            "sha256": _digest(FIXTURE_BYTES),
        }
    )
    _write_manifest(manifest, data)

    with pytest.raises(model_assets.ModelAssetError, match="大小写冲突"):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("a", "a/b"),
        ("a/b", "a"),
        ("A", "a/b"),
        ("a/b", "A"),
    ],
    ids=[
        "file-before-child",
        "child-before-file",
        "casefold-file-before-child",
        "casefold-child-before-file",
    ],
)
def test_validate_rejects_file_directory_ancestor_collisions_in_either_order(
    tmp_path: Path, first: str, second: str
) -> None:
    model_root, manifest, _ = make_model_fixture(tmp_path, create_files=True)
    data = _manifest_data(manifest)
    data["files"] = [
        {
            "path": path,
            "size": len(FIXTURE_BYTES),
            "sha256": _digest(FIXTURE_BYTES),
        }
        for path in (first, second)
    ]
    _write_manifest(manifest, data)

    with pytest.raises(model_assets.ModelAssetError, match="清单.*祖先冲突"):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)


def test_validate_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    model_root, manifest, _ = make_model_fixture(tmp_path, create_files=True)
    manifest.write_text(
        '{"model_id":"intfloat/multilingual-e5-small",'
        '"model_id":"intfloat/multilingual-e5-small",'
        f'"model_revision":"{REVISION}",'
        '"files":[{"path":"config.json","size":20,'
        f'"sha256":"{_digest(FIXTURE_BYTES)}"}}]}}',
        encoding="utf-8",
    )

    with pytest.raises(model_assets.ModelAssetError, match="清单.*重复"):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)


def test_downloader_must_return_exact_expected_snapshot(tmp_path: Path) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=False)

    def download_wrong_path(**_: object) -> str:
        populate_model_fixture(snapshot)
        return str(snapshot.parent / "other")

    with pytest.raises(model_assets.ModelAssetError, match="固定.*快照"):
        model_assets.ensure_model(
            model_root=model_root,
            manifest_path=manifest,
            snapshot_download=download_wrong_path,
        )


def test_cli_success_and_failure_have_chinese_context(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=True)
    assert model_assets.main(
        ["--model-root", str(model_root), "--manifest", str(manifest)]
    ) == 0
    success = capsys.readouterr()
    assert "语义模型" in success.out
    assert str(snapshot) in success.out
    assert success.err == ""

    (snapshot / "config.json").unlink()
    assert model_assets.main(
        ["--model-root", str(model_root), "--manifest", str(manifest)],
        snapshot_download=lambda **_: (_ for _ in ()).throw(OSError("offline")),
    ) != 0
    failure = capsys.readouterr()
    assert failure.out == ""
    assert "语义模型" in failure.err
    assert "失败" in failure.err


def test_validate_rejects_special_file(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO is unavailable")
    model_root, manifest, snapshot = make_model_fixture(tmp_path, create_files=False)
    snapshot.mkdir(parents=True)
    os.mkfifo(snapshot / "config.json")

    with pytest.raises(model_assets.ModelAssetError, match="特殊文件"):
        model_assets.validate_model(model_root=model_root, manifest_path=manifest)
