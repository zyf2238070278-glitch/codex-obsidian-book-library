from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest

from installer import node_runtime


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()

    def add_field(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    pending = [root]
    while pending:
        directory = pending.pop()
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        child_directories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix().encode("utf-8")
            if entry.is_symlink():
                digest.update(b"L")
                add_field(relative)
                add_field(os.readlink(path).encode("utf-8"))
            elif entry.is_dir(follow_symlinks=False):
                digest.update(b"D")
                add_field(relative)
                child_directories.append(path)
            elif entry.is_file(follow_symlinks=False):
                digest.update(b"F")
                add_field(relative)
                size = path.stat(follow_symlinks=False).st_size
                digest.update(size.to_bytes(8, "big"))
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
            else:  # pragma: no cover - fixture never creates special files
                raise AssertionError(f"unexpected fixture entry: {path}")
        pending.extend(reversed(child_directories))
    return digest.hexdigest()


def _expected_tree(tmp_path: Path) -> Path:
    root = tmp_path / "expected-node"
    node = root / "bin" / "node"
    npm_cli = root / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    node.parent.mkdir(parents=True)
    npm_cli.parent.mkdir(parents=True)
    node.write_bytes(b"node")
    node.chmod(0o755)
    npm_cli.write_bytes(b"npm")
    npm_cli.chmod(0o755)
    (root / "bin" / "npm").symlink_to("../lib/node_modules/npm/bin/npm-cli.js")
    return root


def _archive_bytes(*, unsafe: bool = False) -> bytes:
    payload = io.BytesIO()
    prefix = node_runtime.NODE_RUNTIME_TOP_LEVEL
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        node = tarfile.TarInfo(f"{prefix}/bin/node")
        node.mode = 0o755
        node.size = len(b"node")
        archive.addfile(node, io.BytesIO(b"node"))

        npm_cli = tarfile.TarInfo(
            f"{prefix}/lib/node_modules/npm/bin/npm-cli.js"
        )
        npm_cli.mode = 0o755
        npm_cli.size = len(b"npm")
        archive.addfile(npm_cli, io.BytesIO(b"npm"))

        npm = tarfile.TarInfo(f"{prefix}/bin/npm")
        npm.type = tarfile.SYMTYPE
        npm.linkname = "../lib/node_modules/npm/bin/npm-cli.js"
        archive.addfile(npm)

        if unsafe:
            escaped = tarfile.TarInfo(f"{prefix}/../../escape")
            escaped.size = len(b"escape")
            archive.addfile(escaped, io.BytesIO(b"escape"))
    return payload.getvalue()


def test_ensure_node_runtime_verifies_and_publishes_pinned_archive(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    expected_tree_sha256 = _tree_digest(_expected_tree(tmp_path))

    node_root = node_runtime.ensure_node_runtime(
        tmp_path,
        url="https://example.invalid/node.tar.gz",
        sha256=expected_sha256,
        tree_sha256=expected_tree_sha256,
        opener=lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    assert node_root == tmp_path / "runtime" / "node"
    assert (node_root / "bin" / "node").read_bytes() == b"node"
    assert (node_root / "bin" / "node").stat().st_mode & 0o111
    assert (node_root / "bin" / "npm").resolve().read_bytes() == b"npm"


def test_ensure_node_runtime_repairs_mutated_existing_runtime(tmp_path: Path) -> None:
    payload = _archive_bytes()
    archive_sha256 = hashlib.sha256(payload).hexdigest()
    tree_sha256 = _tree_digest(_expected_tree(tmp_path))
    arguments = {
        "url": "https://example.invalid/node.tar.gz",
        "sha256": archive_sha256,
        "tree_sha256": tree_sha256,
        "opener": lambda *_args, **_kwargs: io.BytesIO(payload),
    }
    root = node_runtime.ensure_node_runtime(tmp_path, **arguments)
    (root / "bin" / "node").write_bytes(b"tampered but runnable")
    (root / "bin" / "node").chmod(0o755)

    repaired = node_runtime.ensure_node_runtime(tmp_path, **arguments)

    assert repaired == root
    assert (repaired / "bin" / "node").read_bytes() == b"node"


def test_failed_runtime_repair_preserves_existing_tree(tmp_path: Path) -> None:
    payload = _archive_bytes()
    archive_sha256 = hashlib.sha256(payload).hexdigest()
    tree_sha256 = _tree_digest(_expected_tree(tmp_path))
    root = node_runtime.ensure_node_runtime(
        tmp_path,
        url="https://example.invalid/node.tar.gz",
        sha256=archive_sha256,
        tree_sha256=tree_sha256,
        opener=lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    node = root / "bin" / "node"
    node.write_bytes(b"existing damaged runtime")
    node.chmod(0o755)

    with pytest.raises(node_runtime.NodeRuntimeError, match="下载失败"):
        node_runtime.ensure_node_runtime(
            tmp_path,
            url="https://example.invalid/node.tar.gz",
            sha256=archive_sha256,
            tree_sha256=tree_sha256,
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
        )

    assert node.read_bytes() == b"existing damaged runtime"


def test_ensure_node_runtime_rejects_symlinked_runtime_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "runtime").symlink_to(outside, target_is_directory=True)
    payload = _archive_bytes()

    with pytest.raises(node_runtime.NodeRuntimeError, match="runtime.*符号链接"):
        node_runtime.ensure_node_runtime(
            tmp_path,
            url="https://example.invalid/node.tar.gz",
            sha256=hashlib.sha256(payload).hexdigest(),
            tree_sha256=_tree_digest(_expected_tree(tmp_path)),
            opener=lambda *_args, **_kwargs: io.BytesIO(payload),
        )

    assert list(outside.iterdir()) == []


def test_ensure_node_runtime_rejects_checksum_mismatch(tmp_path: Path) -> None:
    payload = _archive_bytes()

    with pytest.raises(node_runtime.NodeRuntimeError, match="SHA-256"):
        node_runtime.ensure_node_runtime(
            tmp_path,
            url="https://example.invalid/node.tar.gz",
            sha256="0" * 64,
            opener=lambda *_args, **_kwargs: io.BytesIO(payload),
        )

    assert not (tmp_path / "runtime" / "node").exists()


def test_ensure_node_runtime_rejects_archive_escape(tmp_path: Path) -> None:
    payload = _archive_bytes(unsafe=True)

    with pytest.raises(node_runtime.NodeRuntimeError, match="解压"):
        node_runtime.ensure_node_runtime(
            tmp_path,
            url="https://example.invalid/node.tar.gz",
            sha256=hashlib.sha256(payload).hexdigest(),
            opener=lambda *_args, **_kwargs: io.BytesIO(payload),
        )

    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "runtime" / "node").exists()
