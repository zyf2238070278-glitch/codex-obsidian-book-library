from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from installer import node_runtime


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

    node_root = node_runtime.ensure_node_runtime(
        tmp_path,
        url="https://example.invalid/node.tar.gz",
        sha256=expected_sha256,
        opener=lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    assert node_root == tmp_path / "runtime" / "node"
    assert (node_root / "bin" / "node").read_bytes() == b"node"
    assert (node_root / "bin" / "node").stat().st_mode & 0o111
    assert (node_root / "bin" / "npm").resolve().read_bytes() == b"npm"


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
