"""Download and atomically publish the pinned macOS arm64 Node.js runtime."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, ContextManager


NODE_RUNTIME_VERSION = "24.15.0"
NODE_RUNTIME_ARCHIVE = f"node-v{NODE_RUNTIME_VERSION}-darwin-arm64.tar.gz"
NODE_RUNTIME_TOP_LEVEL = f"node-v{NODE_RUNTIME_VERSION}-darwin-arm64"
NODE_RUNTIME_URL = (
    f"https://nodejs.org/dist/v{NODE_RUNTIME_VERSION}/{NODE_RUNTIME_ARCHIVE}"
)
NODE_RUNTIME_SHA256 = (
    "372331b969779ab5d15b949884fc6eaf88d5afe87bde8ba881d6400b9100ffc4"
)
MAX_ARCHIVE_BYTES = 96 * 1024 * 1024


class NodeRuntimeError(RuntimeError):
    """The pinned Node.js runtime could not be downloaded or verified safely."""


def _validate_runtime(root: Path) -> None:
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise NodeRuntimeError(f"Node.js 运行目录不存在：{root}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise NodeRuntimeError(f"Node.js 运行目录无效：{root}")

    node = root / "bin" / "node"
    npm = root / "bin" / "npm"
    try:
        node_info = node.lstat()
        npm_info = npm.lstat()
    except OSError as exc:
        raise NodeRuntimeError("Node.js 运行时缺少 node 或 npm。") from exc
    if not stat.S_ISREG(node_info.st_mode) or not os.access(node, os.X_OK):
        raise NodeRuntimeError("Node.js 运行时中的 node 无效或不可执行。")
    if not (stat.S_ISREG(npm_info.st_mode) or stat.S_ISLNK(npm_info.st_mode)):
        raise NodeRuntimeError("Node.js 运行时中的 npm 无效。")
    try:
        npm_target = npm.resolve(strict=True)
        npm_target.relative_to(root.resolve(strict=True))
        target_info = npm_target.lstat()
    except (OSError, ValueError) as exc:
        raise NodeRuntimeError("Node.js 运行时中的 npm 链接越界或损坏。") from exc
    if not stat.S_ISREG(target_info.st_mode) or not os.access(npm_target, os.X_OK):
        raise NodeRuntimeError("Node.js 运行时中的 npm 不可执行。")


def _download(
    destination: Path,
    *,
    url: str,
    opener: Callable[..., ContextManager[BinaryIO]],
) -> str:
    digest = hashlib.sha256()
    total = 0
    try:
        with opener(url, timeout=30) as response, destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise NodeRuntimeError("Node.js 下载包超过大小上限。")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except NodeRuntimeError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise NodeRuntimeError(f"Node.js 下载失败：{exc}") from exc
    if total == 0:
        raise NodeRuntimeError("Node.js 下载包为空。")
    return digest.hexdigest()


def ensure_node_runtime(
    project_root: Path,
    *,
    url: str = NODE_RUNTIME_URL,
    sha256: str = NODE_RUNTIME_SHA256,
    opener: Callable[..., ContextManager[BinaryIO]] = urllib.request.urlopen,
) -> Path:
    """Return a verified runtime, downloading the pinned archive when absent."""

    if not isinstance(project_root, Path):
        raise ValueError("project_root must be a Path")
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("sha256 must be a lowercase SHA-256 digest")

    runtime_parent = project_root / "runtime"
    destination = runtime_parent / "node"
    if destination.exists() or destination.is_symlink():
        _validate_runtime(destination)
        return destination

    try:
        runtime_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".node-runtime-", dir=runtime_parent
        ) as temporary:
            temporary_root = Path(temporary)
            archive_path = temporary_root / NODE_RUNTIME_ARCHIVE
            actual_sha256 = _download(archive_path, url=url, opener=opener)
            if actual_sha256 != sha256:
                raise NodeRuntimeError("Node.js 下载包 SHA-256 校验失败。")
            extraction_root = temporary_root / "extracted"
            extraction_root.mkdir()
            try:
                with tarfile.open(archive_path, mode="r:gz") as archive:
                    archive.extractall(extraction_root, filter="data")
            except (OSError, tarfile.TarError) as exc:
                raise NodeRuntimeError(f"Node.js 下载包解压失败：{exc}") from exc
            candidate = extraction_root / NODE_RUNTIME_TOP_LEVEL
            _validate_runtime(candidate)
            if destination.exists() or destination.is_symlink():
                raise NodeRuntimeError("Node.js 运行目录在安装期间发生变化。")
            os.replace(candidate, destination)
    except NodeRuntimeError:
        raise
    except OSError as exc:
        raise NodeRuntimeError(f"Node.js 运行时安装失败：{exc}") from exc

    _validate_runtime(destination)
    return destination


__all__ = [
    "NODE_RUNTIME_ARCHIVE",
    "NODE_RUNTIME_SHA256",
    "NODE_RUNTIME_TOP_LEVEL",
    "NODE_RUNTIME_URL",
    "NODE_RUNTIME_VERSION",
    "NodeRuntimeError",
    "ensure_node_runtime",
]
