"""Download and atomically publish the pinned macOS arm64 Node.js runtime."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
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
NODE_RUNTIME_TREE_SHA256 = (
    "efb26b7052ef066e3cf615c56317ad42d104272cda8e4b6dc11a56de14a6a3e8"
)
MAX_ARCHIVE_BYTES = 96 * 1024 * 1024


class NodeRuntimeError(RuntimeError):
    """The pinned Node.js runtime could not be downloaded or verified safely."""


def _add_digest_field(digest: object, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _runtime_tree_digest(root: Path) -> str:
    """Hash every runtime path, file byte, and confined symlink target."""

    digest = hashlib.sha256()
    try:
        resolved_root = root.resolve(strict=True)
        pending = [root]
        while pending:
            directory = pending.pop()
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
            child_directories: list[Path] = []
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix().encode("utf-8")
                if entry.is_symlink():
                    target = path.resolve(strict=True)
                    target.relative_to(resolved_root)
                    digest.update(b"L")
                    _add_digest_field(digest, relative)
                    _add_digest_field(digest, os.readlink(path).encode("utf-8"))
                elif entry.is_dir(follow_symlinks=False):
                    digest.update(b"D")
                    _add_digest_field(digest, relative)
                    child_directories.append(path)
                elif entry.is_file(follow_symlinks=False):
                    descriptor = os.open(
                        path,
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                    )
                    try:
                        info = os.fstat(descriptor)
                        if not stat.S_ISREG(info.st_mode):
                            raise NodeRuntimeError(
                                f"Node.js 运行时包含无效文件：{path}"
                            )
                        digest.update(b"F")
                        _add_digest_field(digest, relative)
                        digest.update(info.st_size.to_bytes(8, "big"))
                        while True:
                            chunk = os.read(descriptor, 1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                    finally:
                        os.close(descriptor)
                else:
                    raise NodeRuntimeError(f"Node.js 运行时包含特殊文件：{path}")
            pending.extend(reversed(child_directories))
    except NodeRuntimeError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise NodeRuntimeError(f"Node.js 运行时完整性检查失败：{exc}") from exc
    return digest.hexdigest()


def _validate_runtime(root: Path, expected_tree_sha256: str) -> None:
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
    if _runtime_tree_digest(root) != expected_tree_sha256:
        raise NodeRuntimeError("Node.js 运行时完整性摘要不匹配。")


def _runtime_parent(project_root: Path) -> tuple[Path, tuple[int, int]]:
    try:
        root_info = project_root.lstat()
    except OSError as exc:
        raise NodeRuntimeError(f"项目目录不可用：{project_root}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise NodeRuntimeError(f"项目目录必须是真实目录：{project_root}")

    runtime_parent = project_root / "runtime"
    try:
        runtime_parent.mkdir(exist_ok=True)
        parent_info = runtime_parent.lstat()
        if stat.S_ISLNK(parent_info.st_mode):
            raise NodeRuntimeError("runtime 目录不能是符号链接。")
        if not stat.S_ISDIR(parent_info.st_mode):
            raise NodeRuntimeError("runtime 路径必须是真实目录。")
        expected = project_root.resolve(strict=True) / "runtime"
        if runtime_parent.resolve(strict=True) != expected:
            raise NodeRuntimeError("runtime 目录越出项目范围。")
    except NodeRuntimeError:
        raise
    except OSError as exc:
        raise NodeRuntimeError(f"无法准备 runtime 目录：{exc}") from exc
    return runtime_parent, (parent_info.st_dev, parent_info.st_ino)


def _require_parent_identity(path: Path, expected: tuple[int, int]) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise NodeRuntimeError("runtime 目录在安装期间不可用。") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or (info.st_dev, info.st_ino) != expected
    ):
        raise NodeRuntimeError("runtime 目录在安装期间发生变化。")


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
    tree_sha256: str = NODE_RUNTIME_TREE_SHA256,
    opener: Callable[..., ContextManager[BinaryIO]] = urllib.request.urlopen,
) -> Path:
    """Return a verified runtime, downloading the pinned archive when absent."""

    if not isinstance(project_root, Path):
        raise ValueError("project_root must be a Path")
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("sha256 must be a lowercase SHA-256 digest")
    if re.fullmatch(r"[0-9a-f]{64}", tree_sha256) is None:
        raise ValueError("tree_sha256 must be a lowercase SHA-256 digest")

    runtime_parent, parent_identity = _runtime_parent(project_root)
    destination = runtime_parent / "node"
    existing_identity: tuple[int, int] | None = None
    try:
        destination_info = destination.lstat()
    except FileNotFoundError:
        destination_info = None
    except OSError as exc:
        raise NodeRuntimeError(f"Node.js 运行目录不可用：{destination}") from exc
    if destination_info is not None:
        if stat.S_ISLNK(destination_info.st_mode) or not stat.S_ISDIR(
            destination_info.st_mode
        ):
            raise NodeRuntimeError("Node.js 运行目录必须是真实目录。")
        existing_identity = (destination_info.st_dev, destination_info.st_ino)
        try:
            _validate_runtime(destination, tree_sha256)
        except NodeRuntimeError:
            pass
        else:
            return destination

    try:
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
            _validate_runtime(candidate, tree_sha256)
            _require_parent_identity(runtime_parent, parent_identity)

            backup: Path | None = None
            if existing_identity is not None:
                current_info = destination.lstat()
                if (
                    stat.S_ISLNK(current_info.st_mode)
                    or not stat.S_ISDIR(current_info.st_mode)
                    or (current_info.st_dev, current_info.st_ino) != existing_identity
                ):
                    raise NodeRuntimeError(
                        "Node.js 运行目录在安装期间发生变化。"
                    )
                backup = runtime_parent / f".node-backup-{secrets.token_hex(12)}"
                os.replace(destination, backup)
            elif destination.exists() or destination.is_symlink():
                raise NodeRuntimeError("Node.js 运行目录在安装期间发生变化。")
            try:
                _require_parent_identity(runtime_parent, parent_identity)
                os.replace(candidate, destination)
                _validate_runtime(destination, tree_sha256)
            except BaseException:
                if backup is not None and not destination.exists():
                    os.replace(backup, destination)
                    backup = None
                raise
            if backup is not None:
                try:
                    shutil.rmtree(backup)
                except OSError:
                    # Publication already succeeded; retain the recoverable
                    # hidden backup instead of turning cleanup into failure.
                    pass
    except NodeRuntimeError:
        raise
    except OSError as exc:
        raise NodeRuntimeError(f"Node.js 运行时安装失败：{exc}") from exc

    _validate_runtime(destination, tree_sha256)
    return destination


__all__ = [
    "NODE_RUNTIME_ARCHIVE",
    "NODE_RUNTIME_SHA256",
    "NODE_RUNTIME_TREE_SHA256",
    "NODE_RUNTIME_TOP_LEVEL",
    "NODE_RUNTIME_URL",
    "NODE_RUNTIME_VERSION",
    "NodeRuntimeError",
    "ensure_node_runtime",
]
