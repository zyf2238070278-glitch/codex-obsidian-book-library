#!/usr/bin/env python3
"""Download and validate the distribution's pinned semantic model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence


MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
_MANIFEST_KEYS = frozenset({"model_id", "model_revision", "files"})
_FILE_KEYS = frozenset({"path", "size", "sha256"})
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW
_MAX_SYMLINKS = 40
_SNAPSHOT_PARTS = (
    "models--intfloat--multilingual-e5-small",
    "snapshots",
    MODEL_REVISION,
)


class ModelAssetError(RuntimeError):
    """The pinned model could not be downloaded or fully validated."""


def _expected_snapshot(model_root: Path) -> Path:
    return (
        model_root
        / "models--intfloat--multilingual-e5-small"
        / "snapshots"
        / MODEL_REVISION
    )


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _same_directory_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and stat.S_IFMT(before.st_mode) == stat.S_IFMT(after.st_mode)
        and stat.S_ISDIR(after.st_mode)
    )


def _remove_path_without_following(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ModelAssetError(f"语义模型清单包含重复 JSON 字段：{key}")
        result[key] = value
    return result


def _load_manifest(manifest_path: Path) -> list[dict[str, object]]:
    try:
        with manifest_path.open("r", encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=_object_without_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelAssetError(f"无法读取语义模型清单：{exc}") from exc

    if not isinstance(data, dict) or set(data) != _MANIFEST_KEYS:
        raise ModelAssetError("语义模型清单顶层字段不符合严格格式。")
    if data["model_id"] != MODEL_ID:
        raise ModelAssetError("语义模型清单的 model_id 不是固定模型。")
    if data["model_revision"] != MODEL_REVISION:
        raise ModelAssetError("语义模型清单的 model_revision 不是固定版本。")
    files = data["files"]
    if not isinstance(files, list) or not files:
        raise ModelAssetError("语义模型清单 files 必须是非空数组。")

    records: list[dict[str, object]] = []
    seen: set[str] = set()
    seen_casefold: set[str] = set()
    seen_nodes: dict[str, str] = {}
    file_nodes: dict[str, str] = {}
    directory_nodes: dict[str, str] = {}
    for record in files:
        if not isinstance(record, dict) or set(record) != _FILE_KEYS:
            raise ModelAssetError("语义模型清单文件字段不符合严格格式。")
        relative = record["path"]
        size = record["size"]
        digest = record["sha256"]
        if not isinstance(relative, str) or not _valid_relative_path(relative):
            raise ModelAssetError("语义模型清单包含不安全的文件路径。")
        folded = relative.casefold()
        if relative in seen:
            raise ModelAssetError("语义模型清单包含重复路径。")
        if folded in seen_casefold:
            raise ModelAssetError("语义模型清单包含大小写冲突路径。")
        components: list[str] = []
        path_parts = PurePosixPath(relative).parts
        for index, component in enumerate(path_parts):
            components.append(component)
            node = "/".join(components)
            folded_node = node.casefold()
            is_file = index == len(path_parts) - 1
            if is_file and folded_node in directory_nodes:
                raise ModelAssetError("语义模型清单包含文件与目录祖先冲突。")
            if not is_file and folded_node in file_nodes:
                raise ModelAssetError("语义模型清单包含文件与目录祖先冲突。")
            previous = seen_nodes.get(folded_node)
            if previous is not None and previous != node:
                raise ModelAssetError("语义模型清单包含大小写冲突路径。")
            seen_nodes[folded_node] = node
            if is_file:
                file_nodes[folded_node] = node
            else:
                directory_nodes[folded_node] = node
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ModelAssetError("语义模型清单包含无效 size。")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ModelAssetError("语义模型清单包含无效 sha256。")
        seen.add(relative)
        seen_casefold.add(folded)
        records.append(record)
    return records


def _valid_relative_path(value: str) -> bool:
    if not value or unicodedata.normalize("NFC", value) != value:
        return False
    if "\\" in value or value.startswith("/"):
        return False
    if any(unicodedata.category(character) == "Cc" for character in value):
        return False
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and path.as_posix() == value


def _open_model_root_fd(model_root: Path) -> int:
    try:
        return os.open(model_root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise ModelAssetError(f"语义模型目录不存在或不安全：{model_root}") from exc


def _open_directory_at(parent_fd: int, name: str, context: str) -> int:
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise ModelAssetError(f"{context}缺失或包含不安全目录：{name}") from exc


def _open_directory_chain(root_fd: int, parts: tuple[str, ...], context: str) -> int:
    current_fd = os.dup(root_fd)
    try:
        for part in parts:
            next_fd = _open_directory_at(current_fd, part, context)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _normalized_symlink_target(
    parent_parts: tuple[str, ...], target: str, relative: str
) -> tuple[str, ...]:
    if os.path.isabs(target):
        raise ModelAssetError(f"语义模型符号链接必须是相对路径：{relative}")
    if any(unicodedata.category(character) == "Cc" for character in target):
        raise ModelAssetError(f"语义模型符号链接包含控制字符：{relative}")
    normalized = list(parent_parts)
    for part in target.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized:
                raise ModelAssetError(f"语义模型符号链接越过模型目录：{relative}")
            normalized.pop()
        else:
            normalized.append(part)
    if not normalized:
        raise ModelAssetError(f"语义模型符号链接目标无效：{relative}")
    return tuple(normalized)


def _open_regular_file_fd(root_fd: int, relative: str) -> int:
    if not _valid_relative_path(relative):
        raise ModelAssetError(f"语义模型文件路径不安全：{relative}")
    parts = tuple(PurePosixPath(relative).parts)
    visited = {parts}
    followed = 0
    while True:
        parent_fd = _open_directory_chain(
            root_fd, parts[:-1], f"语义模型文件 {relative} 的"
        )
        try:
            try:
                file_fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent_fd)
            except OSError as open_error:
                try:
                    target = os.readlink(parts[-1], dir_fd=parent_fd)
                except OSError:
                    raise ModelAssetError(
                        f"语义模型文件缺失或不安全：{relative}"
                    ) from open_error
                followed += 1
                if followed > _MAX_SYMLINKS:
                    raise ModelAssetError(f"语义模型符号链接层数过多：{relative}")
                parts = _normalized_symlink_target(parts[:-1], target, relative)
                if parts in visited:
                    raise ModelAssetError(f"语义模型符号链接形成循环：{relative}")
                visited.add(parts)
                continue
            try:
                info = os.fstat(file_fd)
            except OSError as exc:
                os.close(file_fd)
                raise ModelAssetError(f"无法检查语义模型文件：{relative}") from exc
            if not stat.S_ISREG(info.st_mode):
                os.close(file_fd)
                raise ModelAssetError(f"语义模型条目不是普通文件：{relative}")
            return file_fd
        finally:
            os.close(parent_fd)


def _stable_file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _validate_file_descriptor(
    file_fd: int, relative: str, expected_size: int, expected_hash: str
) -> None:
    size = 0
    digest = hashlib.sha256()
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ModelAssetError(f"语义模型条目不是普通文件：{relative}")
        os.lseek(file_fd, 0, os.SEEK_SET)
        while chunk := os.read(file_fd, 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(file_fd)
    except OSError as exc:
        raise ModelAssetError(f"无法读取语义模型文件 {relative}：{exc}") from exc
    if _stable_file_identity(before) != _stable_file_identity(after):
        raise ModelAssetError(f"语义模型文件在校验期间发生变化：{relative}")
    if size != expected_size:
        raise ModelAssetError(
            f"语义模型文件大小不匹配：{relative}（{size} != {expected_size}）"
        )
    if digest.hexdigest() != expected_hash:
        raise ModelAssetError(f"语义模型文件 SHA-256 不匹配：{relative}")


def _open_listed_file_at(
    root_fd: int,
    parent_fd: int,
    parent_root_parts: tuple[str, ...],
    name: str,
    relative: str,
) -> int:
    try:
        file_fd = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
    except OSError as open_error:
        try:
            target = os.readlink(name, dir_fd=parent_fd)
        except OSError:
            raise ModelAssetError(
                f"语义模型文件缺失或不安全：{relative}"
            ) from open_error
        target_parts = _normalized_symlink_target(
            parent_root_parts, target, relative
        )
        if target_parts == parent_root_parts + (name,):
            raise ModelAssetError(f"语义模型符号链接形成循环：{relative}")
        return _open_regular_file_fd(root_fd, "/".join(target_parts))
    try:
        info = os.fstat(file_fd)
    except OSError as exc:
        os.close(file_fd)
        raise ModelAssetError(f"无法检查语义模型文件：{relative}") from exc
    if not stat.S_ISREG(info.st_mode):
        os.close(file_fd)
        raise ModelAssetError(f"语义模型条目不是普通文件：{relative}")
    return file_fd


def _entry_info(directory_fd: int, name: str, context: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise ModelAssetError(f"无法检查{context}：{name}") from exc


def _validate_cache_tree_fd(cache_fd: int) -> None:
    try:
        names = os.listdir(cache_fd)
    except OSError as exc:
        raise ModelAssetError(f"无法检查语义模型 .cache 元数据：{exc}") from exc
    for name in names:
        info = _entry_info(cache_fd, name, "语义模型 .cache 元数据")
        if stat.S_ISDIR(info.st_mode):
            child_fd = _open_directory_at(cache_fd, name, "语义模型 .cache 元数据")
            try:
                _validate_cache_tree_fd(child_fd)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(info.st_mode):
            try:
                metadata_fd = os.open(name, _FILE_FLAGS, dir_fd=cache_fd)
            except OSError as exc:
                raise ModelAssetError(
                    "语义模型 .cache 元数据包含不安全条目。"
                ) from exc
            try:
                opened_info = os.fstat(metadata_fd)
                if not stat.S_ISREG(opened_info.st_mode):
                    raise ModelAssetError(
                        "语义模型 .cache 元数据包含不安全条目。"
                    )
            except OSError as exc:
                raise ModelAssetError(
                    "语义模型 .cache 元数据包含不安全条目。"
                ) from exc
            finally:
                os.close(metadata_fd)
        else:
            raise ModelAssetError("语义模型 .cache 元数据包含不安全条目。")


def _collect_snapshot_contents_fd(
    root_fd: int,
    directory_fd: int,
    prefix: tuple[str, ...],
    allowed_directories: set[str],
    found: set[str],
    records: dict[str, dict[str, object]],
) -> None:
    try:
        names = os.listdir(directory_fd)
    except OSError as exc:
        raise ModelAssetError(f"无法检查语义模型快照内容：{exc}") from exc
    for name in names:
        relative_parts = prefix + (name,)
        relative = "/".join(relative_parts)
        info = _entry_info(directory_fd, name, "语义模型快照内容")
        if not prefix and name == ".cache":
            if not stat.S_ISDIR(info.st_mode):
                raise ModelAssetError("语义模型 .cache 元数据目录不安全。")
            cache_fd = _open_directory_at(directory_fd, name, "语义模型 .cache 元数据")
            try:
                _validate_cache_tree_fd(cache_fd)
            finally:
                os.close(cache_fd)
        elif stat.S_ISDIR(info.st_mode):
            if relative not in allowed_directories:
                raise ModelAssetError(f"语义模型快照包含额外目录：{relative}")
            child_fd = _open_directory_at(directory_fd, name, "语义模型快照")
            try:
                _collect_snapshot_contents_fd(
                    root_fd,
                    child_fd,
                    relative_parts,
                    allowed_directories,
                    found,
                    records,
                )
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            found.add(relative)
            record = records.get(relative)
            if record is not None:
                file_fd = _open_listed_file_at(
                    root_fd,
                    directory_fd,
                    _SNAPSHOT_PARTS + prefix,
                    name,
                    relative,
                )
                try:
                    _validate_file_descriptor(
                        file_fd,
                        relative,
                        int(record["size"]),
                        str(record["sha256"]),
                    )
                finally:
                    os.close(file_fd)
        else:
            raise ModelAssetError(f"语义模型快照包含特殊文件：{relative}")


def _validate_snapshot_and_files_fd(
    root_fd: int, snapshot_fd: int, records: list[dict[str, object]]
) -> None:
    records_by_path = {str(record["path"]): record for record in records}
    listed = set(records_by_path)
    allowed_directories: set[str] = set()
    for relative in listed:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            allowed_directories.add(parent.as_posix())
            parent = parent.parent

    found: set[str] = set()
    _collect_snapshot_contents_fd(
        root_fd, snapshot_fd, (), allowed_directories, found, records_by_path
    )
    extras = found - listed
    if extras:
        raise ModelAssetError(f"语义模型快照包含额外文件：{sorted(extras)[0]}")
    missing = listed - found
    if missing:
        raise ModelAssetError(f"语义模型文件缺失：{sorted(missing)[0]}")


def validate_model(*, model_root: Path, manifest_path: Path) -> Path:
    """Return the fixed valid snapshot or raise ModelAssetError."""

    root = _absolute(Path(model_root))
    snapshot = _expected_snapshot(root)
    records = _load_manifest(_absolute(Path(manifest_path)))
    root_fd = _open_model_root_fd(root)
    try:
        snapshot_fd = _open_directory_chain(root_fd, _SNAPSHOT_PARTS, "固定语义模型快照")
        try:
            _validate_snapshot_and_files_fd(root_fd, snapshot_fd, records)
        finally:
            os.close(snapshot_fd)
    finally:
        os.close(root_fd)
    return snapshot


def ensure_model(
    *,
    model_root: Path,
    manifest_path: Path,
    snapshot_download: Callable[..., str] | None = None,
) -> Path:
    """Reuse a valid snapshot or download the fixed revision and validate it."""

    root = _absolute(Path(model_root))
    manifest = _absolute(Path(manifest_path))
    expected = _expected_snapshot(root)
    try:
        return validate_model(model_root=root, manifest_path=manifest)
    except ModelAssetError:
        pass

    records = _load_manifest(manifest)
    allow_patterns = [str(record["path"]) for record in records]

    if snapshot_download is None:
        try:
            from huggingface_hub import snapshot_download as hub_snapshot_download
        except (ImportError, OSError) as exc:
            raise ModelAssetError(f"无法加载语义模型下载组件：{exc}") from exc
        snapshot_download = hub_snapshot_download

    backup_holder: Path | None = None
    backup_root: Path | None = None
    try:
        root_info = root.lstat()
    except FileNotFoundError:
        root_info = None
    except OSError as exc:
        raise ModelAssetError(f"无法检查语义模型目录：{exc}") from exc
    if root_info is not None:
        if not stat.S_ISDIR(root_info.st_mode) or root.is_symlink():
            raise ModelAssetError(f"语义模型目录不安全：{root}")
        isolated = False
        try:
            root.parent.mkdir(parents=True, exist_ok=True)
            backup_holder = Path(
                tempfile.mkdtemp(prefix=f".{root.name}.backup-", dir=root.parent)
            )
            backup_root = backup_holder / "original"
            os.replace(root, backup_root)
            isolated = True
            isolated_info = backup_root.lstat()
        except OSError as exc:
            if backup_holder is not None and not isolated:
                shutil.rmtree(backup_holder, ignore_errors=True)
            elif backup_root is not None:
                try:
                    os.replace(backup_root, root)
                except OSError as restore_error:
                    raise ModelAssetError(
                        "无法检查已隔离的旧语义模型缓存，"
                        f"且无法恢复；备份保留在：{backup_holder}"
                    ) from restore_error
                shutil.rmtree(backup_holder, ignore_errors=True)
                raise ModelAssetError(
                    f"无法检查已隔离的旧语义模型缓存：{exc}"
                ) from exc
            raise ModelAssetError(f"无法隔离旧语义模型缓存：{exc}") from exc
        if not _same_directory_identity(root_info, isolated_info):
            try:
                os.replace(backup_root, root)
            except OSError as restore_error:
                raise ModelAssetError(
                    "语义模型目录在隔离期间发生变化，"
                    f"且无法恢复；备份保留在：{backup_holder}"
                ) from restore_error
            shutil.rmtree(backup_holder)
            raise ModelAssetError("语义模型目录在隔离期间发生变化。")

    try:
        try:
            downloaded = snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(root),
                allow_patterns=allow_patterns,
            )
        except Exception as exc:
            raise ModelAssetError(f"下载固定语义模型失败：{exc}") from exc

        try:
            returned = _absolute(Path(downloaded)).resolve(strict=False)
            expected_resolved = expected.resolve(strict=False)
        except (OSError, TypeError, ValueError) as exc:
            raise ModelAssetError(
                f"下载器返回了无效的语义模型路径：{exc}"
            ) from exc
        if returned != expected_resolved:
            raise ModelAssetError("下载器未返回固定语义模型快照。")
        result = validate_model(model_root=root, manifest_path=manifest)
    except BaseException:
        restored = backup_root is None
        try:
            _remove_path_without_following(root)
            if backup_root is not None:
                os.replace(backup_root, root)
                restored = True
        except OSError as restore_error:
            raise ModelAssetError(
                "语义模型准备失败，且无法恢复旧缓存；"
                f"备份保留在：{backup_holder}；原因：{restore_error}"
            ) from restore_error
        finally:
            if backup_holder is not None and restored:
                shutil.rmtree(backup_holder, ignore_errors=True)
        raise
    if backup_holder is not None:
        try:
            shutil.rmtree(backup_holder)
        except OSError as exc:
            raise ModelAssetError(f"无法清理旧语义模型缓存：{exc}") from exc
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载并校验固定语义模型")
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    try:
        snapshot = ensure_model(
            model_root=args.model_root,
            manifest_path=args.manifest,
            snapshot_download=snapshot_download,
        )
    except ModelAssetError as exc:
        print(f"语义模型准备失败：{exc}", file=sys.stderr)
        return 1
    print(f"语义模型已校验：{snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
