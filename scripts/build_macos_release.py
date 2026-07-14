from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable, Mapping


MODEL_FILES = (
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

REQUIRED_SOURCE_FILES = (
    ".gitignore",
    ".python-version",
    "AGENTS.md",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/使用说明.md",
    "docs/安装说明.md",
    "docs/常见问题.md",
    "docs/隐私与数据存放.md",
    "install-macos.command",
    "pyproject.toml",
    "third_party/model/LICENSE-MIT",
    "third_party/uv/LICENSE-APACHE",
    "third_party/uv/LICENSE-MIT",
    "uv.lock",
)

EXECUTABLE_PATHS = frozenset({"install-macos.command", "bin/uv"})
REQUIRED_METADATA_KEYS = frozenset(
    {
        "version",
        "tag",
        "project",
        "model_id",
        "model_revision",
        "uv_version",
        "uv_sha256",
        "python",
        "archive",
        "top_level_directory",
    }
)
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
COPY_CHUNK_SIZE = 1024 * 1024
SCAN_OVERLAP = 4096
SQLITE_FILE_HEADER = b"SQLite format 3\x00"
FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {
        ".codex",
        ".git",
        ".venv",
        ".worktrees",
        "00-待导入",
        "10-原始书籍",
        "20-解析文本",
        "30-ai读书笔记",
        "__macosx",
        "books",
        "notes",
        "obsidian书库",
        "outputs",
        "plans",
        "specs",
        "tests",
        "vault",
        "书库",
    }
)
SQLITE_FILE_SUFFIXES = tuple(
    f"{database_suffix}{sidecar_suffix}"
    for database_suffix in (".db", ".sqlite", ".sqlite3")
    for sidecar_suffix in ("", "-journal", "-shm", "-wal")
)
FORBIDDEN_FILE_SUFFIXES = (".epub", ".pdf", *SQLITE_FILE_SUFFIXES)
PRIVATE_ABSOLUTE_PATH_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9])(?:[A-Za-z]:)?[\\/]+(?:users|home)[\\/]+",
    re.IGNORECASE,
)
BINARY_PAYLOAD_PATHS = frozenset(
    {
        "bin/uv",
        "data/models/model.safetensors",
        "data/models/sentencepiece.bpe.model",
    }
)
GENERIC_ACCOUNT_NAMES = frozenset(
    {
        "admin",
        "administrator",
        "guest",
        "home",
        "public",
        "root",
        "runner",
        "shared",
        "user",
        "users",
    }
)
SECRET_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rb"\bsk-(?:proj-|live-)?[A-Za-z0-9_-]{16,}",
        rb"\bgh[pousr]_[A-Za-z0-9_]{20,}",
        rb"\bgithub_pat_[A-Za-z0-9_]{20,}",
        rb"\bAKIA[0-9A-Z]{16}\b",
        rb"\bBearer\s+[A-Za-z0-9._~+/-]{20,}",
        rb"(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|webhook)"
        rb"\s*[:=]\s*[\"']?[^\s\"']{16,}",
        rb"-----BEGIN[ A-Z]+PRIVATE KEY-----",
        rb"https://(?:hooks\.slack\.com/services|discord(?:app)?\.com/api/webhooks|"
        rb"open\.feishu\.cn/open-apis/bot/v2/hook)/[^\s\"']+",
    )
)


class ReleaseBuildError(ValueError):
    """The requested release cannot be built safely."""


@dataclass(frozen=True)
class BuildResult:
    archive: Path
    checksums: Path


@dataclass(frozen=True)
class ModelFileSpec:
    size: int
    sha256: str


@dataclass
class ArtifactBackup:
    path: Path
    descriptor: int | None
    size: int
    sha256: str
    mode: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(COPY_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_size_and_sha256(source: Any) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: source.read(COPY_CHUNK_SIZE), b""):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def _descriptor_size_and_sha256(descriptor: int) -> tuple[int, str]:
    with os.fdopen(descriptor, "rb", closefd=False) as source:
        source.seek(0)
        result = _stream_size_and_sha256(source)
        source.seek(0)
    return result


def _privacy_markers(private_paths: Iterable[Path | str]) -> tuple[bytes, ...]:
    markers: list[bytes] = []
    home = os.path.abspath(os.fspath(Path.home()))
    home_account = Path(home).name.casefold()
    generic_home = home_account in GENERIC_ACCOUNT_NAMES
    for path in private_paths:
        value = os.path.abspath(os.fspath(path))
        if generic_home and value == home:
            continue
        encoded = value.encode("utf-8")
        if encoded and encoded not in markers:
            markers.append(encoded)
    username = Path.home().name
    normalized_username = username.casefold()
    if (
        len(username) >= 5
        and normalized_username not in GENERIC_ACCOUNT_NAMES
        and not (
            len(username) >= 12
            and all(character in "0123456789abcdefABCDEF" for character in username)
        )
    ):
        encoded_username = username.encode("utf-8")
        if encoded_username not in markers:
            markers.append(encoded_username)
    return tuple(markers)


def _is_binary_payload_path(path_hint: str) -> bool:
    parts = tuple(part for part in path_hint.split("/") if part)
    for relative in BINARY_PAYLOAD_PATHS:
        binary_parts = tuple(relative.split("/"))
        if parts == binary_parts or (
            len(parts) == len(binary_parts) + 1 and parts[1:] == binary_parts
        ):
            return True
    return False


def _scan_chunk(
    chunk: bytes,
    *,
    binary: bool,
    private_markers: tuple[bytes, ...],
    label: str,
) -> None:
    for marker in private_markers:
        if marker and marker in chunk:
            raise ReleaseBuildError(f"private marker detected in {label}")
    if binary:
        return
    if PRIVATE_ABSOLUTE_PATH_PATTERN.search(chunk):
        raise ReleaseBuildError(f"private absolute path detected in {label}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(chunk):
            raise ReleaseBuildError(f"secret pattern detected in {label}")


def _scan_stream(
    source: Any,
    *,
    path_hint: str,
    private_markers: tuple[bytes, ...],
) -> None:
    first = source.read(COPY_CHUNK_SIZE)
    if first.startswith(SQLITE_FILE_HEADER):
        raise ReleaseBuildError(f"SQLite database content detected in {path_hint}")
    binary = _is_binary_payload_path(path_hint)
    decoder = None if binary else codecs.getincrementaldecoder("utf-8")("strict")
    previous = b""
    current = first
    while current:
        if decoder is not None:
            try:
                decoder.decode(current, final=False)
            except UnicodeDecodeError as exc:
                raise ReleaseBuildError(
                    f"text payload must be valid UTF-8: {path_hint}"
                ) from exc
        combined = previous + current
        _scan_chunk(
            combined,
            binary=binary,
            private_markers=private_markers,
            label=path_hint,
        )
        previous = combined[-SCAN_OVERLAP:]
        current = source.read(COPY_CHUNK_SIZE)
    if decoder is not None:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ReleaseBuildError(
                f"text payload must be valid UTF-8: {path_hint}"
            ) from exc


def _scan_file_content(
    path: Path,
    *,
    path_hint: str,
    private_markers: tuple[bytes, ...],
) -> None:
    with path.open("rb") as source:
        _scan_stream(
            source,
            path_hint=path_hint,
            private_markers=private_markers,
        )


def _check_forbidden_path(relative: str) -> None:
    parts = tuple(part for part in relative.replace("\\", "/").split("/") if part)
    lowered = tuple(part.casefold() for part in parts)
    for part in lowered:
        if part == ".ds_store" or part.startswith("._") or part == "__macosx":
            raise ReleaseBuildError(f"forbidden release path: {relative}")
    for directory in lowered[:-1]:
        if directory in FORBIDDEN_DIRECTORY_NAMES:
            raise ReleaseBuildError(f"forbidden release directory: {relative}")
    if lowered and lowered[-1].endswith(FORBIDDEN_FILE_SUFFIXES):
        raise ReleaseBuildError(f"forbidden release file type: {relative}")


def _validate_real_directory(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReleaseBuildError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ReleaseBuildError(f"{label} must be a real directory: {path}")
    return info


def _copy_file(source: Path, destination: Path, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, destination.open("wb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=COPY_CHUNK_SIZE)
    destination.chmod(mode)


def _verify_staged_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            initial_info = os.fstat(descriptor)
            if not stat.S_ISREG(initial_info.st_mode):
                raise ReleaseBuildError(f"{label} must be a regular file: {path}")
            if initial_info.st_size != expected_size:
                raise ReleaseBuildError(
                    f"{label} size does not match the trusted payload"
                )
            actual_size, actual_sha256 = _stream_size_and_sha256(source)
            final_info = os.fstat(descriptor)
    except OSError as exc:
        raise ReleaseBuildError(f"cannot verify {label}: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if (
        actual_size != expected_size
        or final_info.st_size != expected_size
        or (initial_info.st_dev, initial_info.st_ino)
        != (final_info.st_dev, final_info.st_ino)
    ):
        raise ReleaseBuildError(f"{label} size changed during verification")
    if actual_sha256 != expected_sha256:
        raise ReleaseBuildError(f"{label} SHA-256 does not match the trusted payload")
    current_info = _ensure_regular_file(path, label)
    if (current_info.st_dev, current_info.st_ino) != (
        initial_info.st_dev,
        initial_info.st_ino,
    ):
        raise ReleaseBuildError(f"{label} changed during verification")


def _load_metadata(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"cannot read release metadata {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ReleaseBuildError("release metadata must be a JSON object")
    missing = sorted(REQUIRED_METADATA_KEYS - raw.keys())
    extra = sorted(raw.keys() - REQUIRED_METADATA_KEYS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected keys: {', '.join(extra)}")
        raise ReleaseBuildError("invalid release metadata (" + "; ".join(details) + ")")
    if any(not isinstance(value, str) or not value for value in raw.values()):
        raise ReleaseBuildError("all release metadata values must be non-empty strings")
    if len(raw["uv_sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in raw["uv_sha256"]
    ):
        raise ReleaseBuildError("uv_sha256 must be a lowercase SHA-256 digest")
    for key in ("archive", "top_level_directory"):
        value = raw[key]
        if Path(value).name != value or value in {".", ".."}:
            raise ReleaseBuildError(f"{key} must be a single safe path component")
    if not raw["archive"].endswith(".zip"):
        raise ReleaseBuildError("archive must end in .zip")
    return {str(key): str(value) for key, value in raw.items()}


def _ensure_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReleaseBuildError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ReleaseBuildError(f"{label} must be a regular file: {path}")
    return info


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _open_regular_descriptor(
    path: Path,
    label: str,
) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseBuildError(f"cannot open {label}: {path}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ReleaseBuildError(f"{label} must be a regular file: {path}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, info


def _load_model_manifest(
    path: Path,
    metadata: Mapping[str, str],
) -> dict[str, ModelFileSpec]:
    _ensure_regular_file(path, "model manifest")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"cannot read model manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "model_id",
        "model_revision",
        "files",
    }:
        raise ReleaseBuildError("model manifest must contain exactly model_id, model_revision, and files")
    if raw["model_id"] != metadata["model_id"]:
        raise ReleaseBuildError("model manifest model_id does not match release metadata")
    if raw["model_revision"] != metadata["model_revision"]:
        raise ReleaseBuildError("model manifest revision does not match release metadata")
    files = raw["files"]
    if not isinstance(files, list):
        raise ReleaseBuildError("model manifest files must be a list")

    records: dict[str, ModelFileSpec] = {}
    ordered_paths: list[str] = []
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ReleaseBuildError(
                f"model manifest file record {index} must contain exactly path, size, and sha256"
            )
        relative = record["path"]
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ReleaseBuildError(f"model manifest file record {index} has an unsafe path")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or parsed.as_posix() != relative
        ):
            raise ReleaseBuildError(f"model manifest file record {index} has an unsafe path")
        if relative in records:
            raise ReleaseBuildError(f"model manifest contains duplicate path: {relative}")
        size = record["size"]
        if type(size) is not int or size <= 0:
            raise ReleaseBuildError(f"model manifest has invalid size for {relative}")
        sha256 = record["sha256"]
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ReleaseBuildError(f"model manifest has invalid sha256 for {relative}")
        records[relative] = ModelFileSpec(size=size, sha256=sha256)
        ordered_paths.append(relative)

    if tuple(ordered_paths) != MODEL_FILES:
        raise ReleaseBuildError(
            "model manifest must list the exact pinned model files in stable order"
        )
    return records


def _ensure_source_file(project_root: Path, path: Path, label: str) -> os.stat_result:
    try:
        relative = path.relative_to(project_root)
    except ValueError as exc:
        raise ReleaseBuildError(f"{label} escapes the project root: {path}") from exc
    current = project_root
    _validate_real_directory(current, "project root")
    for component in relative.parts[:-1]:
        current /= component
        _validate_real_directory(current, f"source directory {component}")
    return _ensure_regular_file(path, label)


def _collect_book_python_files(project_root: Path) -> list[Path]:
    book_root = project_root / "book_agent"
    _validate_real_directory(book_root, "book_agent source directory")
    selected: list[Path] = []
    for current_text, directories, files in os.walk(book_root, followlinks=False):
        current = Path(current_text)
        for directory in list(directories):
            _validate_real_directory(current / directory, f"book_agent directory {directory}")
        for filename in files:
            if filename.endswith(".py"):
                selected.append(current / filename)
    return sorted(selected)


def _collect_installer_python_files(project_root: Path) -> list[Path]:
    installer_root = project_root / "installer"
    _validate_real_directory(installer_root, "installer source directory")
    selected = [
        path
        for path in installer_root.iterdir()
        if path.name.endswith(".py") and not path.is_dir()
    ]
    return sorted(selected)


def _collect_source_files(project_root: Path) -> dict[str, Path]:
    _validate_real_directory(project_root, "project root")
    selected: dict[str, Path] = {}
    missing: list[str] = []
    for relative in REQUIRED_SOURCE_FILES:
        source = project_root / relative
        if not source.exists():
            missing.append(relative)
            continue
        _ensure_source_file(project_root, source, f"required source file {relative}")
        selected[relative] = source

    try:
        book_files = _collect_book_python_files(project_root)
    except ReleaseBuildError:
        book_files = []
    try:
        installer_files = _collect_installer_python_files(project_root)
    except ReleaseBuildError:
        installer_files = []
    if not book_files:
        missing.append("book_agent/**/*.py")
    if not installer_files:
        missing.append("installer/*.py")
    if missing:
        raise ReleaseBuildError(
            "missing required release source files: " + ", ".join(sorted(missing))
        )

    for source in (*book_files, *installer_files):
        relative = source.relative_to(project_root).as_posix()
        _ensure_source_file(
            project_root,
            source,
            f"allowlisted source file {relative}",
        )
        selected[relative] = source
    return dict(sorted(selected.items()))


def _write_payload(destination: Path, data: bytes, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    destination.chmod(mode)


def _copy_source_payload(project_root: Path, payload_root: Path) -> set[str]:
    selected = _collect_source_files(project_root)
    for relative, source in selected.items():
        mode = 0o755 if relative in EXECUTABLE_PATHS else 0o644
        _copy_file(source, payload_root / relative, mode)
    return set(selected)


def _model_snapshot_entries(snapshot: Path) -> set[str]:
    entries: set[str] = set()
    allowed_directories = {"1_Pooling"}
    observed_directories: set[str] = set()
    for current_text, directories, files in os.walk(snapshot, followlinks=False):
        current = Path(current_text)
        for directory in list(directories):
            path = current / directory
            relative = path.relative_to(snapshot).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ReleaseBuildError(f"unsafe model snapshot directory: {relative}")
            observed_directories.add(relative)
        for filename in files:
            path = current / filename
            info = path.lstat()
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                relative = path.relative_to(snapshot).as_posix()
                raise ReleaseBuildError(f"unsafe model snapshot file: {relative}")
            entries.add(path.relative_to(snapshot).as_posix())
    unexpected_directories = sorted(observed_directories - allowed_directories)
    if unexpected_directories:
        raise ReleaseBuildError(
            "unexpected model snapshot directories: "
            + ", ".join(unexpected_directories)
        )
    return entries


def _copy_model_payload(
    snapshot: Path,
    payload_root: Path,
    expected_revision: str,
    trusted_files: Mapping[str, ModelFileSpec],
) -> set[str]:
    if snapshot.name != expected_revision:
        raise ReleaseBuildError(
            f"model snapshot must be the pinned revision {expected_revision}"
        )
    _validate_real_directory(snapshot, "model snapshot")

    actual = _model_snapshot_entries(snapshot)
    expected = set(MODEL_FILES)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise ReleaseBuildError("model snapshot file set mismatch (" + "; ".join(details) + ")")

    blobs = snapshot.parent.parent / "blobs"
    _validate_real_directory(blobs, "model cache blobs directory")
    blobs_root = blobs.resolve(strict=True)

    copied: set[str] = set()
    for relative in MODEL_FILES:
        source = snapshot / relative
        source_info = source.lstat()
        if stat.S_ISLNK(source_info.st_mode):
            try:
                content_source = source.resolve(strict=True)
                content_source.relative_to(blobs_root)
            except (OSError, ValueError) as exc:
                raise ReleaseBuildError(
                    f"model symlink must resolve inside the same cache blobs directory: {relative}"
                ) from exc
        elif stat.S_ISREG(source_info.st_mode):
            content_source = source
        else:
            raise ReleaseBuildError(f"model file must be regular or a safe symlink: {relative}")
        content_info = content_source.lstat()
        if not stat.S_ISREG(content_info.st_mode) or content_info.st_size <= 0:
            raise ReleaseBuildError(f"model file must be a non-empty regular file: {relative}")
        trusted = trusted_files[relative]
        if content_info.st_size != trusted.size:
            raise ReleaseBuildError(
                f"model file size does not match trusted model manifest: {relative}"
            )
        actual_sha256 = _sha256_file(content_source)
        if actual_sha256 != trusted.sha256:
            raise ReleaseBuildError(
                f"model file SHA-256 does not match trusted model manifest: {relative}"
            )
        destination_relative = f"data/models/{relative}"
        destination = payload_root / destination_relative
        _copy_file(content_source, destination, 0o644)
        _verify_staged_file(
            destination,
            expected_size=trusted.size,
            expected_sha256=trusted.sha256,
            label=f"staged model file {relative}",
        )
        copied.add(destination_relative)
    return copied


def _copy_uv_payload(
    uv_binary: Path,
    payload_root: Path,
    expected_sha256: str,
) -> tuple[str, ModelFileSpec]:
    info = _ensure_regular_file(uv_binary, "uv binary")
    if not info.st_mode & 0o111 or not os.access(uv_binary, os.X_OK):
        raise ReleaseBuildError(f"uv binary is not executable: {uv_binary}")
    actual_sha256 = _sha256_file(uv_binary)
    if actual_sha256 != expected_sha256:
        raise ReleaseBuildError(
            f"uv SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    destination = payload_root / "bin/uv"
    _copy_file(uv_binary, destination, 0o755)
    _verify_staged_file(
        destination,
        expected_size=info.st_size,
        expected_sha256=expected_sha256,
        label="staged uv binary",
    )
    return "bin/uv", ModelFileSpec(
        size=info.st_size,
        sha256=expected_sha256,
    )


def _payload_record(payload_root: Path, relative: str) -> dict[str, Any]:
    path = payload_root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "mode": "0755" if relative in EXECUTABLE_PATHS else "0644",
    }


def _write_manifest(
    payload_root: Path,
    metadata: Mapping[str, str],
    payload_files: Iterable[str],
) -> None:
    manifest = {
        "release": dict(metadata),
        "files": [
            _payload_record(payload_root, relative)
            for relative in sorted(payload_files)
        ],
    }
    data = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_payload(payload_root / "RELEASE-MANIFEST.json", data, 0o644)


def _safe_member_name(name: str) -> str:
    if "\x00" in name:
        raise ReleaseBuildError(f"unsafe ZIP member path: {name!r}")
    replaced = name.replace("\\", "/")
    if (
        not name
        or replaced.startswith("/")
        or re.match(r"^[A-Za-z]:/", replaced)
    ):
        raise ReleaseBuildError(f"unsafe ZIP member path: {name!r}")
    raw_parts = replaced.split("/")
    if ".." in raw_parts:
        raise ReleaseBuildError(f"unsafe ZIP member path: {name!r}")
    normalized = "/".join(
        unicodedata.normalize("NFC", part)
        for part in raw_parts
        if part not in {"", "."}
    )
    if not normalized:
        raise ReleaseBuildError(f"unsafe ZIP member path: {name!r}")
    return normalized


def scan_staging(
    staging_root: Path,
    *,
    private_paths: Iterable[Path | str] = (),
    expected_files: set[str] | None = None,
    expected_top_level: str | None = None,
) -> None:
    _validate_real_directory(staging_root, "staging root")
    private_markers = _privacy_markers(private_paths)
    normalized_names: set[str] = set()
    actual_files: set[str] = set()
    for path in staging_root.rglob("*"):
        info = path.lstat()
        relative = path.relative_to(staging_root).as_posix()
        normalized = unicodedata.normalize("NFC", relative).casefold()
        if normalized in normalized_names:
            raise ReleaseBuildError(f"duplicate normalized staging path: {relative}")
        normalized_names.add(normalized)
        _check_forbidden_path(relative)
        if stat.S_ISLNK(info.st_mode):
            raise ReleaseBuildError(f"staging must not contain symlinks: {path}")
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ReleaseBuildError(f"staging contains a special file: {path}")
        if stat.S_ISREG(info.st_mode):
            actual_files.add(relative)
            _scan_file_content(
                path,
                path_hint=relative,
                private_markers=private_markers,
            )

    if expected_top_level is not None:
        roots = {relative.split("/", 1)[0] for relative in actual_files}
        if roots != {expected_top_level}:
            raise ReleaseBuildError("staging does not have the expected single top-level directory")
    if expected_files is not None and actual_files != expected_files:
        raise ReleaseBuildError("staging payload does not match the release allowlist")
    if expected_top_level is not None and expected_files is not None:
        prefix = f"{expected_top_level}/"
        for relative in sorted(actual_files):
            payload_relative = relative.removeprefix(prefix)
            expected_mode = 0o755 if payload_relative in EXECUTABLE_PATHS else 0o644
            actual_mode = stat.S_IMODE((staging_root / relative).stat().st_mode)
            if actual_mode != expected_mode:
                raise ReleaseBuildError(f"staging file mode mismatch: {payload_relative}")


def _scan_zip_archive(
    archive: zipfile.ZipFile,
    *,
    private_markers: tuple[bytes, ...],
) -> None:
    normalized_names: set[str] = set()
    for info in archive.infolist():
        normalized = _safe_member_name(info.filename)
        collision_key = normalized.casefold()
        if collision_key in normalized_names:
            raise ReleaseBuildError(
                f"duplicate normalized ZIP member: {info.filename}"
            )
        normalized_names.add(collision_key)
        _check_forbidden_path(normalized)
        unix_mode = info.external_attr >> 16
        if info.is_dir():
            if unix_mode and not stat.S_ISDIR(unix_mode):
                raise ReleaseBuildError(
                    f"ZIP directory has an unsafe file type: {info.filename}"
                )
            continue
        if not stat.S_ISREG(unix_mode):
            raise ReleaseBuildError(
                f"ZIP member is not a regular file: {info.filename}"
            )
        with archive.open(info, "r") as source:
            _scan_stream(
                source,
                path_hint=normalized,
                private_markers=private_markers,
            )


def scan_zip(
    archive_path: Path,
    *,
    private_paths: Iterable[Path | str] = (),
) -> None:
    private_markers = _privacy_markers(private_paths)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _scan_zip_archive(
                archive,
                private_markers=private_markers,
            )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, ReleaseBuildError):
            raise
        raise ReleaseBuildError(f"cannot safely scan ZIP {archive_path}: {exc}") from exc


def _write_zip(payload_root: Path, archive_path: Path, top_level: str) -> None:
    files = sorted(path for path in payload_root.rglob("*") if path.is_file())
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path in files:
            relative = path.relative_to(payload_root).as_posix()
            mode = 0o755 if relative in EXECUTABLE_PATHS else 0o644
            info = zipfile.ZipInfo(
                filename=f"{top_level}/{relative}",
                date_time=FIXED_ZIP_TIMESTAMP,
            )
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | mode) << 16
            with path.open("rb") as source, archive.open(
                info,
                mode="w",
                force_zip64=True,
            ) as destination:
                shutil.copyfileobj(source, destination, length=COPY_CHUNK_SIZE)


def _verify_open_archive(
    archive: zipfile.ZipFile,
    metadata: Mapping[str, str],
    expected_payload: set[str],
    *,
    private_markers: tuple[bytes, ...],
    trusted_payload: Mapping[str, ModelFileSpec] | None = None,
) -> None:
    _scan_zip_archive(archive, private_markers=private_markers)
    top_level = metadata["top_level_directory"]
    expected_with_manifest = expected_payload | {"RELEASE-MANIFEST.json"}
    relative_by_name = {
        info.filename: info.filename.removeprefix(f"{top_level}/")
        for info in archive.infolist()
    }
    if any(
        not name.startswith(f"{top_level}/") or relative == name
        for name, relative in relative_by_name.items()
    ):
        raise ReleaseBuildError("ZIP does not have the expected single top-level directory")
    if set(relative_by_name.values()) != expected_with_manifest:
        raise ReleaseBuildError("ZIP payload does not match the release allowlist")
    ordered_relatives = list(relative_by_name.values())
    if ordered_relatives != sorted(ordered_relatives):
        raise ReleaseBuildError("ZIP central directory is not deterministically ordered")

    manifest_name = f"{top_level}/RELEASE-MANIFEST.json"
    try:
        with archive.open(manifest_name) as manifest_source:
            manifest = json.load(manifest_source)
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError("ZIP release manifest is missing or invalid") from exc
    if manifest.get("release") != dict(metadata):
        raise ReleaseBuildError("ZIP release metadata does not match the build metadata")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ReleaseBuildError("ZIP release manifest files must be a list")
    manifest_mode = stat.S_IMODE(
        archive.getinfo(manifest_name).external_attr >> 16
    )
    if manifest_mode != 0o644:
        raise ReleaseBuildError("ZIP release manifest mode must be 0644")
    expected_records = []
    for relative in sorted(expected_payload):
        info = archive.getinfo(f"{top_level}/{relative}")
        with archive.open(info) as source:
            size, digest = _stream_size_and_sha256(source)
        trusted = (
            None
            if trusted_payload is None
            else trusted_payload.get(relative)
        )
        if trusted is not None and (
            size != trusted.size or digest != trusted.sha256
        ):
            raise ReleaseBuildError(
                f"ZIP trusted payload size/SHA-256 mismatch: {relative}"
            )
        mode = stat.S_IMODE(info.external_attr >> 16)
        expected_mode = 0o755 if relative in EXECUTABLE_PATHS else 0o644
        if mode != expected_mode:
            raise ReleaseBuildError(f"ZIP member mode mismatch: {relative}")
        expected_records.append(
            {
                "path": relative,
                "size": size,
                "sha256": digest,
                "mode": format(expected_mode, "04o"),
            }
        )
    if records != expected_records:
        raise ReleaseBuildError("ZIP release manifest does not match archive contents")


def _verify_archive_stream(
    archive_source: BinaryIO,
    archive_label: str,
    metadata: Mapping[str, str],
    expected_payload: set[str],
    *,
    private_markers: tuple[bytes, ...],
    trusted_payload: Mapping[str, ModelFileSpec] | None,
) -> None:
    try:
        archive_source.seek(0)
        with zipfile.ZipFile(archive_source) as archive:
            _verify_open_archive(
                archive,
                metadata,
                expected_payload,
                private_markers=private_markers,
                trusted_payload=trusted_payload,
            )
    except ReleaseBuildError:
        raise
    except Exception as exc:
        raise ReleaseBuildError(
            f"cannot safely verify ZIP {archive_label}: {exc}"
        ) from exc


def _verify_archive(
    archive_source: Path | BinaryIO,
    metadata: Mapping[str, str],
    expected_payload: set[str],
    *,
    private_paths: Iterable[Path | str] = (),
    trusted_payload: Mapping[str, ModelFileSpec] | None = None,
    archive_label: str | None = None,
) -> None:
    private_markers = _privacy_markers(private_paths)
    if not isinstance(archive_source, os.PathLike):
        _verify_archive_stream(
            archive_source,
            archive_label or "opened release ZIP",
            metadata,
            expected_payload,
            private_markers=private_markers,
            trusted_payload=trusted_payload,
        )
        return

    archive_path = Path(archive_source)
    descriptor, initial_info = _open_regular_descriptor(
        archive_path,
        "release ZIP",
    )
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            _verify_archive_stream(
                source,
                str(archive_path),
                metadata,
                expected_payload,
                private_markers=private_markers,
                trusted_payload=trusted_payload,
            )
            final_info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(initial_info) != _file_identity(final_info):
        raise ReleaseBuildError("release ZIP changed during verification")
    if _file_identity(final_info) != _file_identity(
        _ensure_regular_file(archive_path, "release ZIP")
    ):
        raise ReleaseBuildError("release ZIP path changed during verification")


def _validate_output_target(path: Path, label: str) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ReleaseBuildError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ReleaseBuildError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise ReleaseBuildError(f"{label} must be a regular file: {path}")
    return True


def _unlink_regular_artifact(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode):
        raise ReleaseBuildError(f"cannot remove non-regular {label}: {path}")
    path.unlink()


def _create_artifact_backup(
    target: Path,
    backup_dir: Path,
    label: str,
) -> ArtifactBackup:
    placeholder_descriptor: int | None = None
    backup_descriptor: int | None = None
    backup: Path | None = None
    try:
        placeholder_descriptor, backup_name = tempfile.mkstemp(
            prefix=".macos-release-backup-",
            dir=backup_dir,
        )
        backup = Path(backup_name)
        os.close(placeholder_descriptor)
        placeholder_descriptor = None
        backup.unlink()
        os.link(target, backup, follow_symlinks=False)
        backup_descriptor, backup_info = _open_regular_descriptor(
            backup,
            f"backup {label}",
        )
        target_info = _ensure_regular_file(target, label)
        if (backup_info.st_dev, backup_info.st_ino) != (
            target_info.st_dev,
            target_info.st_ino,
        ):
            raise ReleaseBuildError(
                f"backup {label} does not match the existing artifact"
            )
        size, digest = _descriptor_size_and_sha256(backup_descriptor)
        final_info = os.fstat(backup_descriptor)
        if size != backup_info.st_size or _file_identity(
            backup_info
        ) != _file_identity(final_info):
            raise ReleaseBuildError(f"backup {label} changed while being captured")
        return ArtifactBackup(
            path=backup,
            descriptor=backup_descriptor,
            size=size,
            sha256=digest,
            mode=stat.S_IMODE(backup_info.st_mode),
        )
    except (OSError, ReleaseBuildError) as exc:
        if placeholder_descriptor is not None:
            os.close(placeholder_descriptor)
        if backup_descriptor is not None:
            os.close(backup_descriptor)
        if backup is not None:
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass
        raise ReleaseBuildError(f"cannot back up existing {label}: {target}") from exc


def _copy_artifact_backup(
    backup: ArtifactBackup,
    label: str,
) -> tuple[Path, int, os.stat_result]:
    if backup.descriptor is None:
        raise ReleaseBuildError(f"backup {label} descriptor is closed")
    candidate_descriptor: int | None = None
    candidate: Path | None = None
    try:
        source_info = os.fstat(backup.descriptor)
        if not stat.S_ISREG(source_info.st_mode):
            raise ReleaseBuildError(f"backup {label} is not a regular file")
        candidate_descriptor, candidate_name = tempfile.mkstemp(
            prefix=".macos-release-backup-",
            dir=backup.path.parent,
        )
        candidate = Path(candidate_name)
        digest = hashlib.sha256()
        size = 0
        with (
            os.fdopen(backup.descriptor, "rb", closefd=False) as source,
            os.fdopen(candidate_descriptor, "wb", closefd=False) as destination,
        ):
            source.seek(0)
            for chunk in iter(lambda: source.read(COPY_CHUNK_SIZE), b""):
                destination.write(chunk)
                size += len(chunk)
                digest.update(chunk)
            destination.flush()
            source.seek(0)
        os.fchmod(candidate_descriptor, backup.mode)
        candidate_info = os.fstat(candidate_descriptor)
        final_source_info = os.fstat(backup.descriptor)
        if (
            size != backup.size
            or digest.hexdigest() != backup.sha256
            or candidate_info.st_size != backup.size
            or source_info.st_size != backup.size
            or final_source_info.st_size != backup.size
        ):
            raise ReleaseBuildError(f"backup {label} content changed")
        return candidate, candidate_descriptor, candidate_info
    except Exception as exc:
        if candidate_descriptor is not None:
            os.close(candidate_descriptor)
        if candidate is not None:
            candidate.unlink(missing_ok=True)
        if isinstance(exc, ReleaseBuildError):
            raise
        raise ReleaseBuildError(f"cannot copy backup {label}: {exc}") from exc


def _restore_artifact_backup(
    backup: ArtifactBackup,
    target: Path,
    label: str,
) -> None:
    restore_candidate, restore_descriptor, restore_info = _copy_artifact_backup(
        backup,
        label,
    )
    try:
        os.replace(restore_candidate, target)
        _validate_output_target(target, label)
        restored_info = _ensure_regular_file(target, f"restored {label}")
        if (restored_info.st_dev, restored_info.st_ino) != (
            restore_info.st_dev,
            restore_info.st_ino,
        ):
            raise ReleaseBuildError(
                f"restored {label} does not match its recovery backup"
            )
        initial_restore_info = os.fstat(restore_descriptor)
        size, digest = _descriptor_size_and_sha256(restore_descriptor)
        final_restore_info = os.fstat(restore_descriptor)
        if (
            size != backup.size
            or digest != backup.sha256
            or _file_identity(initial_restore_info)
            != _file_identity(final_restore_info)
        ):
            raise ReleaseBuildError(f"restored {label} changed during rollback")
    except Exception:
        restore_candidate.unlink(missing_ok=True)
        raise
    finally:
        os.close(restore_descriptor)


def _backup_path_matches_descriptor(backup: ArtifactBackup) -> bool:
    if backup.descriptor is None:
        return False
    try:
        path_info = backup.path.lstat()
        descriptor_info = os.fstat(backup.descriptor)
    except OSError:
        return False
    return (
        stat.S_ISREG(path_info.st_mode)
        and (path_info.st_dev, path_info.st_ino)
        == (descriptor_info.st_dev, descriptor_info.st_ino)
    )


def _ensure_recovery_backup(
    backup: ArtifactBackup,
    label: str,
) -> Path:
    if backup.descriptor is None:
        raise ReleaseBuildError(f"backup {label} descriptor is closed")
    size, digest = _descriptor_size_and_sha256(backup.descriptor)
    if size != backup.size or digest != backup.sha256:
        raise ReleaseBuildError(f"backup {label} content changed")
    if _backup_path_matches_descriptor(backup):
        return backup.path

    recovery, recovery_descriptor, _ = _copy_artifact_backup(backup, label)
    old_path = backup.path
    old_descriptor = backup.descriptor
    backup.path = recovery
    backup.descriptor = recovery_descriptor
    try:
        old_path.unlink(missing_ok=True)
    except OSError:
        pass
    os.close(old_descriptor)
    return recovery


def _cleanup_artifact_backups(
    backups: Iterable[ArtifactBackup],
    *,
    retained: set[Path] | None = None,
) -> list[str]:
    retained = retained or set()
    errors: list[str] = []
    for backup in backups:
        if backup.path not in retained:
            try:
                backup.path.unlink(missing_ok=True)
            except Exception as exc:
                errors.append(f"{backup.path}: {exc}")
        if backup.descriptor is not None:
            try:
                os.close(backup.descriptor)
            except OSError as exc:
                errors.append(f"{backup.path} descriptor: {exc}")
            backup.descriptor = None
    return errors


def _publish_release_artifacts(
    *,
    candidate_archive: Path,
    candidate_checksums: Path,
    archive_path: Path,
    checksums_path: Path,
    backup_dir: Path,
    validate_published: Callable[[], None],
) -> None:
    artifacts = (
        (candidate_archive, archive_path, "release ZIP"),
        (candidate_checksums, checksums_path, "SHA256SUMS"),
    )
    backups: dict[Path, ArtifactBackup] = {}
    published: list[tuple[Path, str]] = []

    try:
        for candidate, target, label in artifacts:
            _ensure_regular_file(candidate, f"candidate {label}")
            existed = _validate_output_target(target, label)
            if existed:
                backups[target] = _create_artifact_backup(
                    target,
                    backup_dir,
                    label,
                )
    except Exception as exc:
        cleanup_errors = _cleanup_artifact_backups(backups.values())
        if cleanup_errors:
            raise ReleaseBuildError(
                f"release artifact preparation failed: {exc}; "
                "backup cleanup failed for " + ", ".join(cleanup_errors)
            ) from exc
        raise

    try:
        for candidate, target, label in artifacts:
            os.replace(candidate, target)
            published.append((target, label))
            _validate_output_target(target, label)
        validate_published()
    except Exception as exc:
        rollback_errors: list[str] = []
        retained_backups: set[Path] = set()
        for target, label in reversed(published):
            backup = backups.get(target)
            try:
                if backup is None:
                    _unlink_regular_artifact(target, label)
                else:
                    _restore_artifact_backup(backup, target, label)
            except Exception as rollback_exc:
                rollback_errors.append(f"{label}: {rollback_exc}")
                if backup is not None:
                    try:
                        recovery_path = _ensure_recovery_backup(backup, label)
                    except Exception as recovery_exc:
                        rollback_errors.append(
                            f"{label} recovery backup unavailable: {recovery_exc}"
                        )
                    else:
                        retained_backups.add(recovery_path)
        cleanup_errors = _cleanup_artifact_backups(
            backups.values(),
            retained=retained_backups,
        )
        message = f"release artifact publication failed: {exc}"
        if rollback_errors:
            message += "; rollback failed for " + ", ".join(rollback_errors)
        if retained_backups:
            message += "; recovery backup retained at " + ", ".join(
                str(path) for path in sorted(retained_backups)
            )
        if cleanup_errors:
            message += "; backup cleanup failed for " + ", ".join(
                cleanup_errors
            )
        raise ReleaseBuildError(message) from exc
    else:
        cleanup_errors = _cleanup_artifact_backups(backups.values())
        if cleanup_errors:
            raise ReleaseBuildError(
                "release artifacts were published but backup cleanup failed for "
                + ", ".join(cleanup_errors)
            )


def build_release(
    *,
    project_root: Path,
    model_snapshot: Path,
    uv_binary: Path,
    output_dir: Path,
    metadata_path: Path | None = None,
    model_manifest_path: Path | None = None,
) -> BuildResult:
    project_root = Path(project_root).expanduser().absolute()
    model_snapshot = Path(model_snapshot).expanduser().absolute()
    uv_binary = Path(uv_binary).expanduser().absolute()
    output_dir = Path(output_dir).expanduser().absolute()
    if metadata_path is None:
        metadata_path = project_root / "distribution" / "release.json"
    metadata = _load_metadata(Path(metadata_path).expanduser().absolute())
    if model_manifest_path is None:
        model_manifest_path = project_root / "distribution" / "model-manifest.json"
    trusted_model_files = _load_model_manifest(
        Path(model_manifest_path).expanduser().absolute(),
        metadata,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_real_directory(output_dir, "output directory")
    archive_path = output_dir / metadata["archive"]
    checksums_path = output_dir / "SHA256SUMS"
    _validate_output_target(archive_path, "release ZIP")
    _validate_output_target(checksums_path, "SHA256SUMS")
    private_paths = (project_root, Path.home(), model_snapshot)
    with tempfile.TemporaryDirectory(prefix=".macos-release-", dir=output_dir) as temp:
        staging_root = Path(temp)
        payload_root = staging_root / metadata["top_level_directory"]
        payload_root.mkdir()
        payload_files = _copy_source_payload(project_root, payload_root)
        payload_files.update(
            _copy_model_payload(
                model_snapshot,
                payload_root,
                metadata["model_revision"],
                trusted_model_files,
            )
        )
        trusted_payload = {
            f"data/models/{relative}": trusted_model_files[relative]
            for relative in MODEL_FILES
        }
        uv_relative, trusted_uv = _copy_uv_payload(
            uv_binary,
            payload_root,
            metadata["uv_sha256"],
        )
        payload_files.add(uv_relative)
        trusted_payload[uv_relative] = trusted_uv
        _write_manifest(payload_root, metadata, payload_files)
        staged_payload = payload_files | {"RELEASE-MANIFEST.json"}
        expected_staging_files = {
            f'{metadata["top_level_directory"]}/{relative}'
            for relative in staged_payload
        }
        scan_staging(
            staging_root,
            private_paths=private_paths,
            expected_files=expected_staging_files,
            expected_top_level=metadata["top_level_directory"],
        )
        candidate_archive = staging_root / metadata["archive"]
        _write_zip(
            payload_root,
            candidate_archive,
            metadata["top_level_directory"],
        )
        _verify_archive(
            candidate_archive,
            metadata,
            payload_files,
            private_paths=private_paths,
            trusted_payload=trusted_payload,
        )
        candidate_archive.chmod(0o644)
        candidate_checksums = staging_root / "candidate-SHA256SUMS"
        checksum_data = (
            f"{_sha256_file(candidate_archive)}  {archive_path.name}\n"
        ).encode("utf-8")
        _write_payload(candidate_checksums, checksum_data, 0o644)
        if candidate_checksums.read_bytes() != checksum_data:
            raise ReleaseBuildError("candidate SHA256SUMS verification failed")

        def validate_published() -> None:
            archive_descriptor, archive_info = _open_regular_descriptor(
                archive_path,
                "published release ZIP",
            )
            try:
                checksums_descriptor, checksums_info = _open_regular_descriptor(
                    checksums_path,
                    "published SHA256SUMS",
                )
            except Exception:
                os.close(archive_descriptor)
                raise
            try:
                with (
                    os.fdopen(
                        archive_descriptor,
                        "rb",
                        closefd=False,
                    ) as archive_source,
                    os.fdopen(
                        checksums_descriptor,
                        "rb",
                        closefd=False,
                    ) as checksums_source,
                ):
                    _verify_archive(
                        archive_source,
                        metadata,
                        payload_files,
                        private_paths=private_paths,
                        trusted_payload=trusted_payload,
                        archive_label=str(archive_path),
                    )
                    archive_source.seek(0)
                    _, archive_sha256 = _stream_size_and_sha256(
                        archive_source
                    )
                    expected_checksums = (
                        f"{archive_sha256}  {archive_path.name}\n"
                    ).encode("utf-8")
                    checksums_source.seek(0)
                    actual_checksums = checksums_source.read(
                        len(expected_checksums) + 1
                    )
                    final_archive_info = os.fstat(archive_descriptor)
                    final_checksums_info = os.fstat(checksums_descriptor)
            finally:
                os.close(checksums_descriptor)
                os.close(archive_descriptor)
            if _file_identity(archive_info) != _file_identity(
                final_archive_info
            ):
                raise ReleaseBuildError(
                    "published release ZIP changed during final validation"
                )
            if _file_identity(checksums_info) != _file_identity(
                final_checksums_info
            ):
                raise ReleaseBuildError(
                    "published SHA256SUMS changed during final validation"
                )
            if _file_identity(final_archive_info) != _file_identity(
                _ensure_regular_file(archive_path, "published release ZIP")
            ):
                raise ReleaseBuildError(
                    "published release ZIP path changed during final validation"
                )
            if _file_identity(final_checksums_info) != _file_identity(
                _ensure_regular_file(checksums_path, "published SHA256SUMS")
            ):
                raise ReleaseBuildError(
                    "published SHA256SUMS path changed during final validation"
                )
            if actual_checksums != expected_checksums:
                raise ReleaseBuildError(
                    "published SHA256SUMS does not match the published release ZIP"
                )

        _publish_release_artifacts(
            candidate_archive=candidate_archive,
            candidate_checksums=candidate_checksums,
            archive_path=archive_path,
            checksums_path=checksums_path,
            backup_dir=output_dir,
            validate_published=validate_published,
        )
    return BuildResult(archive=archive_path, checksums=checksums_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic macOS Apple Silicon all-in-one release ZIP."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--uv-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Override distribution/release.json (intended for offline test fixtures).",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        help="Override distribution/model-manifest.json (intended for offline test fixtures).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = build_release(
            project_root=arguments.project_root,
            model_snapshot=arguments.model_snapshot,
            uv_binary=arguments.uv_binary,
            output_dir=arguments.output_dir,
            metadata_path=arguments.metadata,
            model_manifest_path=arguments.model_manifest,
        )
    except ReleaseBuildError as exc:
        raise SystemExit(f"release build failed: {exc}") from exc
    print(result.archive)
    print(result.checksums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
