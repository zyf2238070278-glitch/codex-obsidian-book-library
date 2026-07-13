from __future__ import annotations

import argparse
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
from pathlib import Path
from typing import Any, Iterable, Mapping


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
FORBIDDEN_FILE_SUFFIXES = (
    ".db",
    ".db-shm",
    ".db-wal",
    ".epub",
    ".pdf",
    ".sqlite",
    ".sqlite3",
)
TEXT_PATH_MARKERS = (b"/Users/", b"/home/", b"C:\\Users\\")
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


def _privacy_markers(private_paths: Iterable[Path | str]) -> tuple[bytes, ...]:
    markers: list[bytes] = []
    for path in private_paths:
        value = os.path.abspath(os.fspath(path))
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


def _looks_binary(path_hint: str, first_chunk: bytes) -> bool:
    lowered = path_hint.casefold()
    if lowered.endswith("/bin/uv") or lowered == "bin/uv":
        return True
    if lowered.endswith((".safetensors", ".model")):
        return True
    sample = first_chunk[:8192]
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
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
    for marker in TEXT_PATH_MARKERS:
        if marker in chunk:
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
    binary = _looks_binary(path_hint, first)
    previous = b""
    current = first
    while current:
        combined = previous + current
        _scan_chunk(
            combined,
            binary=binary,
            private_markers=private_markers,
            label=path_hint,
        )
        previous = combined[-SCAN_OVERLAP:]
        current = source.read(COPY_CHUNK_SIZE)


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
        destination_relative = f"data/models/{relative}"
        _copy_file(content_source, payload_root / destination_relative, 0o644)
        copied.add(destination_relative)
    return copied


def _copy_uv_payload(
    uv_binary: Path,
    payload_root: Path,
    expected_sha256: str,
) -> str:
    info = _ensure_regular_file(uv_binary, "uv binary")
    if not info.st_mode & 0o111 or not os.access(uv_binary, os.X_OK):
        raise ReleaseBuildError(f"uv binary is not executable: {uv_binary}")
    actual_sha256 = _sha256_file(uv_binary)
    if actual_sha256 != expected_sha256:
        raise ReleaseBuildError(
            f"uv SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    _copy_file(uv_binary, payload_root / "bin/uv", 0o755)
    return "bin/uv"


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


def scan_zip(
    archive_path: Path,
    *,
    private_paths: Iterable[Path | str] = (),
) -> None:
    private_markers = _privacy_markers(private_paths)
    normalized_names: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
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


def _verify_archive(
    archive_path: Path,
    metadata: Mapping[str, str],
    expected_payload: set[str],
    *,
    private_paths: Iterable[Path | str] = (),
) -> None:
    scan_zip(archive_path, private_paths=private_paths)
    top_level = metadata["top_level_directory"]
    expected_with_manifest = expected_payload | {"RELEASE-MANIFEST.json"}
    with zipfile.ZipFile(archive_path) as archive:
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


def build_release(
    *,
    project_root: Path,
    model_snapshot: Path,
    uv_binary: Path,
    output_dir: Path,
    metadata_path: Path | None = None,
) -> BuildResult:
    project_root = Path(project_root).expanduser().absolute()
    model_snapshot = Path(model_snapshot).expanduser().absolute()
    uv_binary = Path(uv_binary).expanduser().absolute()
    output_dir = Path(output_dir).expanduser().absolute()
    if metadata_path is None:
        metadata_path = project_root / "distribution" / "release.json"
    metadata = _load_metadata(Path(metadata_path).expanduser().absolute())

    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_real_directory(output_dir, "output directory")
    archive_path = output_dir / metadata["archive"]
    checksums_path = output_dir / "SHA256SUMS"
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
            )
        )
        payload_files.add(
            _copy_uv_payload(uv_binary, payload_root, metadata["uv_sha256"])
        )
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
        )
        os.replace(candidate_archive, archive_path)

    checksums_path.write_text(
        f"{_sha256_file(archive_path)}  {archive_path.name}\n",
        encoding="utf-8",
    )
    checksums_path.chmod(0o644)
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
        )
    except ReleaseBuildError as exc:
        raise SystemExit(f"release build failed: {exc}") from exc
    print(result.archive)
    print(result.checksums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
