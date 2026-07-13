from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts import build_macos_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

SOURCE_FILES = (
    ".gitignore",
    ".python-version",
    "AGENTS.md",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "book_agent/__init__.py",
    "book_agent/parsers/example.py",
    "docs/使用说明.md",
    "docs/安装说明.md",
    "docs/常见问题.md",
    "docs/隐私与数据存放.md",
    "install-macos.command",
    "installer/__init__.py",
    "installer/install_macos.py",
    "pyproject.toml",
    "third_party/model/LICENSE-MIT",
    "third_party/uv/LICENSE-APACHE",
    "third_party/uv/LICENSE-MIT",
    "uv.lock",
)

FIXED_METADATA = {
    "version": "0.1.0-beta.1",
    "tag": "v0.1.0-beta.1",
    "project": "codex-obsidian-book-library",
    "model_id": "intfloat/multilingual-e5-small",
    "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "uv_version": "0.11.26",
    "uv_sha256": "c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554",
    "python": "3.12",
    "archive": "codex-obsidian-book-library-v0.1.0-beta.1-macos-arm64-all-in-one.zip",
    "top_level_directory": "codex-obsidian-book-library-v0.1.0-beta.1-macos-arm64",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: str | bytes = "fixture\n", mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    path.chmod(mode)


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "public-project"
    for relative in SOURCE_FILES:
        content = f"fixture payload: {relative}\n"
        if relative == ".python-version":
            content = "3.12\n"
        elif relative == ".gitignore":
            content = ".venv/\ndist/\n"
        _write(project / relative, content)

    # These files are deliberately private or outside the public release allowlist.
    _write(project / "tests" / "test_private.py", "PRIVATE_TEST_SENTINEL\n")
    _write(project / ".codex" / "config.toml", "/Users/private/project\n")
    _write(project / "outputs" / "library.db", b"sqlite user data")
    _write(project / "vault" / "book.epub", b"private book")
    _write(project / "docs" / "plans" / "internal.md", "internal plan\n")
    _write(project / "installer" / "nested" / "not_allowed.py", "secret = True\n")
    _write(project / "third_party" / "uv" / "PRIVATE.txt", "not allowlisted\n")
    return project


def _make_model_snapshot(tmp_path: Path) -> Path:
    model_cache = tmp_path / "hf-cache" / "models--intfloat--multilingual-e5-small"
    blobs = model_cache / "blobs"
    snapshot = model_cache / "snapshots" / FIXED_METADATA["model_revision"]
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    for index, relative in enumerate(MODEL_FILES):
        blob = blobs / f"blob-{index:02d}"
        blob.write_bytes(f"model fixture {relative}\n".encode())
        source = snapshot / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.symlink_to(os.path.relpath(blob, source.parent))
    return snapshot


def _make_uv_and_metadata(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    # The NUL keeps this binary for privacy scanning; /Users/runner is expected in
    # some official build artifacts and must not be treated as the local user.
    uv_bytes = b"\xcf\xfa\xed\xfe\x00fake uv /Users/runner\x00"
    uv = tmp_path / "uv"
    _write(uv, uv_bytes, 0o755)
    metadata = dict(FIXED_METADATA)
    metadata["uv_sha256"] = _sha256(uv_bytes)
    metadata_path = tmp_path / "release.fixture.json"
    _write(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return uv, metadata_path, metadata


def _build_fixture(tmp_path: Path, output_name: str = "dist"):
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    result = build_macos_release.build_release(
        project_root=project,
        model_snapshot=snapshot,
        uv_binary=uv,
        output_dir=tmp_path / output_name,
        metadata_path=metadata_path,
    )
    return result, project, snapshot, metadata


def _make_release_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    project = _make_project(tmp_path)
    snapshot = _make_model_snapshot(tmp_path)
    uv, metadata_path, metadata = _make_uv_and_metadata(tmp_path)
    return project, snapshot, uv, metadata_path, metadata


def _write_zip_entries(
    archive_path: Path,
    entries: list[tuple[str, bytes, int]],
) -> None:
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, data)


def test_release_metadata_is_pinned_for_macos_arm64() -> None:
    metadata = json.loads(
        (PROJECT_ROOT / "distribution" / "release.json").read_text(encoding="utf-8")
    )
    assert metadata == FIXED_METADATA


def test_build_has_one_top_level_and_exact_allowlisted_payload(tmp_path: Path) -> None:
    result, _, _, metadata = _build_fixture(tmp_path)
    top = metadata["top_level_directory"]
    expected_payload = {
        *SOURCE_FILES,
        *(f"data/models/{relative}" for relative in MODEL_FILES),
        "bin/uv",
        "RELEASE-MANIFEST.json",
    }

    with zipfile.ZipFile(result.archive) as archive:
        names = archive.namelist()
        assert {name.split("/", 1)[0] for name in names} == {top}
        assert {name.removeprefix(f"{top}/") for name in names} == expected_payload
        assert all(not info.is_dir() for info in archive.infolist())

    assert result.archive.name == metadata["archive"]
    assert result.checksums.name == "SHA256SUMS"
    assert "PRIVATE_TEST_SENTINEL" not in result.archive.read_bytes().decode(
        "latin-1"
    )


def test_model_files_are_dereferenced_into_regular_zip_members(tmp_path: Path) -> None:
    result, _, _, metadata = _build_fixture(tmp_path)
    prefix = f'{metadata["top_level_directory"]}/data/models/'

    with zipfile.ZipFile(result.archive) as archive:
        for relative in MODEL_FILES:
            info = archive.getinfo(prefix + relative)
            unix_mode = info.external_attr >> 16
            assert stat.S_ISREG(unix_mode)
            assert not stat.S_ISLNK(unix_mode)
            assert archive.read(info) == f"model fixture {relative}\n".encode()


def test_manifest_checksum_and_zip_are_reproducible(tmp_path: Path) -> None:
    result_one, project, snapshot, metadata = _build_fixture(tmp_path, "dist-one")
    uv = tmp_path / "uv"
    metadata_path = tmp_path / "release.fixture.json"
    result_two = build_macos_release.build_release(
        project_root=project,
        model_snapshot=snapshot,
        uv_binary=uv,
        output_dir=tmp_path / "dist-two",
        metadata_path=metadata_path,
    )

    assert result_one.archive.read_bytes() == result_two.archive.read_bytes()
    assert result_one.checksums.read_bytes() == result_two.checksums.read_bytes()
    archive_hash = _sha256(result_one.archive.read_bytes())
    assert result_one.checksums.read_text(encoding="utf-8") == (
        f'{archive_hash}  {metadata["archive"]}\n'
    )

    manifest_name = f'{metadata["top_level_directory"]}/RELEASE-MANIFEST.json'
    with zipfile.ZipFile(result_one.archive) as archive:
        manifest = json.loads(archive.read(manifest_name))
        payload_names = [item["path"] for item in manifest["files"]]
        assert manifest["release"] == metadata
        assert payload_names == sorted(payload_names)
        assert "RELEASE-MANIFEST.json" not in payload_names
        for item in manifest["files"]:
            data = archive.read(f'{metadata["top_level_directory"]}/{item["path"]}')
            assert item == {
                "path": item["path"],
                "size": len(data),
                "sha256": _sha256(data),
                "mode": "0755" if item["path"] in {"install-macos.command", "bin/uv"} else "0644",
            }


def test_unzip_preserves_two_executables_and_normalizes_other_modes(
    tmp_path: Path,
) -> None:
    result, _, _, metadata = _build_fixture(tmp_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    subprocess.run(
        ["/usr/bin/unzip", "-qq", str(result.archive), "-d", str(extracted)],
        check=True,
    )
    root = extracted / metadata["top_level_directory"]

    for path in root.rglob("*"):
        if path.is_file():
            expected = 0o755 if path.relative_to(root).as_posix() in {
                "install-macos.command",
                "bin/uv",
            } else 0o644
            assert stat.S_IMODE(path.stat().st_mode) == expected


def test_missing_required_source_file_fails_with_its_relative_path(
    tmp_path: Path,
) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    missing = project / "docs" / "隐私与数据存放.md"
    missing.unlink()

    with pytest.raises(build_macos_release.ReleaseBuildError, match="隐私与数据存放.md"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


def test_model_snapshot_rejects_multiple_missing_files(tmp_path: Path) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    (snapshot / "config.json").unlink()
    (snapshot / "tokenizer.json").unlink()

    with pytest.raises(build_macos_release.ReleaseBuildError) as caught:
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )

    message = str(caught.value)
    assert "config.json" in message
    assert "tokenizer.json" in message


def test_model_snapshot_rejects_extra_file(tmp_path: Path) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    _write(snapshot / "unapproved.bin", b"extra")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="unapproved.bin"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


@pytest.mark.parametrize("broken", [False, True], ids=["escaping", "broken"])
def test_model_snapshot_rejects_unsafe_symlink(
    tmp_path: Path,
    broken: bool,
) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    source = snapshot / "config.json"
    source.unlink()
    outside = tmp_path / "outside-model-file"
    if not broken:
        _write(outside, b"outside")
    source.symlink_to(outside)

    with pytest.raises(build_macos_release.ReleaseBuildError, match="blobs"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


def test_uv_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    uv.write_bytes(uv.read_bytes() + b"tampered")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="SHA-256 mismatch"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


@pytest.mark.parametrize("as_symlink", [False, True], ids=["not-executable", "symlink"])
def test_uv_must_be_regular_and_executable(tmp_path: Path, as_symlink: bool) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    if as_symlink:
        target = tmp_path / "real-uv"
        uv.rename(target)
        uv.symlink_to(target)
    else:
        uv.chmod(0o644)

    with pytest.raises(build_macos_release.ReleaseBuildError, match="regular|executable"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


@pytest.mark.parametrize(
    "relative,leak",
    [
        ("README.md", "/Users/alice/private/library"),
        ("AGENTS.md", f"username={Path.home().name}"),
        ("docs/使用说明.md", "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456"),
        (
            "docs/常见问题.md",
            "webhook=https://open.feishu.cn/open-apis/bot/v2/hook/12345678-abcd-1234-abcd-1234567890ab",
        ),
        ("installer/install_macos.py", "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456"),
    ],
    ids=["users-path", "username", "api-key", "webhook", "bearer-token"],
)
def test_build_rejects_private_text_and_secrets(
    tmp_path: Path,
    relative: str,
    leak: str,
) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    _write(project / relative, leak + "\n")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="private|secret"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


def test_build_rejects_actual_project_home_and_model_source_paths(
    tmp_path: Path,
) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    _write(
        project / "README.md",
        f"project={project.absolute()}\nhome={Path.home()}\nmodel={snapshot.absolute()}\n",
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="private"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


def test_binary_rejects_current_private_marker_but_not_users_runner(
    tmp_path: Path,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    uv_bytes = (
        b"\xcf\xfa\xed\xfe\x00/Users/runner\x00"
        + Path.home().name.encode()
        + b"\x00"
    )
    uv.write_bytes(uv_bytes)
    uv.chmod(0o755)
    metadata["uv_sha256"] = _sha256(uv_bytes)
    _write(metadata_path, json.dumps(metadata, ensure_ascii=False) + "\n")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="private"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


def test_private_username_marker_is_derived_from_current_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_username = "release-owner-unique"
    monkeypatch.setattr(Path, "home", lambda: Path("/Users") / synthetic_username)
    staging = tmp_path / "staging"
    _write(
        staging / "release" / "bin" / "uv",
        b"\xcf\xfa\xed\xfe\x00" + synthetic_username.encode() + b"\x00",
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="private"):
        build_macos_release.scan_staging(staging)


@pytest.mark.parametrize(
    "relative",
    [
        "release/__MACOSX/junk",
        "release/.DS_Store",
        "release/._README.md",
        "release/tests/test_secret.py",
        "release/outputs/result.txt",
        "release/library.sqlite3",
        "release/private-book.PDF",
        "release/private-book.epub",
    ],
)
def test_staging_scan_rejects_junk_data_and_denylisted_directories(
    tmp_path: Path,
    relative: str,
) -> None:
    staging = tmp_path / "staging"
    _write(staging / relative, b"forbidden")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="forbidden"):
        build_macos_release.scan_staging(staging)


def test_staging_scan_rejects_private_path_and_secret_content(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    private_path = tmp_path / "private-source"
    _write(
        staging / "release" / "README.md",
        f"source={private_path}\napi_key=abcdefghijklmnopqrstuvwxyz123456\n",
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="private|secret"):
        build_macos_release.scan_staging(staging, private_paths=[private_path])


def test_staging_scan_rejects_sqlite_header_with_allowlisted_extension(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    _write(
        staging / "release" / "README.md",
        b"SQLite format 3\x00" + b"renamed database payload",
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="SQLite"):
        build_macos_release.scan_staging(staging)


@pytest.mark.parametrize(
    "member_name",
    [
        "release/../escape.txt",
        "/absolute.txt",
        r"C:\Users\alice\secret.txt",
    ],
)
def test_zip_scan_rejects_unsafe_member_paths(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive_path = tmp_path / "malicious.zip"
    _write_zip_entries(
        archive_path,
        [(member_name, b"bad", stat.S_IFREG | 0o644)],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="unsafe"):
        build_macos_release.scan_zip(archive_path)


def test_zip_scan_rejects_duplicate_normalized_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate.zip"
    _write_zip_entries(
        archive_path,
        [
            ("release/docs/guide.md", b"one", stat.S_IFREG | 0o644),
            ("release/docs//guide.md", b"two", stat.S_IFREG | 0o644),
        ],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="duplicate"):
        build_macos_release.scan_zip(archive_path)


def test_zip_scan_rejects_sqlite_header_with_allowlisted_extension(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "renamed-database.zip"
    _write_zip_entries(
        archive_path,
        [
            (
                "release/README.md",
                b"SQLite format 3\x00" + b"renamed database payload",
                stat.S_IFREG | 0o644,
            )
        ],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="SQLite"):
        build_macos_release.scan_zip(archive_path)


@pytest.mark.parametrize(
    "file_type",
    [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFBLK],
    ids=["symlink", "fifo", "character-device", "block-device"],
)
def test_zip_scan_rejects_non_regular_members(tmp_path: Path, file_type: int) -> None:
    archive_path = tmp_path / "special.zip"
    _write_zip_entries(
        archive_path,
        [("release/member", b"target", file_type | 0o755)],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="regular"):
        build_macos_release.scan_zip(archive_path)


def test_zip_scan_rejects_denylist_private_text_and_secret(tmp_path: Path) -> None:
    archive_path = tmp_path / "private.zip"
    private_path = tmp_path / "private-model-source"
    _write_zip_entries(
        archive_path,
        [
            ("release/tests/test_internal.py", b"internal", stat.S_IFREG | 0o644),
            (
                "release/README.md",
                f"{private_path}\nwebhook=https://hooks.slack.com/services/A/B/C\n".encode(),
                stat.S_IFREG | 0o644,
            ),
        ],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="forbidden|private|secret"):
        build_macos_release.scan_zip(archive_path, private_paths=[private_path])


def test_large_payload_paths_are_copied_hashed_and_verified_streamingly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    real_path_read_bytes = Path.read_bytes
    real_zip_read = zipfile.ZipFile.read

    def guarded_path_read_bytes(path: Path) -> bytes:
        normalized = path.as_posix()
        if path == uv or normalized.endswith("data/models/model.safetensors") or (
            path.name == "model.safetensors"
        ):
            raise AssertionError(f"large payload read_bytes is forbidden: {path}")
        return real_path_read_bytes(path)

    def guarded_zip_read(
        archive: zipfile.ZipFile,
        name: str | zipfile.ZipInfo,
        pwd: bytes | None = None,
    ) -> bytes:
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if filename.endswith("data/models/model.safetensors"):
            raise AssertionError(f"large ZIP member read is forbidden: {filename}")
        return real_zip_read(archive, name, pwd)

    monkeypatch.setattr(Path, "read_bytes", guarded_path_read_bytes)
    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_zip_read)

    build_macos_release.build_release(
        project_root=project,
        model_snapshot=snapshot,
        uv_binary=uv,
        output_dir=tmp_path / "dist",
        metadata_path=metadata_path,
    )


def test_generic_runner_home_is_not_a_binary_private_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_home = Path("/Users/runner")
    explicit_private_path = tmp_path / "real-private-model-source"
    monkeypatch.setattr(Path, "home", lambda: runner_home)
    staging = tmp_path / "staging"
    uv = staging / "release" / "bin" / "uv"
    _write(
        uv,
        b"\xcf\xfa\xed\xfe\x00/Users/runner/work/uv-build\x00",
    )

    build_macos_release.scan_staging(
        staging,
        private_paths=[runner_home, explicit_private_path],
    )

    _write(
        uv,
        b"\xcf\xfa\xed\xfe\x00" + str(explicit_private_path).encode() + b"\x00",
    )
    with pytest.raises(build_macos_release.ReleaseBuildError, match="private"):
        build_macos_release.scan_staging(
            staging,
            private_paths=[runner_home, explicit_private_path],
        )
