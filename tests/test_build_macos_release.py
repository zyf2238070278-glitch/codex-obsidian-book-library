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
    "version": "0.2.0-beta.1",
    "tag": "v0.2.0-beta.1",
    "project": "codex-obsidian-book-library",
    "model_id": "intfloat/multilingual-e5-small",
    "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "uv_version": "0.11.26",
    "uv_sha256": "c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554",
    "python": "3.12",
    "archive": "codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64-all-in-one.zip",
    "top_level_directory": "codex-obsidian-book-library-v0.2.0-beta.1-macos-arm64",
    "vision_helper_version": "0.1.0",
    "vision_schema_version": "1",
}

FIXED_MODEL_FILES = (
    {
        "path": "1_Pooling/config.json",
        "size": 200,
        "sha256": "987f7a67a38fa564c849bb5d277c52ab9088a84368fc0be31a354125aebb12a0",
    },
    {
        "path": "README.md",
        "size": 497538,
        "sha256": "0038de97aee16258cecbad7ffda4b4febd6953e747a00e0ddbc8e6ed241e9c1c",
    },
    {
        "path": "config.json",
        "size": 655,
        "sha256": "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    },
    {
        "path": "model.safetensors",
        "size": 470641600,
        "sha256": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
    },
    {
        "path": "modules.json",
        "size": 387,
        "sha256": "c6e29747481e8b5dd2b58401966aeac910de39092f90cda9a704b1545f902b04",
    },
    {
        "path": "sentence_bert_config.json",
        "size": 57,
        "sha256": "948201d8329907aae938fa62f9ceeed53f5694dacc2b87b9f3b78b37ee986529",
    },
    {
        "path": "sentencepiece.bpe.model",
        "size": 5069051,
        "sha256": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    },
    {
        "path": "special_tokens_map.json",
        "size": 167,
        "sha256": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    },
    {
        "path": "tokenizer.json",
        "size": 17082730,
        "sha256": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    },
    {
        "path": "tokenizer_config.json",
        "size": 443,
        "sha256": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
    },
)


@pytest.fixture(autouse=True)
def _inject_vision_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Supply a synthetic arm64 helper to legacy release fixtures.

    The helper is deliberately a temporary Mach-O-shaped file; no compiled
    binary or user artifact is added to the repository.
    """

    helper = tmp_path / "book-vision-ocr"
    helper.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 64)
    helper.chmod(0o755)

    def run_command(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        if Path(argv[0]).name == "lipo":
            return subprocess.CompletedProcess(argv, 0, "arm64\n", "")
        if Path(argv[0]).name == "codesign":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[-1:] == ["--capabilities"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps({"schema_version": 1, "languages": ["zh-Hans", "en-US"]}),
                "",
            )
        raise AssertionError(f"unexpected validation command: {argv}")

    original = build_macos_release.build_release

    def wrapped(*args: object, **kwargs: object) -> build_macos_release.BuildResult:
        kwargs.setdefault("vision_helper", helper)
        kwargs.setdefault("run_command", run_command)
        return original(*args, **kwargs)

    monkeypatch.setattr(build_macos_release, "build_release", wrapped)


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


def _write_fixture_model_manifest(project: Path, snapshot: Path) -> Path:
    records = []
    for relative in MODEL_FILES:
        data = (snapshot / relative).resolve(strict=True).read_bytes()
        records.append(
            {
                "path": relative,
                "size": len(data),
                "sha256": _sha256(data),
            }
        )
    manifest = {
        "model_id": FIXED_METADATA["model_id"],
        "model_revision": FIXED_METADATA["model_revision"],
        "files": records,
    }
    path = project / "distribution" / "model-manifest.json"
    _write(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path


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
    _write_fixture_model_manifest(project, snapshot)
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


def test_model_manifest_is_pinned_to_release_and_real_snapshot() -> None:
    manifest_path = PROJECT_ROOT / "distribution" / "model-manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "model_id": FIXED_METADATA["model_id"],
        "model_revision": FIXED_METADATA["model_revision"],
        "files": list(FIXED_MODEL_FILES),
    }


def test_build_requires_default_project_model_manifest(tmp_path: Path) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    (project / "distribution" / "model-manifest.json").unlink()

    with pytest.raises(build_macos_release.ReleaseBuildError, match="model manifest"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


def test_model_content_must_match_trusted_manifest_sha256(tmp_path: Path) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    target = (snapshot / "config.json").resolve(strict=True)
    original = target.read_bytes()
    target.write_bytes(b"x" + original[1:])
    assert target.stat().st_size == len(original)

    with pytest.raises(build_macos_release.ReleaseBuildError, match="model manifest|SHA-256"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "extra-top-level-key",
        "wrong-model-id",
        "wrong-revision",
        "duplicate-path",
        "unsafe-path",
        "missing-path",
        "invalid-size",
        "invalid-sha256",
        "extra-record-key",
    ],
)
def test_model_manifest_schema_is_strict(
    tmp_path: Path,
    corruption: str,
) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    manifest_path = project / "distribution" / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "extra-top-level-key":
        manifest["unexpected"] = True
    elif corruption == "wrong-model-id":
        manifest["model_id"] = "other/model"
    elif corruption == "wrong-revision":
        manifest["model_revision"] = "0" * 40
    elif corruption == "duplicate-path":
        manifest["files"].append(dict(manifest["files"][0]))
    elif corruption == "unsafe-path":
        manifest["files"][0]["path"] = "../config.json"
    elif corruption == "missing-path":
        manifest["files"].pop()
    elif corruption == "invalid-size":
        manifest["files"][0]["size"] = 0
    elif corruption == "invalid-sha256":
        manifest["files"][0]["sha256"] = "not-a-sha256"
    elif corruption == "extra-record-key":
        manifest["files"][0]["unexpected"] = True
    _write(manifest_path, json.dumps(manifest, ensure_ascii=False) + "\n")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="model manifest"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )


def test_build_has_one_top_level_and_exact_allowlisted_payload(tmp_path: Path) -> None:
    result, _, _, metadata = _build_fixture(tmp_path)
    top = metadata["top_level_directory"]
    expected_payload = {
        *SOURCE_FILES,
            *(f"data/models/{relative}" for relative in MODEL_FILES),
            "bin/uv",
            "bin/book-vision-ocr",
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
                "mode": "0755" if item["path"] in {"install-macos.command", "bin/uv", "bin/book-vision-ocr"} else "0644",
            }


def test_build_rejects_checksum_symlink_without_touching_victim(
    tmp_path: Path,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    victim = tmp_path / "victim.txt"
    _write(victim, "keep victim unchanged\n")
    (output / "SHA256SUMS").symlink_to(victim)

    with pytest.raises(build_macos_release.ReleaseBuildError, match="SHA256SUMS|symlink"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert victim.read_text(encoding="utf-8") == "keep victim unchanged\n"
    assert not (output / metadata["archive"]).exists()
    assert (output / "SHA256SUMS").is_symlink()


def test_build_rejects_checksum_directory_without_publishing_zip(
    tmp_path: Path,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    (output / "SHA256SUMS").mkdir(parents=True)

    with pytest.raises(build_macos_release.ReleaseBuildError, match="SHA256SUMS|regular"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert not (output / metadata["archive"]).exists()
    assert (output / "SHA256SUMS").is_dir()


@pytest.mark.parametrize("with_existing", [False, True], ids=["new", "overwrite"])
def test_checksum_publish_failure_rolls_back_both_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_existing: bool,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    if with_existing:
        _write(archive_path, b"old archive")
        _write(checksums_path, "old checksum\n")

    real_replace = os.replace
    failed = False

    def fail_second_publish(source: Path | str, destination: Path | str) -> None:
        nonlocal failed
        if Path(destination) == checksums_path and not failed:
            failed = True
            raise OSError("simulated checksum publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(build_macos_release.os, "replace", fail_second_publish)

    with pytest.raises(build_macos_release.ReleaseBuildError, match="publication"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    if with_existing:
        assert archive_path.read_bytes() == b"old archive"
        assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"
    else:
        assert not archive_path.exists()
        assert not checksums_path.exists()
    assert not any(path.name.startswith(".macos-release-") for path in output.iterdir())


def test_successful_overwrite_cleans_artifact_backups(tmp_path: Path) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"

    for _ in range(2):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert not any(path.name.startswith(".macos-release-") for path in output.iterdir())


def test_backup_preparation_failure_cleans_prior_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")
    real_create_artifact_backup = build_macos_release._create_artifact_backup
    invalidated = False

    def create_first_backup_then_invalidate_second_target(
        target: Path,
        backup_dir: Path,
        label: str,
    ) -> Path:
        nonlocal invalidated
        backup = real_create_artifact_backup(target, backup_dir, label)
        if Path(target) == archive_path:
            checksums_path.unlink()
            checksums_path.mkdir()
            invalidated = True
        return backup

    monkeypatch.setattr(
        build_macos_release,
        "_create_artifact_backup",
        create_first_backup_then_invalidate_second_target,
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="regular"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert invalidated
    assert archive_path.read_bytes() == b"old archive"
    assert not any(
        path.name.startswith(".macos-release-backup-")
        for path in output.iterdir()
    )


@pytest.mark.parametrize("with_existing", [False, True], ids=["new", "overwrite"])
@pytest.mark.parametrize(
    "tamper_point",
    ["after-archive-verification", "after-checksum-creation"],
)
def test_final_published_artifacts_reject_candidate_tamper_and_roll_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_existing: bool,
    tamper_point: str,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    if with_existing:
        _write(archive_path, b"old archive")
        _write(checksums_path, "old checksum\n")
    tampered = False

    if tamper_point == "after-archive-verification":
        real_verify_archive = build_macos_release._verify_archive

        def verify_then_tamper(*args, **kwargs) -> None:
            nonlocal tampered
            real_verify_archive(*args, **kwargs)
            if not tampered:
                candidate_archive = Path(args[0])
                candidate_archive.write_bytes(b"tampered after verification")
                candidate_archive.chmod(0o644)
                tampered = True

        monkeypatch.setattr(
            build_macos_release,
            "_verify_archive",
            verify_then_tamper,
        )
    else:
        real_publish_release_artifacts = (
            build_macos_release._publish_release_artifacts
        )

        def tamper_after_checksum_then_publish(**kwargs) -> None:
            nonlocal tampered
            candidate_archive = Path(kwargs["candidate_archive"])
            candidate_archive.write_bytes(b"tampered after checksum")
            candidate_archive.chmod(0o644)
            tampered = True
            real_publish_release_artifacts(**kwargs)

        monkeypatch.setattr(
            build_macos_release,
            "_publish_release_artifacts",
            tamper_after_checksum_then_publish,
        )

    with pytest.raises(
        build_macos_release.ReleaseBuildError,
        match="publication|ZIP|archive|checksum",
    ):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert tampered
    if with_existing:
        assert archive_path.read_bytes() == b"old archive"
        assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"
    else:
        assert not archive_path.exists()
        assert not checksums_path.exists()
    assert not any(
        path.name.startswith(".macos-release-backup-")
        for path in output.iterdir()
    )


def test_final_validation_rejects_matching_archive_and_checksum_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")
    real_verify_archive = build_macos_release._verify_archive
    verification_count = 0
    swapped = False

    def swap_after_final_archive_verification(*args, **kwargs) -> None:
        nonlocal verification_count, swapped
        real_verify_archive(*args, **kwargs)
        verification_count += 1
        if verification_count == 2:
            malicious_archive = b"matching malicious archive and checksum"
            archive_path.write_bytes(malicious_archive)
            checksums_path.write_text(
                f"{_sha256(malicious_archive)}  {archive_path.name}\n",
                encoding="utf-8",
            )
            swapped = True

    monkeypatch.setattr(
        build_macos_release,
        "_verify_archive",
        swap_after_final_archive_verification,
    )

    with pytest.raises(
        build_macos_release.ReleaseBuildError,
        match="publication|changed|archive|checksum",
    ):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert swapped
    assert archive_path.read_bytes() == b"old archive"
    assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"
    assert not any(
        path.name.startswith(".macos-release-backup-")
        for path in output.iterdir()
    )


def test_unexpected_final_validator_error_rolls_back_existing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")
    real_verify_archive = build_macos_release._verify_archive
    verification_count = 0

    def fail_final_archive_verification(*args, **kwargs) -> None:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            raise zipfile.BadZipFile("simulated final ZIP race")
        real_verify_archive(*args, **kwargs)

    monkeypatch.setattr(
        build_macos_release,
        "_verify_archive",
        fail_final_archive_verification,
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="publication"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert verification_count == 2
    assert archive_path.read_bytes() == b"old archive"
    assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"
    assert not any(
        path.name.startswith(".macos-release-backup-")
        for path in output.iterdir()
    )


def test_final_validation_is_bound_to_published_file_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")
    real_verify_archive = build_macos_release._verify_archive
    real_publish_release_artifacts = build_macos_release._publish_release_artifacts
    trusted_archive = b""
    verification_count = 0
    decoy_used = False

    def verify_with_temporary_decoy(*args, **kwargs) -> None:
        nonlocal verification_count, decoy_used
        verification_count += 1
        if verification_count != 2:
            real_verify_archive(*args, **kwargs)
            return

        real_output = tmp_path / "real-dist-during-validation"
        output.rename(real_output)
        output.mkdir()
        _write(archive_path, trusted_archive)
        _write(
            checksums_path,
            f"{_sha256(trusted_archive)}  {archive_path.name}\n",
        )
        decoy_used = True
        try:
            real_verify_archive(*args, **kwargs)
        finally:
            archive_path.unlink()
            checksums_path.unlink()
            output.rmdir()
            real_output.rename(output)

    def publish_matching_malicious_candidates(**kwargs) -> None:
        nonlocal trusted_archive
        candidate_archive = Path(kwargs["candidate_archive"])
        candidate_checksums = Path(kwargs["candidate_checksums"])
        trusted_archive = candidate_archive.read_bytes()
        malicious_archive = b"matching malicious non-ZIP payload"
        candidate_archive.write_bytes(malicious_archive)
        candidate_archive.chmod(0o644)
        candidate_checksums.write_text(
            f"{_sha256(malicious_archive)}  {archive_path.name}\n",
            encoding="utf-8",
        )
        candidate_checksums.chmod(0o644)
        real_publish_release_artifacts(**kwargs)

    monkeypatch.setattr(
        build_macos_release,
        "_verify_archive",
        verify_with_temporary_decoy,
    )
    monkeypatch.setattr(
        build_macos_release,
        "_publish_release_artifacts",
        publish_matching_malicious_candidates,
    )

    with pytest.raises(
        build_macos_release.ReleaseBuildError,
        match="publication|ZIP|archive|checksum",
    ):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert verification_count == 2
    assert decoy_used
    assert archive_path.read_bytes() == b"old archive"
    assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"
    assert not any(
        path.name.startswith(".macos-release-backup-")
        for path in output.iterdir()
    )


def test_rollback_failure_retains_old_archive_recovery_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")

    real_replace = os.replace
    publication_failed = False
    rollback_failed = False

    def fail_publication_and_rollback(
        source: Path | str,
        destination: Path | str,
    ) -> None:
        nonlocal publication_failed, rollback_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == checksums_path and not publication_failed:
            publication_failed = True
            raise OSError("simulated checksum publication failure")
        if (
            destination_path == archive_path
            and source_path.name != metadata["archive"]
            and not rollback_failed
        ):
            rollback_failed = True
            raise OSError("simulated archive rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        build_macos_release.os,
        "replace",
        fail_publication_and_rollback,
    )

    with pytest.raises(
        build_macos_release.ReleaseBuildError,
        match="rollback failed",
    ) as caught:
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert publication_failed
    assert rollback_failed
    recovery_files = [
        path
        for path in output.iterdir()
        if path.name.startswith(".macos-release-backup-")
    ]
    assert len(recovery_files) == 1
    recovery = recovery_files[0]
    assert stat.S_ISREG(recovery.lstat().st_mode)
    assert not recovery.is_symlink()
    assert recovery.read_bytes() == b"old archive"
    assert str(recovery) in str(caught.value)
    assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"


def test_post_restore_validation_failure_retains_old_archive_recovery_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")

    real_replace = os.replace
    real_validate_output_target = build_macos_release._validate_output_target
    publication_failed = False
    restore_replaced = False
    validation_failed = False

    def fail_checksum_publication(
        source: Path | str,
        destination: Path | str,
    ) -> None:
        nonlocal publication_failed, restore_replaced
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == checksums_path and not publication_failed:
            publication_failed = True
            raise OSError("simulated checksum publication failure")
        real_replace(source, destination)
        if (
            destination_path == archive_path
            and source_path.name != metadata["archive"]
        ):
            restore_replaced = True

    def fail_restored_archive_validation(path: Path, label: str) -> bool:
        nonlocal validation_failed
        if Path(path) == archive_path and restore_replaced and not validation_failed:
            validation_failed = True
            raise build_macos_release.ReleaseBuildError(
                "simulated restored archive validation failure"
            )
        return real_validate_output_target(path, label)

    monkeypatch.setattr(
        build_macos_release.os,
        "replace",
        fail_checksum_publication,
    )
    monkeypatch.setattr(
        build_macos_release,
        "_validate_output_target",
        fail_restored_archive_validation,
    )

    with pytest.raises(
        build_macos_release.ReleaseBuildError,
        match="rollback failed",
    ) as caught:
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert publication_failed
    assert restore_replaced
    assert validation_failed
    recovery_files = [
        path
        for path in output.iterdir()
        if path.name.startswith(".macos-release-backup-")
    ]
    assert len(recovery_files) == 1
    recovery = recovery_files[0]
    assert stat.S_ISREG(recovery.lstat().st_mode)
    assert not recovery.is_symlink()
    assert recovery.read_bytes() == b"old archive"
    assert archive_path.read_bytes() == b"old archive"
    assert str(recovery) in str(caught.value)
    assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"


def test_rollback_rejects_regular_file_swap_and_retains_old_archive_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    attacker_archive = tmp_path / "attacker-archive"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")
    _write(attacker_archive, b"attacker archive")
    real_replace = os.replace
    publication_failed = False
    rollback_swapped = False

    def fail_publication_and_swap_restored_archive(
        source: Path | str,
        destination: Path | str,
    ) -> None:
        nonlocal publication_failed, rollback_swapped
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == checksums_path and not publication_failed:
            publication_failed = True
            raise OSError("simulated checksum publication failure")
        real_replace(source, destination)
        if (
            destination_path == archive_path
            and source_path.name.startswith(".macos-release-backup-")
            and not rollback_swapped
        ):
            real_replace(attacker_archive, archive_path)
            rollback_swapped = True

    monkeypatch.setattr(
        build_macos_release.os,
        "replace",
        fail_publication_and_swap_restored_archive,
    )

    with pytest.raises(build_macos_release.ReleaseBuildError) as caught:
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert publication_failed
    assert rollback_swapped
    assert "rollback failed" in str(caught.value)
    recovery_files = [
        path
        for path in output.iterdir()
        if path.name.startswith(".macos-release-backup-")
    ]
    assert len(recovery_files) == 1
    recovery = recovery_files[0]
    assert recovery.read_bytes() == b"old archive"
    assert str(recovery) in str(caught.value)
    assert archive_path.read_bytes() == b"attacker archive"
    assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"


def test_rollback_uses_bound_old_bytes_when_backup_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")
    real_create_artifact_backup = build_macos_release._create_artifact_backup
    real_replace = os.replace
    backup_replaced = False
    publication_failed = False

    def create_then_replace_archive_backup(
        target: Path,
        backup_dir: Path,
        label: str,
    ):
        nonlocal backup_replaced
        backup = real_create_artifact_backup(target, backup_dir, label)
        if Path(target) == archive_path:
            backup_path = Path(getattr(backup, "path", backup))
            attacker_backup = tmp_path / "attacker-backup"
            _write(attacker_backup, b"attacker archive")
            real_replace(attacker_backup, backup_path)
            backup_replaced = True
        return backup

    def fail_checksum_publication(
        source: Path | str,
        destination: Path | str,
    ) -> None:
        nonlocal publication_failed
        if Path(destination) == checksums_path and not publication_failed:
            publication_failed = True
            raise OSError("simulated checksum publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        build_macos_release,
        "_create_artifact_backup",
        create_then_replace_archive_backup,
    )
    monkeypatch.setattr(
        build_macos_release.os,
        "replace",
        fail_checksum_publication,
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="publication"):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert backup_replaced
    assert publication_failed
    assert archive_path.read_bytes() == b"old archive"
    assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"
    assert not any(
        path.name.startswith(".macos-release-backup-")
        for path in output.iterdir()
    )


def test_rollback_failure_rematerializes_replaced_backup_from_bound_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")
    real_create_artifact_backup = build_macos_release._create_artifact_backup
    real_replace = os.replace
    backup_replaced = False
    publication_failed = False
    rollback_failed = False

    def create_then_replace_archive_backup(
        target: Path,
        backup_dir: Path,
        label: str,
    ):
        nonlocal backup_replaced
        backup = real_create_artifact_backup(target, backup_dir, label)
        if Path(target) == archive_path:
            attacker_backup = tmp_path / "attacker-backup"
            _write(attacker_backup, b"attacker archive")
            real_replace(attacker_backup, backup.path)
            backup_replaced = True
        return backup

    def fail_publication_and_archive_rollback(
        source: Path | str,
        destination: Path | str,
    ) -> None:
        nonlocal publication_failed, rollback_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == checksums_path and not publication_failed:
            publication_failed = True
            raise OSError("simulated checksum publication failure")
        if (
            destination_path == archive_path
            and source_path.name.startswith(".macos-release-backup-")
            and not rollback_failed
        ):
            rollback_failed = True
            raise OSError("simulated archive rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        build_macos_release,
        "_create_artifact_backup",
        create_then_replace_archive_backup,
    )
    monkeypatch.setattr(
        build_macos_release.os,
        "replace",
        fail_publication_and_archive_rollback,
    )

    with pytest.raises(
        build_macos_release.ReleaseBuildError,
        match="rollback failed",
    ) as caught:
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    assert backup_replaced
    assert publication_failed
    assert rollback_failed
    recovery_files = [
        path
        for path in output.iterdir()
        if path.name.startswith(".macos-release-backup-")
    ]
    assert len(recovery_files) == 1
    recovery = recovery_files[0]
    assert recovery.read_bytes() == b"old archive"
    assert str(recovery) in str(caught.value)
    assert archive_path.read_bytes() != b"attacker archive"
    assert checksums_path.read_text(encoding="utf-8") == "old checksum\n"


def test_rollback_cleanup_failure_does_not_mask_recovery_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, snapshot, uv, metadata_path, metadata = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive_path = output / metadata["archive"]
    checksums_path = output / "SHA256SUMS"
    _write(archive_path, b"old archive")
    _write(checksums_path, "old checksum\n")
    real_replace = os.replace
    real_unlink = Path.unlink
    publication_failed = False
    rollback_failed = False
    cleanup_failed = False

    def fail_publication_and_rollback(
        source: Path | str,
        destination: Path | str,
    ) -> None:
        nonlocal publication_failed, rollback_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == checksums_path and not publication_failed:
            publication_failed = True
            raise OSError("simulated checksum publication failure")
        if (
            destination_path == archive_path
            and source_path.name.startswith(".macos-release-backup-")
            and not rollback_failed
        ):
            rollback_failed = True
            raise OSError("simulated archive rollback failure")
        real_replace(source, destination)

    def fail_unused_checksum_backup_cleanup(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        nonlocal cleanup_failed
        if (
            path.name.startswith(".macos-release-backup-")
            and path.exists()
            and path.read_bytes() == b"old checksum\n"
        ):
            cleanup_failed = True
            raise OSError("simulated checksum backup cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(
        build_macos_release.os,
        "replace",
        fail_publication_and_rollback,
    )
    monkeypatch.setattr(Path, "unlink", fail_unused_checksum_backup_cleanup)

    with pytest.raises(build_macos_release.ReleaseBuildError) as caught:
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=output,
            metadata_path=metadata_path,
        )

    message = str(caught.value)
    assert publication_failed
    assert rollback_failed
    assert cleanup_failed
    assert "rollback failed" in message
    assert "backup cleanup failed" in message
    archive_backups = [
        path
        for path in output.iterdir()
        if path.name.startswith(".macos-release-backup-")
        and path.read_bytes() == b"old archive"
    ]
    assert len(archive_backups) == 1
    assert str(archive_backups[0]) in message


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
                    "bin/book-vision-ocr",
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


@pytest.mark.parametrize("payload", [*MODEL_FILES, "uv"])
@pytest.mark.parametrize(
    "tamper_point",
    ["source-before-copy", "staging-after-copy"],
)
def test_trusted_payload_copy_rejects_same_size_race_and_staging_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    tamper_point: str,
) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    real_copy_file = build_macos_release._copy_file
    tampered = False

    def tampering_copy(source: Path, destination: Path, mode: int) -> None:
        nonlocal tampered
        destination_path = Path(destination)
        is_target = (
            payload != "uv"
            and destination_path.as_posix().endswith(f"data/models/{payload}")
        ) or (
            payload == "uv"
            and destination_path.as_posix().endswith("bin/uv")
        )
        if not is_target:
            real_copy_file(source, destination, mode)
            return

        if tamper_point == "source-before-copy":
            original = Path(source).read_bytes()
            Path(source).write_bytes(b"x" + original[1:])
            assert Path(source).stat().st_size == len(original)
            real_copy_file(source, destination, mode)
        else:
            real_copy_file(source, destination, mode)
            original = destination_path.read_bytes()
            destination_path.write_bytes(b"x" + original[1:])
            assert destination_path.stat().st_size == len(original)
        tampered = True

    monkeypatch.setattr(build_macos_release, "_copy_file", tampering_copy)

    with pytest.raises(
        build_macos_release.ReleaseBuildError,
        match="model|uv|SHA-256|size",
    ):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )

    assert tampered


@pytest.mark.parametrize("payload", [*MODEL_FILES, "uv"])
def test_archive_rejects_trusted_staging_tamper_after_immediate_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    project, snapshot, uv, metadata_path, _ = _make_release_inputs(tmp_path)
    real_verify_staged_file = build_macos_release._verify_staged_file
    tampered = False

    def verify_then_tamper(
        path: Path,
        *,
        expected_size: int,
        expected_sha256: str,
        label: str,
    ) -> None:
        nonlocal tampered
        real_verify_staged_file(
            path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=label,
        )
        path_hint = Path(path).as_posix()
        is_target = (
            payload != "uv" and path_hint.endswith(f"data/models/{payload}")
        ) or (payload == "uv" and path_hint.endswith("bin/uv"))
        if is_target:
            original = Path(path).read_bytes()
            Path(path).write_bytes(b"x" + original[1:])
            assert Path(path).stat().st_size == len(original)
            tampered = True

    monkeypatch.setattr(
        build_macos_release,
        "_verify_staged_file",
        verify_then_tamper,
    )

    with pytest.raises(
        build_macos_release.ReleaseBuildError,
        match="trusted|model|uv|SHA-256|size",
    ):
        build_macos_release.build_release(
            project_root=project,
            model_snapshot=snapshot,
            uv_binary=uv,
            output_dir=tmp_path / "dist",
            metadata_path=metadata_path,
        )

    assert tampered


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
        (
            "docs/使用说明.md",
            "OPENAI_API_KEY=" + "sk-proj-" + "abcdefghijklmnopqrstuvwxyz123456",
        ),
        (
            "docs/常见问题.md",
            "webhook=https://open.feishu.cn/open-apis/bot/v2/"
            + "hook/12345678-abcd-1234-abcd-1234567890ab",
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


def test_staging_scan_treats_allowlisted_markdown_with_nul_as_text(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    _write(
        staging / "release" / "README.md",
        b"\x00/Users/alice/private\napi_key=abcdefghijklmnopqrstuvwxyz123456\n",
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="private|secret"):
        build_macos_release.scan_staging(staging)


def test_staging_scan_rejects_invalid_utf8_in_allowlisted_text(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _write(staging / "release" / "README.md", b"\xffinvalid utf-8")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="UTF-8"):
        build_macos_release.scan_staging(staging)


@pytest.mark.parametrize(
    "private_path",
    [
        "C:/Users/alice/private",
        r"c:\users\alice\private",
        r"C:\\Users\\alice\\private",
        r"C:\\\\Users\\\\alice\\\\private",
        "/users/alice/private",
        "/HOME/alice/private",
    ],
    ids=[
        "windows-forward",
        "windows-backslash",
        "windows-json-escaped",
        "windows-multiply-escaped",
        "lowercase-users",
        "uppercase-home",
    ],
)
def test_staging_scan_rejects_windows_private_path_variants(
    tmp_path: Path,
    private_path: str,
) -> None:
    staging = tmp_path / "staging"
    _write(staging / "release" / "README.md", private_path + "\n")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="private"):
        build_macos_release.scan_staging(staging)


def test_staging_scan_allows_only_the_three_binary_payload_paths(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    binary_payloads = {
        "release/bin/uv": (
            b"\xcf\xfa\xed\xfe\x00/Users/runner/build\x00"
            b"C:\\\\Users\\\\runner\\\\build\x00/HOME/runner/build\x00"
        ),
        "release/data/models/model.safetensors": b"\xff\x00safetensors",
        "release/data/models/sentencepiece.bpe.model": b"\x80\x00sentencepiece",
    }
    for relative, data in binary_payloads.items():
        _write(staging / relative, data)

    build_macos_release.scan_staging(staging)


@pytest.mark.parametrize(
    "relative",
    ["release/bin/uv.bak", "release/nested/bin/uv", r"release\bin\uv"],
    ids=["suffix", "nested", "posix-backslash-filename"],
)
def test_staging_binary_allowlist_rejects_near_miss_paths(
    tmp_path: Path,
    relative: str,
) -> None:
    staging = tmp_path / "staging"
    _write(staging / relative, b"\xffnot allowlisted binary")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="UTF-8"):
        build_macos_release.scan_staging(staging)


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
    "suffix",
    [".sqlite-wal", ".sqlite-shm", ".sqlite-journal", ".db-journal"],
)
def test_staging_scan_rejects_sqlite_sidecar_suffixes(
    tmp_path: Path,
    suffix: str,
) -> None:
    staging = tmp_path / "staging"
    _write(staging / "release" / f"library{suffix}", b"sidecar payload")

    with pytest.raises(build_macos_release.ReleaseBuildError, match="forbidden"):
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
    "suffix",
    [".sqlite-wal", ".sqlite-shm", ".sqlite-journal", ".db-journal"],
)
def test_zip_scan_rejects_sqlite_sidecar_suffixes(
    tmp_path: Path,
    suffix: str,
) -> None:
    archive_path = tmp_path / "sqlite-sidecar.zip"
    _write_zip_entries(
        archive_path,
        [
            (
                f"release/library{suffix}",
                b"sidecar payload",
                stat.S_IFREG | 0o644,
            )
        ],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="forbidden"):
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


def test_zip_scan_treats_allowlisted_markdown_with_nul_as_text(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "nul-markdown.zip"
    _write_zip_entries(
        archive_path,
        [
            (
                "release/README.md",
                b"\x00/Users/alice/private\napi_key=abcdefghijklmnopqrstuvwxyz123456\n",
                stat.S_IFREG | 0o644,
            )
        ],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="private|secret"):
        build_macos_release.scan_zip(archive_path)


def test_zip_scan_rejects_invalid_utf8_in_allowlisted_text(tmp_path: Path) -> None:
    archive_path = tmp_path / "invalid-text.zip"
    _write_zip_entries(
        archive_path,
        [("release/README.md", b"\xffinvalid utf-8", stat.S_IFREG | 0o644)],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="UTF-8"):
        build_macos_release.scan_zip(archive_path)


@pytest.mark.parametrize(
    "private_path",
    [
        "C:/Users/alice/private",
        r"c:\users\alice\private",
        r"C:\\Users\\alice\\private",
        r"C:\\\\Users\\\\alice\\\\private",
        "/users/alice/private",
        "/HOME/alice/private",
    ],
    ids=[
        "windows-forward",
        "windows-backslash",
        "windows-json-escaped",
        "windows-multiply-escaped",
        "lowercase-users",
        "uppercase-home",
    ],
)
def test_zip_scan_rejects_windows_private_path_variants(
    tmp_path: Path,
    private_path: str,
) -> None:
    archive_path = tmp_path / "windows-private-path.zip"
    _write_zip_entries(
        archive_path,
        [
            (
                "release/README.md",
                (private_path + "\n").encode(),
                stat.S_IFREG | 0o644,
            )
        ],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="private"):
        build_macos_release.scan_zip(archive_path)


def test_zip_scan_allows_only_the_three_binary_payload_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "binary-payloads.zip"
    _write_zip_entries(
        archive_path,
        [
            (
                "release/bin/uv",
                (
                    b"\xcf\xfa\xed\xfe\x00/Users/runner/build\x00"
                    b"C:\\\\Users\\\\runner\\\\build\x00/HOME/runner/build\x00"
                ),
                stat.S_IFREG | 0o755,
            ),
            (
                "release/data/models/model.safetensors",
                b"\xff\x00safetensors",
                stat.S_IFREG | 0o644,
            ),
            (
                "release/data/models/sentencepiece.bpe.model",
                b"\x80\x00sentencepiece",
                stat.S_IFREG | 0o644,
            ),
        ],
    )

    build_macos_release.scan_zip(archive_path)


@pytest.mark.parametrize(
    "member_name",
    ["release/bin/uv.bak", "release/nested/bin/uv"],
    ids=["suffix", "nested"],
)
def test_zip_binary_allowlist_rejects_near_miss_paths(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive_path = tmp_path / "near-miss-binary.zip"
    _write_zip_entries(
        archive_path,
        [(member_name, b"\xffnot allowlisted binary", stat.S_IFREG | 0o644)],
    )

    with pytest.raises(build_macos_release.ReleaseBuildError, match="UTF-8"):
        build_macos_release.scan_zip(archive_path)


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
