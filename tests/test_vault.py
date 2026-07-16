import errno
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

import book_agent.vault as vault_module
from book_agent.config import AppPaths
from book_agent.vault import VaultManager


def _source(
    root: Path,
    directory: str,
    name: str,
    content: str,
) -> Path:
    source = root / directory / name
    source.parent.mkdir(parents=True)
    source.write_text(content, encoding="utf-8")
    return source


def test_vault_manager_exposes_only_atomic_public_api() -> None:
    public_methods = {
        name
        for name, value in vars(VaultManager).items()
        if callable(value) and not name.startswith("_")
    }

    assert public_methods == {"ensure_layout", "import_original"}


def test_ensure_layout_creates_all_user_directories(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)

    VaultManager(paths).ensure_layout()

    assert paths.inbox.is_dir()
    assert paths.originals.is_dir()
    assert paths.parsed.is_dir()
    assert paths.notes.is_dir()
    assert paths.ocr_reports.is_dir()
    assert paths.models.is_dir()
    assert paths.database.parent.is_dir()


def test_ensure_layout_splits_external_vault_from_project_data(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    obsidian_vault = tmp_path / "Obsidian_workspace"
    obsidian_vault.mkdir()
    paths = AppPaths.from_root(project, vault_root=obsidian_vault)

    VaultManager(paths).ensure_layout()

    assert paths.vault == obsidian_vault
    for directory in (
        paths.inbox,
        paths.originals,
        paths.parsed,
        paths.notes,
        paths.ocr_reports,
    ):
        assert directory.is_dir()
        assert directory.is_relative_to(obsidian_vault)
    assert paths.models.is_dir()
    assert paths.database.parent.is_dir()
    assert paths.models.is_relative_to(project)
    assert paths.database.parent.is_relative_to(project)
    assert not (project / "vault").exists()


@pytest.mark.parametrize("replacement_kind", ["directory", "symlink"])
def test_managed_directory_rejects_leaf_identity_change_on_exit(
    tmp_path: Path,
    replacement_kind: str,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    manager = VaultManager(paths)
    manager.ensure_layout()
    displaced_notes = tmp_path / "displaced-notes"
    symlink_target = tmp_path / "notes-target"

    with pytest.raises(ValueError, match=r"notes.*identity"):
        with manager._managed_directory(
            paths.notes,
            "notes",
            create=False,
        ):
            paths.notes.rename(displaced_notes)
            if replacement_kind == "directory":
                paths.notes.mkdir()
            else:
                symlink_target.mkdir()
                paths.notes.symlink_to(symlink_target, target_is_directory=True)

    assert list(displaced_notes.iterdir()) == []
    if replacement_kind == "directory":
        assert list(paths.notes.iterdir()) == []
    else:
        assert list(symlink_target.iterdir()) == []


def test_ensure_layout_rejects_symlinked_external_vault_before_writing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "real-vault"
    target.mkdir()
    vault_alias = tmp_path / "vault-alias"
    vault_alias.symlink_to(target, target_is_directory=True)
    paths = AppPaths.from_root(project, vault_root=vault_alias)

    with pytest.raises(ValueError, match=r"vault root.*symlink"):
        VaultManager(paths).ensure_layout()

    assert list(target.iterdir()) == []
    assert not (project / "data").exists()
    assert not (project / "vault").exists()


def test_tracked_vault_docs_and_ignore_rules() -> None:
    project = Path(__file__).parents[1]
    homepage = (project / "vault" / "首页.md").read_text(encoding="utf-8")
    guide = (project / "vault" / "书库" / "说明.md").read_text(
        encoding="utf-8"
    )
    ignore_lines = (project / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert homepage.startswith("# ")
    assert "[[书库/说明]]" in homepage
    for directory in (
        "00-待导入",
        "10-原始书籍",
        "20-解析文本",
        "30-AI读书笔记",
    ):
        assert directory in guide
        assert (project / "vault" / "书库" / directory / ".gitkeep").is_file()
    assert "AI 读书笔记不属于原始证据，也不参与原始证据索引" in guide
    assert "/data/" in ignore_lines
    assert "data/" not in ignore_lines


def test_import_original_preserves_metadata_and_sequential_collisions(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    first_source = _source(tmp_path, "first", "book.txt", "first")
    second_source = _source(tmp_path, "second", "book.txt", "second")
    first_source.chmod(0o640)
    os.utime(first_source, ns=(1_500_000_000, 2_500_000_000))

    first = manager.import_original(first_source)
    second = manager.import_original(second_source)

    assert first.name == "book.txt"
    assert second.name == "book-2.txt"
    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"
    first_info = first.stat()
    assert stat.S_IMODE(first_info.st_mode) == 0o640
    assert first_info.st_mtime_ns == 2_500_000_000
    assert list(paths.inbox.iterdir()) == []


def test_import_post_link_interruption_removes_unregistered_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "interrupted import")
    real_link_final = manager._link_final
    real_unlink_all = manager._unlink_all_if_same
    cleanup_saw_live_temp = False

    def link_then_interrupt(
        inbox_fd: int,
        temp_name: str,
        originals_fd: int,
        source_name: str,
    ) -> str:
        real_link_final(
            inbox_fd,
            temp_name,
            originals_fd,
            source_name,
        )
        raise KeyboardInterrupt("interrupted after original link")

    def unlink_while_temp_is_live(
        directory_fd: int,
        expected: os.stat_result,
    ) -> None:
        nonlocal cleanup_saw_live_temp
        cleanup_saw_live_temp = any(
            (entry.lstat().st_dev, entry.lstat().st_ino)
            == (expected.st_dev, expected.st_ino)
            for entry in paths.inbox.iterdir()
        )
        assert cleanup_saw_live_temp
        real_unlink_all(directory_fd, expected)

    monkeypatch.setattr(manager, "_link_final", link_then_interrupt)
    monkeypatch.setattr(manager, "_unlink_all_if_same", unlink_while_temp_is_live)

    with pytest.raises(KeyboardInterrupt, match="after original link"):
        manager.import_original(source)

    assert list(paths.originals.iterdir()) == []
    assert list(paths.inbox.iterdir()) == []
    assert cleanup_saw_live_temp is True


def test_import_post_link_interruption_removes_renamed_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "renamed temp import")
    real_link_final = manager._link_final

    def link_then_rename_temp_and_interrupt(
        inbox_fd: int,
        temp_name: str,
        originals_fd: int,
        source_name: str,
    ) -> str:
        real_link_final(
            inbox_fd,
            temp_name,
            originals_fd,
            source_name,
        )
        os.rename(
            temp_name,
            "renamed-temp",
            src_dir_fd=inbox_fd,
            dst_dir_fd=inbox_fd,
        )
        raise KeyboardInterrupt("interrupted after temp rename")

    monkeypatch.setattr(
        manager,
        "_link_final",
        link_then_rename_temp_and_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="after temp rename"):
        manager.import_original(source)

    assert list(paths.originals.iterdir()) == []
    assert list(paths.inbox.iterdir()) == []


def test_import_post_link_stat_failure_removes_unregistered_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "stat failure import")
    real_stat = vault_module.os.stat
    real_unlink_if_same = manager._unlink_if_same
    failed = False
    published_cleanup_calls = 0

    def fail_published_stat_once(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal failed
        if not failed and path == "book.txt" and kwargs.get("dir_fd") is not None:
            failed = True
            raise OSError("published original stat unavailable")
        return real_stat(path, *args, **kwargs)

    def count_published_cleanup(
        directory_fd: int,
        name: str,
        expected: os.stat_result,
    ) -> None:
        nonlocal published_cleanup_calls
        if name == "book.txt":
            published_cleanup_calls += 1
        real_unlink_if_same(directory_fd, name, expected)

    monkeypatch.setattr(vault_module.os, "stat", fail_published_stat_once)
    monkeypatch.setattr(
        VaultManager,
        "_unlink_if_same",
        staticmethod(count_published_cleanup),
    )

    with pytest.raises(OSError, match="original stat unavailable"):
        manager.import_original(source)

    assert list(paths.originals.iterdir()) == []
    assert list(paths.inbox.iterdir()) == []
    assert published_cleanup_calls == 1


def test_import_post_link_stat_failure_removes_renamed_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "renamed import")
    real_stat = vault_module.os.stat
    renamed = False

    def rename_then_fail_published_stat(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal renamed
        directory_fd = kwargs.get("dir_fd")
        if not renamed and path == "book.txt" and isinstance(directory_fd, int):
            renamed = True
            os.rename(
                "book.txt",
                "moved.txt",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            raise OSError("published original stat unavailable after rename")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        vault_module.os,
        "stat",
        rename_then_fail_published_stat,
    )

    with pytest.raises(OSError, match="unavailable after rename"):
        manager.import_original(source)

    assert renamed is True
    assert list(paths.originals.iterdir()) == []
    assert list(paths.inbox.iterdir()) == []


def test_import_exit_validation_removes_renamed_original_while_temp_is_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "exit validation import")
    displaced_originals = tmp_path / "displaced-originals"
    real_stat = vault_module.os.stat
    swapped = False

    def rename_then_swap_after_published_stat(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal swapped
        directory_fd = kwargs.get("dir_fd")
        if not swapped and path == "book.txt" and isinstance(directory_fd, int):
            published = real_stat(path, *args, **kwargs)
            os.rename(
                "book.txt",
                "moved.txt",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            paths.originals.rename(displaced_originals)
            paths.originals.mkdir()
            swapped = True
            return published
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        vault_module.os,
        "stat",
        rename_then_swap_after_published_stat,
    )

    with pytest.raises(ValueError, match=r"originals.*identity"):
        manager.import_original(source)

    assert swapped is True
    assert list(displaced_originals.iterdir()) == []
    assert list(paths.originals.iterdir()) == []
    assert list(paths.inbox.iterdir()) == []


def test_concurrent_imports_preserve_24_same_basename_payloads(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    payloads = {f"payload-{index:02d}" for index in range(24)}
    sources = [
        _source(tmp_path, f"source-{index}", "book.txt", payload)
        for index, payload in enumerate(sorted(payloads))
    ]
    start = Barrier(len(sources))

    def import_book(source: Path) -> Path:
        start.wait(timeout=10)
        return manager.import_original(source)

    with ThreadPoolExecutor(max_workers=len(sources)) as executor:
        imported = list(executor.map(import_book, sources))

    assert len({path.name for path in imported}) == len(sources)
    assert {path.read_text(encoding="utf-8") for path in imported} == payloads
    assert len(list(paths.originals.iterdir())) == len(sources)
    assert list(paths.inbox.iterdir()) == []


@pytest.mark.parametrize("managed_name", ["inbox", "originals"])
def test_import_rejects_symlinked_managed_directory(
    tmp_path: Path,
    managed_name: str,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    managed = getattr(paths, managed_name)
    managed.rmdir()
    outside = tmp_path / f"outside-{managed_name}"
    outside.mkdir()
    managed.symlink_to(outside, target_is_directory=True)
    source = _source(tmp_path, "source", "book.txt", "secret")

    with pytest.raises(ValueError, match=managed_name):
        manager.import_original(source)

    assert list(outside.iterdir()) == []


def test_dangling_final_symlink_is_occupied(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    external = tmp_path / "external.txt"
    dangling = paths.originals / "book.txt"
    dangling.symlink_to(external)
    source = _source(tmp_path, "source", "book.txt", "inside")

    imported = manager.import_original(source)

    assert imported.name == "book-2.txt"
    assert imported.read_text(encoding="utf-8") == "inside"
    assert dangling.is_symlink()
    assert not external.exists()
    assert list(paths.inbox.iterdir()) == []


@pytest.mark.parametrize("source_kind", ["missing", "directory"])
def test_import_requires_existing_regular_source(
    tmp_path: Path,
    source_kind: str,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = tmp_path / source_kind
    if source_kind == "directory":
        source.mkdir()

    with pytest.raises(ValueError, match="Source"):
        manager.import_original(source)

    assert list(paths.inbox.iterdir()) == []
    assert list(paths.originals.iterdir()) == []


def test_import_rejects_source_symlink(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    target = _source(tmp_path, "source", "target.txt", "content")
    source_symlink = tmp_path / "book.txt"
    source_symlink.symlink_to(target)

    with pytest.raises(ValueError, match="Source"):
        manager.import_original(source_symlink)

    assert list(paths.inbox.iterdir()) == []
    assert list(paths.originals.iterdir()) == []


def test_import_fsyncs_complete_temp_before_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "content")
    synced_files: list[tuple[int, int]] = []
    original_fsync = os.fsync

    def record_fsync(file_descriptor: int) -> None:
        original_fsync(file_descriptor)
        file_info = os.fstat(file_descriptor)
        synced_files.append((file_info.st_dev, file_info.st_ino))

    monkeypatch.setattr(os, "fsync", record_fsync)

    imported = manager.import_original(source)

    imported_info = imported.stat()
    assert (imported_info.st_dev, imported_info.st_ino) in synced_files
    assert imported.read_text(encoding="utf-8") == "content"
    assert list(paths.inbox.iterdir()) == []


def test_silent_short_copy_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "0123456789abcdef")

    def copy_one_byte(source_stream: object, target_stream: object) -> None:
        target_stream.write(source_stream.read(1))

    monkeypatch.setattr(shutil, "copyfileobj", copy_one_byte)

    with pytest.raises(ValueError, match="copy|size"):
        manager.import_original(source)

    assert list(paths.inbox.iterdir()) == []
    assert list(paths.originals.iterdir()) == []


def test_source_change_during_copy_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "original")
    original_copy = shutil.copyfileobj

    def copy_then_change(source_stream: object, target_stream: object) -> None:
        original_copy(source_stream, target_stream)
        source.write_text("changed while copying", encoding="utf-8")

    monkeypatch.setattr(shutil, "copyfileobj", copy_then_change)

    with pytest.raises(ValueError, match="copy|changed"):
        manager.import_original(source)

    assert list(paths.inbox.iterdir()) == []
    assert list(paths.originals.iterdir()) == []


def test_copy_failure_leaves_no_temp_or_partial_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "content")

    def fail_after_one_byte(source_stream: object, target_stream: object) -> None:
        target_stream.write(source_stream.read(1))
        target_stream.flush()
        raise OSError("forced copy failure")

    monkeypatch.setattr(shutil, "copyfileobj", fail_after_one_byte)

    with pytest.raises(ValueError, match="copy|import"):
        manager.import_original(source)

    assert list(paths.inbox.iterdir()) == []
    assert list(paths.originals.iterdir()) == []


@pytest.mark.parametrize(
    ("error_number", "expected_message"),
    [(errno.EIO, "link"), (errno.EXDEV, "same filesystem")],
)
def test_link_failure_leaves_no_temp_or_partial_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    expected_message: str,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = _source(tmp_path, "source", "book.txt", "content")

    def fail_link(*args: object, **kwargs: object) -> None:
        raise OSError(error_number, "forced link failure")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(ValueError, match=expected_message):
        manager.import_original(source)

    assert list(paths.inbox.iterdir()) == []
    assert list(paths.originals.iterdir()) == []


def test_max_length_utf8_filename_collision_is_truncated_safely(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    name_max = os.pathconf(paths.originals, "PC_NAME_MAX")
    extension = ".txt"
    stem_bytes = name_max - len(extension.encode("utf-8"))
    source_name = (
        "书" * (stem_bytes // len("书".encode("utf-8")))
        + "a" * (stem_bytes % len("书".encode("utf-8")))
        + extension
    )
    first_source = _source(tmp_path, "first", source_name, "first")
    second_source = _source(tmp_path, "second", source_name, "second")

    first = manager.import_original(first_source)
    second = manager.import_original(second_source)

    assert len(os.fsencode(first.name)) == name_max
    assert len(os.fsencode(second.name)) <= name_max
    assert second.name.endswith("-2.txt")
    assert second.read_text(encoding="utf-8") == "second"
    assert list(paths.inbox.iterdir()) == []
