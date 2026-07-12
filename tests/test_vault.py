import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from book_agent.config import AppPaths
from book_agent.vault import VaultManager


def test_ensure_layout_creates_all_user_directories(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)

    VaultManager(paths).ensure_layout()

    assert paths.inbox.is_dir()
    assert paths.originals.is_dir()
    assert paths.parsed.is_dir()
    assert paths.notes.is_dir()
    assert paths.models.is_dir()
    assert paths.database.parent.is_dir()


def test_stage_and_promote_preserve_existing_file(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = tmp_path / "book.txt"
    source.write_text("first", encoding="utf-8")
    existing = paths.originals / "book.txt"
    existing.write_text("existing", encoding="utf-8")

    staged = manager.stage(source)
    promoted = manager.promote(staged)

    assert promoted.name == "book-2.txt"
    assert promoted.read_text(encoding="utf-8") == "first"
    assert existing.read_text(encoding="utf-8") == "existing"


def test_ensure_layout_rejects_symlinked_inbox_outside_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    paths = AppPaths.from_root(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.library.mkdir(parents=True)
    paths.inbox.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="inbox"):
        VaultManager(paths).ensure_layout()


def test_stage_rejects_symlinked_inbox_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    paths = AppPaths.from_root(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.library.mkdir(parents=True)
    paths.inbox.symlink_to(outside, target_is_directory=True)
    source = tmp_path / "book.txt"
    source.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="inbox"):
        VaultManager(paths).stage(source)

    assert not (outside / "book.txt").exists()


def test_promote_rejects_symlinked_originals_outside_project(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.originals.rmdir()
    paths.originals.symlink_to(outside, target_is_directory=True)
    staged = paths.inbox / "book.txt"
    staged.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="originals"):
        manager.promote(staged)

    assert staged.is_file()
    assert not (outside / "book.txt").exists()


def test_stage_treats_dangling_target_symlink_as_occupied(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    external_target = tmp_path / "external.txt"
    dangling = paths.inbox / "book.txt"
    dangling.symlink_to(external_target)
    source = tmp_path / "book.txt"
    source.write_text("inside", encoding="utf-8")

    staged = manager.stage(source)

    assert staged.name == "book-2.txt"
    assert staged.read_text(encoding="utf-8") == "inside"
    assert dangling.is_symlink()
    assert not external_target.exists()


def test_concurrent_stage_calls_reserve_distinct_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    first_source = tmp_path / "first" / "book.txt"
    second_source = tmp_path / "second" / "book.txt"
    first_source.parent.mkdir()
    second_source.parent.mkdir()
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")
    copy_barrier = Barrier(2)
    original_copyfileobj = shutil.copyfileobj

    def synchronized_copyfileobj(
        source: object,
        destination: object,
        length: int = 0,
    ) -> None:
        copy_barrier.wait(timeout=5)
        original_copyfileobj(source, destination, length)

    monkeypatch.setattr(shutil, "copyfileobj", synchronized_copyfileobj)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(manager.stage, first_source),
            executor.submit(manager.stage, second_source),
        )
        staged = [future.result(timeout=10) for future in futures]

    assert {path.name for path in staged} == {"book.txt", "book-2.txt"}
    assert {path.read_text(encoding="utf-8") for path in staged} == {
        "first",
        "second",
    }


def test_promote_rejects_directory_in_inbox(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    staged_directory = paths.inbox / "book"
    staged_directory.mkdir()

    with pytest.raises(ValueError, match="direct regular file"):
        manager.promote(staged_directory)

    assert staged_directory.is_dir()


def test_promote_rejects_non_direct_inbox_entry(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    nested = paths.inbox / "nested"
    nested.mkdir()
    staged = nested / "book.txt"
    staged.write_text("nested", encoding="utf-8")

    with pytest.raises(ValueError, match="direct regular file"):
        manager.promote(staged)

    assert staged.is_file()


def test_promote_rejects_symlinked_inbox_entry(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    real_file = paths.inbox / "real.txt"
    real_file.write_text("real", encoding="utf-8")
    staged_symlink = paths.inbox / "book.txt"
    staged_symlink.symlink_to(real_file.name)

    with pytest.raises(ValueError, match="direct regular file"):
        manager.promote(staged_symlink)

    assert real_file.is_file()
    assert staged_symlink.is_symlink()


def test_promote_treats_dangling_target_symlink_as_occupied(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    staged = paths.inbox / "book.txt"
    staged.write_text("inside", encoding="utf-8")
    external_target = tmp_path / "external.txt"
    dangling = paths.originals / "book.txt"
    dangling.symlink_to(external_target)

    promoted = manager.promote(staged)

    assert promoted.name == "book-2.txt"
    assert promoted.read_text(encoding="utf-8") == "inside"
    assert dangling.is_symlink()
    assert not external_target.exists()


def test_promote_cleans_candidate_if_staged_entry_changes_during_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    staged = paths.inbox / "book.txt"
    staged.write_text("original", encoding="utf-8")
    original_link = os.link

    def replace_then_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        os.unlink(source, dir_fd=src_dir_fd)
        replacement_fd = os.open(
            source,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
            dir_fd=src_dir_fd,
        )
        try:
            os.write(replacement_fd, b"replacement")
        finally:
            os.close(replacement_fd)
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", replace_then_link)

    with pytest.raises(ValueError, match="changed during promotion"):
        manager.promote(staged)

    assert staged.read_text(encoding="utf-8") == "replacement"
    assert list(paths.originals.iterdir()) == []


def test_promote_cleans_linked_symlink_if_staged_entry_changes_during_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    staged = paths.inbox / "book.txt"
    staged.write_text("original", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    original_link = os.link

    def replace_with_symlink_then_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        os.unlink(source, dir_fd=src_dir_fd)
        os.symlink(outside, source, dir_fd=src_dir_fd)
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", replace_with_symlink_then_link)

    with pytest.raises(ValueError, match="changed during promotion"):
        manager.promote(staged)

    assert staged.is_symlink()
    assert list(paths.originals.iterdir()) == []


def test_stage_cleans_reserved_file_if_destination_inspection_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = tmp_path / "book.txt"
    source.write_text("content", encoding="utf-8")
    original_fstat = os.fstat
    regular_file_calls = 0

    def fail_second_regular_file_fstat(file_descriptor: int) -> os.stat_result:
        nonlocal regular_file_calls
        file_info = original_fstat(file_descriptor)
        if stat.S_ISREG(file_info.st_mode):
            regular_file_calls += 1
            if regular_file_calls == 2:
                raise OSError("forced destination inspection failure")
        return file_info

    monkeypatch.setattr(os, "fstat", fail_second_regular_file_fstat)

    with pytest.raises(ValueError, match="reserved inbox target"):
        manager.stage(source)

    assert list(paths.inbox.iterdir()) == []


def test_stage_reports_source_inspection_failure_as_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = tmp_path / "book.txt"
    source.write_text("content", encoding="utf-8")
    original_fstat = os.fstat
    failed = False

    def fail_first_regular_file_fstat(file_descriptor: int) -> os.stat_result:
        nonlocal failed
        file_info = original_fstat(file_descriptor)
        if stat.S_ISREG(file_info.st_mode) and not failed:
            failed = True
            raise OSError("forced source inspection failure")
        return file_info

    monkeypatch.setattr(os, "fstat", fail_first_regular_file_fstat)

    with pytest.raises(ValueError, match="Source file"):
        manager.stage(source)

    assert list(paths.inbox.iterdir()) == []


def test_stage_uses_third_sequential_name_after_two_collisions(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = tmp_path / "book.txt"
    source.write_text("new", encoding="utf-8")
    first = paths.inbox / "book.txt"
    second = paths.inbox / "book-2.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    staged = manager.stage(source)

    assert staged.name == "book-3.txt"
    assert staged.read_text(encoding="utf-8") == "new"
    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"


def test_stage_expands_user_source_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    source = home / "book.txt"
    source.write_text("home", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()

    staged = manager.stage(Path("~/book.txt"))

    assert staged.read_text(encoding="utf-8") == "home"


def test_stage_reports_missing_source_as_value_error(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()

    with pytest.raises(ValueError, match="Source file"):
        manager.stage(tmp_path / "missing.txt")
