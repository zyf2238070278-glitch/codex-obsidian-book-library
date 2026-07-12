import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

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
    source = tmp_path / "book.txt"
    source.write_text("secret", encoding="utf-8")
    staged = manager.stage(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.originals.rmdir()
    paths.originals.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="originals"):
        manager.promote(staged)

    assert staged.is_file()
    assert not (outside / "book.txt").exists()
    paths.originals.unlink()
    paths.originals.mkdir()

    promoted = manager.promote(staged)

    assert promoted.read_text(encoding="utf-8") == "secret"


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

    with pytest.raises(ValueError, match="created by this VaultManager"):
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

    with pytest.raises(ValueError, match="created by this VaultManager"):
        manager.promote(staged)

    assert staged.is_file()


def test_promote_rejects_symlinked_inbox_entry(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = tmp_path / "book.txt"
    source.write_text("original", encoding="utf-8")
    staged_symlink = manager.stage(source)
    real_file = paths.inbox / "real.txt"
    real_file.write_text("real", encoding="utf-8")
    staged_symlink.unlink()
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
    source = tmp_path / "book.txt"
    source.write_text("inside", encoding="utf-8")
    staged = manager.stage(source)
    external_target = tmp_path / "external.txt"
    dangling = paths.originals / "book.txt"
    dangling.symlink_to(external_target)

    promoted = manager.promote(staged)

    assert promoted.name == "book-2.txt"
    assert promoted.read_text(encoding="utf-8") == "inside"
    assert dangling.is_symlink()
    assert not external_target.exists()


def test_concurrent_promote_calls_claim_staged_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")
    staged = manager.stage(source)
    staged_info = staged.stat()
    inbox_info = paths.inbox.stat()
    public_unlink_barrier = Barrier(2)
    attempts_lock = Lock()
    staged_unlink_attempts: list[str] = []
    original_unlink = os.unlink

    def synchronized_unlink(
        path: str,
        *args: object,
        dir_fd: int | None = None,
        **kwargs: object,
    ) -> None:
        if dir_fd is not None:
            directory_info = os.fstat(dir_fd)
            try:
                entry_info = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
            except FileNotFoundError:
                entry_info = None
            if (
                entry_info is not None
                and (directory_info.st_dev, directory_info.st_ino)
                == (inbox_info.st_dev, inbox_info.st_ino)
                and (entry_info.st_dev, entry_info.st_ino)
                == (staged_info.st_dev, staged_info.st_ino)
            ):
                with attempts_lock:
                    staged_unlink_attempts.append(path)
                if path == staged.name:
                    public_unlink_barrier.wait(timeout=5)
        original_unlink(path, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(os, "unlink", synchronized_unlink)

    def promote(manager: VaultManager) -> Path | ValueError:
        try:
            return manager.promote(staged)
        except ValueError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(promote, (manager, manager)))

    successes = [result for result in results if isinstance(result, Path)]
    failures = [result for result in results if isinstance(result, ValueError)]
    originals = list(paths.originals.iterdir())
    assert len(staged_unlink_attempts) == 1
    assert len(successes) == 1
    assert len(failures) == 1
    assert "already" in str(failures[0])
    assert len(originals) == 1
    assert originals[0].read_text(encoding="utf-8") == "payload"
    assert list(paths.inbox.iterdir()) == []


def test_promote_cleans_candidate_if_staged_entry_changes_during_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = tmp_path / "book.txt"
    source.write_text("original", encoding="utf-8")
    staged = manager.stage(source)
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
    source = tmp_path / "book.txt"
    source.write_text("original", encoding="utf-8")
    staged = manager.stage(source)
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


def test_promote_rejects_manually_dropped_regular_file(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    manual = paths.inbox / "manual.txt"
    manual.write_text("manual", encoding="utf-8")

    with pytest.raises(ValueError, match="created by this VaultManager"):
        manager.promote(manual)

    assert manual.read_text(encoding="utf-8") == "manual"


def test_promote_rejects_file_staged_by_another_manager(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    owner = VaultManager(paths)
    other = VaultManager(paths)
    owner.ensure_layout()
    source = tmp_path / "book.txt"
    source.write_text("owned", encoding="utf-8")
    staged = owner.stage(source)

    with pytest.raises(ValueError, match="created by this VaultManager"):
        other.promote(staged)

    promoted = owner.promote(staged)
    assert promoted.read_text(encoding="utf-8") == "owned"


def test_stale_promotes_do_not_consume_later_same_name_generations(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    paths.root.mkdir()
    manager = VaultManager(paths)
    manager.ensure_layout()
    payloads = {f"payload-{index}" for index in range(24)}
    staged_paths: list[Path] = []
    promoted_paths: list[Path] = []
    previous_staged: Path | None = None

    for index, payload in enumerate(sorted(payloads)):
        source_directory = tmp_path / "sources" / str(index)
        source_directory.mkdir(parents=True)
        source = source_directory / "book.txt"
        source.write_text(payload, encoding="utf-8")
        staged = manager.stage(source)
        staged_paths.append(staged)

        if previous_staged is not None:
            with pytest.raises(ValueError, match="created by this VaultManager"):
                manager.promote(previous_staged)

        promoted_paths.append(manager.promote(staged))
        previous_staged = staged

    assert previous_staged is not None
    with pytest.raises(ValueError, match="created by this VaultManager"):
        manager.promote(previous_staged)

    originals = list(paths.originals.iterdir())
    assert len({path.name for path in staged_paths}) == 24
    assert len(promoted_paths) == 24
    assert len(originals) == 24
    assert {path.read_text(encoding="utf-8") for path in originals} == payloads
    assert list(paths.inbox.iterdir()) == []
