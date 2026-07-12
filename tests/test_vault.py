from pathlib import Path

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


def test_stage_and_promote_preserve_existing_file(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    manager = VaultManager(paths)
    manager.ensure_layout()
    source = tmp_path / "book.txt"
    source.write_text("first")
    existing = paths.originals / "book.txt"
    existing.write_text("existing")

    staged = manager.stage(source)
    promoted = manager.promote(staged)

    assert promoted.name == "book-2.txt"
    assert promoted.read_text() == "first"
    assert existing.read_text() == "existing"
