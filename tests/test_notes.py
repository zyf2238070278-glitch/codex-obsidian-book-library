import errno
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from threading import Barrier

import pytest

from book_agent.config import AppPaths
from book_agent.models import Passage
from book_agent.notes import NoteService, SavedNote
from book_agent.storage import Database
from book_agent.vault import VaultManager


FIXED_NOW = datetime(2026, 7, 13, 8, 9, 10)


def _add_book(
    database: Database,
    book_id: str,
    *,
    title: str,
    status: str = "ready",
) -> None:
    database.create_book(
        book_id=book_id,
        title=title,
        author=None,
        source_format="pdf",
        content_sha256=f"hash-{book_id}",
        original_path=f"/books/{book_id}.pdf",
    )
    database.update_book_status(book_id, status)


def _passage(
    passage_id: str,
    *,
    book_id: str = "book-1",
    ordinal: int = 0,
    text: str = "不会自动写进笔记的原文",
    section: str | None = "库存周期",
    page_start: int | None = 12,
    page_end: int | None = 14,
    markdown_path: str | None = None,
    anchor: str | None = None,
) -> Passage:
    return Passage(
        passage_id=passage_id,
        book_id=book_id,
        ordinal=ordinal,
        text=text,
        section=section,
        page_start=page_start,
        page_end=page_end,
        page_label=None,
        markdown_path=markdown_path or f"书库/20-解析文本/{book_id}/正文.md",
        anchor=anchor or passage_id,
        text_sha256=f"sha-{passage_id}",
    )


def _database(paths: AppPaths) -> Database:
    database = Database(paths.database)
    database.initialize()
    return database


def test_saved_note_is_a_frozen_value_object() -> None:
    saved = SavedNote(path="/vault/note.md", wiki_link="[[note]]")

    assert saved.path == "/vault/note.md"
    assert saved.wiki_link == "[[note]]"
    with pytest.raises(FrozenInstanceError):
        saved.path = "/elsewhere.md"


def test_save_writes_ai_markdown_and_citations_without_copying_source_text(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="芯片周期")
    database.replace_passages(
        "book-1",
        [
            _passage("passage-1"),
            _passage(
                "passage-2",
                ordinal=1,
                text="另一段不应复制的原文",
                section=None,
                page_start=None,
                page_end=None,
            ),
        ],
    )

    saved = NoteService(paths, database, clock=lambda: FIXED_NOW).save(
        " 半导体/周期 ",
        " 这是 Codex 生成的分析。\n\n- 需求可能反转。 ",
        ["passage-1", "passage-2"],
    )

    destination = Path(saved.path)
    content = destination.read_text(encoding="utf-8")
    assert destination.is_absolute()
    assert destination.parent == paths.notes
    assert destination.name == "半导体-周期.md"
    assert saved.wiki_link == "[[书库/30-AI读书笔记/半导体-周期]]"
    assert content.startswith(
        "---\n"
        "source_type: ai_generated\n"
        "index_for_evidence: false\n"
        "created_by: codex-book-agent\n"
        "---\n\n"
        "# 半导体/周期\n\n"
        "这是 Codex 生成的分析。\n\n- 需求可能反转。\n\n"
        "## 原文依据\n\n"
    )
    assert (
        "- 《芯片周期》：库存周期，PDF 页 12–14 "
        "[[书库/20-解析文本/book-1/正文.md#^passage-1]]"
    ) in content
    assert (
        "- 《芯片周期》：passage-2 "
        "[[书库/20-解析文本/book-1/正文.md#^passage-2]]"
    ) in content
    assert "不会自动写进笔记的原文" not in content
    assert "另一段不应复制的原文" not in content


def test_save_deduplicates_passage_ids_and_keeps_first_requested_order(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="顺序测试")
    database.replace_passages(
        "book-1",
        [
            _passage("passage-1", ordinal=0, section="第一段"),
            _passage("passage-2", ordinal=1, section="第二段"),
        ],
    )

    class RecordingDatabase:
        requested: list[str] | None = None

        def get_passages(self, passage_ids: list[str]):
            self.requested = list(passage_ids)
            return database.get_passages(passage_ids)

    recording = RecordingDatabase()
    saved = NoteService(paths, recording, clock=lambda: FIXED_NOW).save(
        "引用顺序",
        "分析正文",
        ["passage-2", "passage-1", "passage-2"],
    )

    content = Path(saved.path).read_text(encoding="utf-8")
    assert recording.requested == ["passage-2", "passage-1"]
    assert content.count("#^passage-2]]") == 1
    assert content.count("#^passage-1]]") == 1
    assert content.index("#^passage-2]]") < content.index("#^passage-1]]")


@pytest.mark.parametrize(
    ("title", "markdown", "passage_ids"),
    [
        ("", "正文", ["passage-1"]),
        (" \n\t", "正文", ["passage-1"]),
        (None, "正文", ["passage-1"]),
        ("标题", "", ["passage-1"]),
        ("标题", " \n\t", ["passage-1"]),
        ("标题", None, ["passage-1"]),
        ("标题", "正文", []),
        ("标题", "正文", "passage-1"),
        ("标题", "正文", b"passage-1"),
        ("标题", "正文", {"passage-1"}),
        ("标题", "正文", {"passage-1": True}),
        ("标题", "正文", (value for value in ["passage-1"])),
        ("标题", "正文", [""]),
        ("标题", "正文", [" \n"]),
        ("标题", "正文", [1]),
        ("...---", "正文", ["passage-1"]),
        ("\x00", "正文", ["passage-1"]),
    ],
)
def test_invalid_inputs_raise_value_error_before_database_or_disk_access(
    tmp_path: Path,
    title: object,
    markdown: object,
    passage_ids: object,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")

    class ForbiddenDatabase:
        def get_passages(self, requested: object) -> None:
            raise AssertionError("invalid input reached the database")

    service = NoteService(paths, ForbiddenDatabase(), clock=lambda: FIXED_NOW)

    with pytest.raises(ValueError):
        service.save(title, markdown, passage_ids)

    assert not paths.root.exists()


def test_unknown_and_nonsearchable_passages_are_reported_before_writing(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    for ordinal, status in enumerate(("ready", "processing", "failed")):
        book_id = f"book-{status}"
        passage_id = f"passage-{status}"
        _add_book(database, book_id, title=status, status=status)
        database.replace_passages(
            book_id,
            [_passage(passage_id, book_id=book_id, ordinal=ordinal)],
        )

    service = NoteService(paths, database, clock=lambda: FIXED_NOW)

    with pytest.raises(
        ValueError,
        match="passage-missing.*passage-processing.*passage-failed",
    ):
        service.save(
            "状态门控",
            "不能保存",
            [
                "passage-ready",
                "passage-missing",
                "passage-processing",
                "passage-failed",
            ],
        )

    assert not paths.notes.exists()


def test_existing_note_and_repeated_timestamp_collisions_are_all_preserved(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="碰撞测试")
    database.replace_passages("book-1", [_passage("passage-1")])
    VaultManager(paths).ensure_layout()
    existing = paths.notes / "同名笔记.md"
    existing.write_text("原内容不得覆盖", encoding="utf-8")
    service = NoteService(paths, database, clock=lambda: FIXED_NOW)

    saved = [
        service.save("同名笔记", f"新内容 {index}", ["passage-1"])
        for index in range(1, 4)
    ]

    assert existing.read_text(encoding="utf-8") == "原内容不得覆盖"
    assert [Path(item.path).name for item in saved] == [
        "同名笔记-20260713-080910.md",
        "同名笔记-20260713-080910-2.md",
        "同名笔记-20260713-080910-3.md",
    ]
    for index, item in enumerate(saved, start=1):
        assert f"\n新内容 {index}\n" in Path(item.path).read_text(encoding="utf-8")
    assert not [path for path in paths.notes.iterdir() if path.name.startswith(".")]


def test_ten_concurrent_saves_get_unique_complete_files_without_temps(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="并发测试")
    database.replace_passages("book-1", [_passage("passage-1")])
    service = NoteService(paths, database, clock=lambda: FIXED_NOW)
    start = Barrier(10)

    def save_one(index: int) -> SavedNote:
        start.wait(timeout=10)
        return service.save("并发笔记", f"线程正文 {index}", ["passage-1"])

    with ThreadPoolExecutor(max_workers=10) as executor:
        saved = list(executor.map(save_one, range(10)))

    destinations = [Path(item.path) for item in saved]
    assert len({path.name for path in destinations}) == 10
    assert len(list(paths.notes.iterdir())) == 10
    for index, destination in enumerate(destinations):
        content = destination.read_text(encoding="utf-8")
        assert f"\n线程正文 {index}\n" in content
        assert content.endswith("#^passage-1]]\n")
    assert not [path for path in paths.notes.iterdir() if path.name.startswith(".")]


def test_save_fsyncs_a_0600_complete_temp_before_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="持久化测试")
    database.replace_passages("book-1", [_passage("passage-1")])
    synced: list[tuple[int, int]] = []
    original_fsync = os.fsync

    def record_fsync(file_descriptor: int) -> None:
        original_fsync(file_descriptor)
        info = os.fstat(file_descriptor)
        if stat.S_ISREG(info.st_mode):
            synced.append((info.st_dev, info.st_ino))

    monkeypatch.setattr(os, "fsync", record_fsync)

    saved = NoteService(paths, database, clock=lambda: FIXED_NOW).save(
        "持久化",
        "完整 UTF-8：中文 😀",
        ["passage-1"],
    )

    info = Path(saved.path).stat()
    assert stat.S_IMODE(info.st_mode) == 0o600
    assert (info.st_dev, info.st_ino) in synced
    assert "完整 UTF-8：中文 😀" in Path(saved.path).read_text(encoding="utf-8")


def test_hard_link_failure_cleans_temp_and_publishes_no_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="失败测试")
    database.replace_passages("book-1", [_passage("passage-1")])

    def fail_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(RuntimeError, match="hard.link|atomic|cross-filesystem"):
        NoteService(paths, database, clock=lambda: FIXED_NOW).save(
            "不能发布",
            "正文",
            ["passage-1"],
        )

    assert paths.notes.is_dir()
    assert list(paths.notes.iterdir()) == []


def test_long_chinese_emoji_titles_fit_name_max_for_every_collision(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="长文件名测试")
    database.replace_passages("book-1", [_passage("passage-1")])
    service = NoteService(paths, database, clock=lambda: FIXED_NOW)
    title = "半导体😀周期" * 100

    saved = [
        service.save(title, f"长标题正文 {index}", ["passage-1"])
        for index in range(3)
    ]

    name_max = os.pathconf(paths.notes, "PC_NAME_MAX")
    names = [Path(item.path).name for item in saved]
    assert len(set(names)) == 3
    assert names[0].endswith(".md")
    assert names[1].endswith("-20260713-080910.md")
    assert names[2].endswith("-20260713-080910-2.md")
    for name in names:
        encoded = name.encode("utf-8")
        assert len(encoded) <= name_max
        assert encoded.decode("utf-8") == name


def test_title_and_citation_metadata_newlines_cannot_inject_blocks(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="真实书名\n# 伪标题")
    database.replace_passages(
        "book-1",
        [
            _passage(
                "passage-1",
                section="真实章节\n- 伪列表",
            )
        ],
    )

    saved = NoteService(paths, database, clock=lambda: FIXED_NOW).save(
        "主标题\n## 伪造标题",
        "只有一段正文。",
        ["passage-1"],
    )

    destination = Path(saved.path)
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert destination.name == "主标题--- 伪造标题.md"
    assert [line for line in lines if line.startswith("# ")] == [
        r"# 主标题 \#\# 伪造标题"
    ]
    assert [line for line in lines if line.startswith("- ")] == [
        r"- 《真实书名 \# 伪标题》：真实章节 \- 伪列表，PDF 页 12–14 "
        "[[书库/20-解析文本/book-1/正文.md#^passage-1]]"
    ]
    assert "## 伪造标题" not in lines
    assert "# 伪标题" not in lines
    assert "- 伪列表" not in lines


def test_note_title_and_citation_metadata_are_literal_while_internal_link_works(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    dangerous = (
        "可读 ![远程图](https://evil.example/image.png) "
        "[链接](https://evil.example) ![[嵌入]] [[页面]] "
        "<iframe src='https://evil.example'> \u202e结束"
    )
    _add_book(database, "book-1", title=dangerous)
    database.replace_passages(
        "book-1",
        [_passage("passage-1", text=dangerous, section=dangerous)],
    )
    intentional_body = "正文可保留 [用户要求的链接](https://allowed.example)。"

    saved = NoteService(paths, database, clock=lambda: FIXED_NOW).save(
        dangerous,
        intentional_body,
        ["passage-1"],
    )

    content = Path(saved.path).read_text(encoding="utf-8")
    assert intentional_body in content
    assert "可读" in content
    assert "结束" in content
    assert "\\!\\[远程图\\]\\(" in content
    assert "\\[链接\\]\\(" in content
    assert "\\!\\[\\[嵌入\\]\\]" in content
    assert "\\[\\[页面\\]\\]" in content
    assert "\\<iframe" in content
    assert "https\\:\\/\\/evil\\.example" in content
    assert r"⟦U\+202E⟧" in content
    assert "\u202e" not in content
    assert content.count("[[书库/20-解析文本/book-1/正文.md#^passage-1]]") == 1
    assert "![远程图](" not in content
    assert "[[页面]]" not in content
    assert "\u202e" not in Path(saved.path).name
    assert database.get_passages(["passage-1"])[0].text == dangerous


def test_note_rejects_unverified_internal_citation_target(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="书名")
    database.replace_passages(
        "book-1",
        [
            _passage(
                "passage-1",
                markdown_path="https://evil.example/steal.md",
                anchor="坏锚点]] ![[远程嵌入",
            )
        ],
    )

    with pytest.raises(ValueError, match="citation|引用|link|链接|target|目标"):
        NoteService(paths, database, clock=lambda: FIXED_NOW).save(
            "不能保存",
            "正文",
            ["passage-1"],
        )

    assert not paths.notes.exists()


@pytest.mark.parametrize("symlink_level", ["vault", "notes"])
def test_save_rejects_symlinked_notes_or_ancestor(
    tmp_path: Path,
    symlink_level: str,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="链接测试")
    database.replace_passages("book-1", [_passage("passage-1")])
    outside = tmp_path / f"outside-{symlink_level}"
    outside.mkdir()
    managed = getattr(paths, symlink_level)
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        NoteService(paths, database, clock=lambda: FIXED_NOW).save(
            "不能越界",
            "正文",
            ["passage-1"],
        )

    assert managed.is_symlink()
    assert list(outside.iterdir()) == []


def test_ancestor_swapped_to_symlink_after_layout_check_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="竞态链接测试")
    database.replace_passages("book-1", [_passage("passage-1")])
    outside_library = tmp_path / "outside-library"
    outside_library.mkdir()
    outside_notes = outside_library / paths.notes.name
    outside_notes.mkdir()
    original_ensure_layout = VaultManager.ensure_layout

    def swap_library_after_check(manager: VaultManager) -> None:
        original_ensure_layout(manager)
        paths.library.rename(paths.root / "detached-library")
        paths.library.symlink_to(outside_library, target_is_directory=True)

    monkeypatch.setattr(VaultManager, "ensure_layout", swap_library_after_check)

    with pytest.raises(ValueError, match="symlink"):
        NoteService(paths, database, clock=lambda: FIXED_NOW).save(
            "竞态不能越界",
            "正文",
            ["passage-1"],
        )

    assert list(outside_notes.iterdir()) == []


def test_dangling_candidate_symlink_is_occupied_without_touching_target(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="悬空链接测试")
    database.replace_passages("book-1", [_passage("passage-1")])
    VaultManager(paths).ensure_layout()
    external = tmp_path / "external-note.md"
    occupied = paths.notes / "悬空目标.md"
    occupied.symlink_to(external)

    saved = NoteService(paths, database, clock=lambda: FIXED_NOW).save(
        "悬空目标",
        "内部正文",
        ["passage-1"],
    )

    assert Path(saved.path).name == "悬空目标-20260713-080910.md"
    assert Path(saved.path).read_text(encoding="utf-8").find("内部正文") >= 0
    assert occupied.is_symlink()
    assert not external.exists()
    assert not [path for path in paths.notes.iterdir() if path.name.startswith(".")]


@pytest.mark.parametrize("cleanup_failure", ["unlink", "close"])
def test_cleanup_failure_after_final_link_does_not_turn_success_into_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_failure: str,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="清理测试")
    database.replace_passages("book-1", [_passage("passage-1")])
    linked = False
    original_link = os.link
    original_unlink = os.unlink
    original_close = os.close

    def track_link(*args: object, **kwargs: object) -> None:
        nonlocal linked
        original_link(*args, **kwargs)
        linked = True

    def maybe_fail_unlink(path: object, *args: object, **kwargs: object) -> None:
        if cleanup_failure == "unlink" and str(path).startswith(".note-"):
            raise OSError(errno.EIO, "temp unlink failed after publication")
        original_unlink(path, *args, **kwargs)

    def maybe_fail_close(file_descriptor: int) -> None:
        is_directory = stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
        original_close(file_descriptor)
        if cleanup_failure == "close" and linked and is_directory:
            raise OSError(errno.EIO, "directory close failed after publication")

    monkeypatch.setattr(os, "link", track_link)
    monkeypatch.setattr(os, "unlink", maybe_fail_unlink)
    monkeypatch.setattr(os, "close", maybe_fail_close)

    saved = NoteService(paths, database, clock=lambda: FIXED_NOW).save(
        "清理后成功",
        "完整 final",
        ["passage-1"],
    )

    destination = Path(saved.path)
    assert destination.name == "清理后成功.md"
    assert "完整 final" in destination.read_text(encoding="utf-8")
    assert [path.name for path in paths.notes.iterdir() if not path.name.startswith(".")] == [
        "清理后成功.md"
    ]


def test_cleanup_failures_do_not_mask_a_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    database = _database(paths)
    _add_book(database, "book-1", title="主异常测试")
    database.replace_passages("book-1", [_passage("passage-1")])
    publication_attempted = False
    original_close = os.close

    def fail_link(*args: object, **kwargs: object) -> None:
        nonlocal publication_attempted
        publication_attempted = True
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    def fail_temp_unlink(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EIO, "temp cleanup also failed")

    def fail_directory_close(file_descriptor: int) -> None:
        is_directory = stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
        original_close(file_descriptor)
        if publication_attempted and is_directory:
            raise OSError(errno.EIO, "directory cleanup also failed")

    monkeypatch.setattr(os, "link", fail_link)
    monkeypatch.setattr(os, "unlink", fail_temp_unlink)
    monkeypatch.setattr(os, "close", fail_directory_close)

    with pytest.raises(RuntimeError, match="hard.link|atomic|cross-filesystem"):
        NoteService(paths, database, clock=lambda: FIXED_NOW).save(
            "发布主失败",
            "正文",
            ["passage-1"],
        )

    assert not [path for path in paths.notes.iterdir() if not path.name.startswith(".")]
