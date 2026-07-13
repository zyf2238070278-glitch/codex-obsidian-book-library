import json
import os
from pathlib import Path

import pytest

import book_agent.rendering as rendering
from book_agent.markdown import markdown_literal
from book_agent.models import ParsedBook, Passage


def _parsed() -> ParsedBook:
    return ParsedBook(
        title='周期："繁荣"与萧条',
        author="作者甲",
        source_format="pdf",
        units=(),
    )


def _passage(
    ordinal: int,
    text: str,
    *,
    section: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> Passage:
    passage_id = f"passage-{ordinal}"
    return Passage(
        passage_id=passage_id,
        book_id="book-1",
        ordinal=ordinal,
        text=text,
        section=section,
        page_start=page_start,
        page_end=page_end,
        page_label=None,
        markdown_path="书库/20-解析文本/book-1.md",
        anchor=passage_id,
        text_sha256=f"digest-{ordinal}",
    )


def test_render_writes_safe_frontmatter_source_and_unescaped_chinese(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "book.md"
    source_file = tmp_path / '原始："书".pdf'

    result = rendering.render_parsed_book(
        destination,
        "book-1",
        _parsed(),
        source_file,
        [_passage(0, "中文原文。")],
    )

    content = destination.read_text(encoding="utf-8")
    assert result == destination
    assert content.startswith("---\n")
    assert f"book_id: {json.dumps('book-1', ensure_ascii=False)}" in content
    assert f"title: {json.dumps(_parsed().title, ensure_ascii=False)}" in content
    assert 'source_format: "pdf"' in content
    escaped_source = markdown_literal(str(source_file), single_line=True)
    assert f"source_file: {json.dumps(escaped_source, ensure_ascii=False)}" in content
    assert "source_type: original" in content
    assert f"# {_parsed().title}" in content
    assert "中文原文。" in content
    assert "\\u" not in content


def test_render_includes_locations_page_ranges_and_each_anchor_once(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "book.md"
    passages = [
        _passage(0, "第一页原文。", section="第一章", page_start=7, page_end=7),
        _passage(1, "跨页原文。", section="第二章", page_start=8, page_end=10),
        _passage(2, "无位置原文。"),
    ]

    rendering.render_parsed_book(
        destination, "book-1", _parsed(), "incoming/book.pdf", passages
    )

    content = destination.read_text(encoding="utf-8")
    assert "## 第一章 · PDF 页 7" in content
    assert "## 第二章 · PDF 页 8–10" in content
    assert "## 段落 3" in content
    anchor_offsets = []
    for passage in passages:
        marker = f"^{passage.anchor}"
        assert content.count(marker) == 1
        anchor_offsets.append(content.index(marker))
    assert anchor_offsets == sorted(anchor_offsets)
    assert content.index("第一页原文。") < content.index("跨页原文。")
    assert content.index("跨页原文。") < content.index("无位置原文。")


def test_render_collapses_title_line_breaks_in_the_h1_only(tmp_path: Path) -> None:
    destination = tmp_path / "book.md"
    parsed = ParsedBook(
        title="标题\n## 伪章节",
        author=None,
        source_format="txt",
        units=(),
    )

    rendering.render_parsed_book(
        destination,
        "book-1",
        parsed,
        "source.txt",
        [_passage(0, "正文。")],
    )

    content = destination.read_text(encoding="utf-8")
    escaped_title = markdown_literal(parsed.title)
    assert f"title: {json.dumps(escaped_title, ensure_ascii=False)}" in content
    assert r"# 标题 \#\# 伪章节" in content
    assert "\n## 伪章节\n" not in content
    assert [line for line in content.splitlines() if line.startswith("# ")] == [
        r"# 标题 \#\# 伪章节"
    ]


def test_render_neutralizes_untrusted_markdown_and_bidi_but_keeps_text_readable(
    tmp_path: Path,
) -> None:
    dangerous = (
        "可读内容 ![远程图](https://evil.example/image.png) "
        "[远程链接](https://evil.example) ![[恶意嵌入]] [[恶意页面]] "
        "<img src='https://evil.example/tracker'> \u202e结束"
    )
    parsed = ParsedBook(
        title=dangerous,
        author=dangerous,
        source_format="md",
        units=(),
    )
    destination = tmp_path / "book.md"

    rendering.render_parsed_book(
        destination,
        "book-1",
        parsed,
        "source.md",
        [_passage(0, dangerous, section=dangerous)],
    )

    content = destination.read_text(encoding="utf-8")
    assert "可读内容" in content
    assert "结束" in content
    assert "author:" in content
    assert "\\!\\[远程图\\]\\(" in content
    assert "\\[远程链接\\]\\(" in content
    assert "\\!\\[\\[恶意嵌入\\]\\]" in content
    assert "\\[\\[恶意页面\\]\\]" in content
    assert "\\<img" in content
    assert "https\\:\\/\\/evil\\.example" in content
    assert r"⟦U\+202E⟧" in content
    assert "\u202e" not in content
    assert "![远程图](" not in content
    assert "[[恶意页面]]" not in content

    frontmatter_lines = content.split("---\n", 2)[1].splitlines()
    encoded_values = {
        key: value.strip()
        for key, _, value in (line.partition(":") for line in frontmatter_lines)
        if key in {"book_id", "title", "author", "source_format", "source_file"}
    }
    for encoded in encoded_values.values():
        assert isinstance(json.loads(encoded), str)


def test_render_atomically_replaces_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "book.md"
    destination.write_text("旧文件内容", encoding="utf-8")

    rendering.render_parsed_book(
        destination,
        "book-1",
        _parsed(),
        "source.pdf",
        [_passage(0, "完整的新文件。")],
    )

    content = destination.read_text(encoding="utf-8")
    assert content != "旧文件内容"
    assert "完整的新文件。" in content
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_render_does_not_recreate_a_missing_managed_root(tmp_path: Path) -> None:
    managed_root = tmp_path / "managed-root"
    managed_root.mkdir()
    destination = managed_root / "nested" / "book.md"
    managed_root.rmdir()

    with pytest.raises(ValueError, match="managed root"):
        rendering.render_parsed_book(
            destination,
            "book-1",
            _parsed(),
            "source.pdf",
            [_passage(0, "不得写入。")],
            managed_root=managed_root,
        )

    assert not managed_root.exists()


def test_render_rejects_replaced_managed_root_identity(tmp_path: Path) -> None:
    managed_root = tmp_path / "managed-root"
    managed_root.mkdir()
    root_info = os.lstat(managed_root)
    expected_identity = (root_info.st_dev, root_info.st_ino)
    displaced_root = tmp_path / "displaced-root"
    managed_root.rename(displaced_root)
    managed_root.mkdir()
    destination = managed_root / "nested" / "book.md"

    with pytest.raises(ValueError, match=r"managed root.*identity"):
        rendering.render_parsed_book(
            destination,
            "book-1",
            _parsed(),
            "source.pdf",
            [_passage(0, "不得写入。")],
            managed_root=managed_root,
            expected_root_identity=expected_identity,
        )

    assert list(managed_root.iterdir()) == []
    assert list(displaced_root.iterdir()) == []


def test_render_root_swap_after_replace_rolls_back_without_returning_stale_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed_root = tmp_path / "managed-root"
    managed_root.mkdir()
    root_info = os.lstat(managed_root)
    expected_identity = (root_info.st_dev, root_info.st_ino)
    displaced_root = tmp_path / "displaced-root"
    destination = managed_root / "nested" / "book.md"
    sentinel_payload = b"replacement render sentinel"
    real_replace = rendering.os.replace

    def replace_then_replace_root(*args: object, **kwargs: object) -> None:
        real_replace(*args, **kwargs)
        managed_root.rename(displaced_root)
        replacement_directory = managed_root / "nested"
        replacement_directory.mkdir(parents=True)
        (replacement_directory / "sentinel.md").write_bytes(sentinel_payload)

    monkeypatch.setattr(rendering.os, "replace", replace_then_replace_root)

    with pytest.raises(ValueError, match=r"managed root.*identity"):
        rendering.render_parsed_book(
            destination,
            "book-1",
            _parsed(),
            "source.pdf",
            [_passage(0, "不得返回失真的路径。")],
            managed_root=managed_root,
            expected_root_identity=expected_identity,
        )

    replacement_directory = managed_root / "nested"
    assert (replacement_directory / "sentinel.md").read_bytes() == sentinel_payload
    assert list(replacement_directory.iterdir()) == [
        replacement_directory / "sentinel.md"
    ]
    assert list((displaced_root / "nested").iterdir()) == []


def test_render_root_symlink_swap_after_replace_never_writes_to_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed_root = tmp_path / "managed-root"
    managed_root.mkdir()
    root_info = os.lstat(managed_root)
    expected_identity = (root_info.st_dev, root_info.st_ino)
    displaced_root = tmp_path / "displaced-root"
    symlink_target = tmp_path / "symlink-target"
    target_directory = symlink_target / "nested"
    target_directory.mkdir(parents=True)
    sentinel = target_directory / "sentinel.md"
    sentinel.write_bytes(b"symlink target sentinel")
    destination = managed_root / "nested" / "book.md"
    real_replace = rendering.os.replace

    def replace_then_symlink_root(*args: object, **kwargs: object) -> None:
        real_replace(*args, **kwargs)
        managed_root.rename(displaced_root)
        managed_root.symlink_to(symlink_target, target_is_directory=True)

    monkeypatch.setattr(rendering.os, "replace", replace_then_symlink_root)

    with pytest.raises(ValueError, match=r"managed root.*identity"):
        rendering.render_parsed_book(
            destination,
            "book-1",
            _parsed(),
            "source.pdf",
            [_passage(0, "不得写入符号链接目标。")],
            managed_root=managed_root,
            expected_root_identity=expected_identity,
        )

    assert sentinel.read_bytes() == b"symlink target sentinel"
    assert list(target_directory.iterdir()) == [sentinel]
    assert list((displaced_root / "nested").iterdir()) == []


@pytest.mark.parametrize("replacement_kind", ["directory", "symlink"])
def test_render_root_swap_restores_a_preexisting_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    managed_root = tmp_path / "managed-root"
    original_directory = managed_root / "nested"
    original_directory.mkdir(parents=True)
    destination = original_directory / "book.md"
    destination.write_text("必须恢复的旧解析内容", encoding="utf-8")
    root_info = os.lstat(managed_root)
    expected_identity = (root_info.st_dev, root_info.st_ino)
    displaced_root = tmp_path / "displaced-root"
    symlink_target = tmp_path / "symlink-target"
    replacement_root = managed_root if replacement_kind == "directory" else symlink_target
    replacement_directory = replacement_root / "nested"
    sentinel = replacement_directory / "sentinel.md"
    real_replace = rendering.os.replace
    swapped = False

    def replace_then_swap_root(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        real_replace(*args, **kwargs)
        if swapped:
            return
        swapped = True
        managed_root.rename(displaced_root)
        replacement_directory.mkdir(parents=True)
        sentinel.write_bytes(b"replacement sentinel")
        if replacement_kind == "symlink":
            managed_root.symlink_to(symlink_target, target_is_directory=True)

    monkeypatch.setattr(rendering.os, "replace", replace_then_swap_root)

    with pytest.raises(ValueError, match=r"managed root.*identity"):
        rendering.render_parsed_book(
            destination,
            "book-1",
            _parsed(),
            "source.pdf",
            [_passage(0, "不应覆盖旧解析内容。")],
            managed_root=managed_root,
            expected_root_identity=expected_identity,
        )

    restored = displaced_root / "nested" / "book.md"
    assert restored.read_text(encoding="utf-8") == "必须恢复的旧解析内容"
    assert sentinel.read_bytes() == b"replacement sentinel"
    assert list(replacement_directory.iterdir()) == [sentinel]
    assert not list((displaced_root / "nested").glob(".render-backup-*"))


def test_replace_failure_preserves_old_file_and_cleans_unique_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "book.md"
    destination.write_text("不可破坏的旧文件", encoding="utf-8")

    def fail_replace(
        source: str | Path,
        target: str | Path,
        **kwargs: object,
    ) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(rendering.os, "replace", fail_replace)

    with pytest.raises(OSError, match="publish failed"):
        rendering.render_parsed_book(
            destination,
            "book-1",
            _parsed(),
            "source.pdf",
            [_passage(0, "不能发布的新文件。")],
        )

    assert destination.read_text(encoding="utf-8") == "不可破坏的旧文件"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
