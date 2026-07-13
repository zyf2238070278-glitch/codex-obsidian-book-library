from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Iterable
from pathlib import Path

from book_agent.markdown import markdown_literal
from book_agent.models import ParsedBook, Passage
from book_agent.vault import (
    VaultManager,
    _managed_directory_beneath,
    _secure_create_open_flags,
)


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _location_heading(passage: Passage) -> str:
    parts: list[str] = []
    if passage.section:
        parts.append(markdown_literal(passage.section, single_line=True))

    first_page = passage.page_start
    last_page = passage.page_end
    if first_page is not None or last_page is not None:
        first_page = first_page if first_page is not None else last_page
        last_page = last_page if last_page is not None else first_page
        if first_page == last_page:
            parts.append(f"PDF 页 {first_page}")
        else:
            parts.append(f"PDF 页 {first_page}–{last_page}")

    if not parts:
        parts.append(f"段落 {passage.ordinal + 1}")
    return " · ".join(parts)


def _render(
    book_id: str,
    parsed: ParsedBook,
    source_file: str | Path,
    passages: Iterable[Passage],
) -> str:
    display_title = markdown_literal(parsed.title, single_line=True)
    rendered_title = markdown_literal(parsed.title)
    rendered_author = (
        None if parsed.author is None else markdown_literal(parsed.author)
    )
    lines = [
        "---",
        f"book_id: {_json_string(book_id)}",
        f"title: {_json_string(rendered_title)}",
        (
            "author: null"
            if rendered_author is None
            else f"author: {_json_string(rendered_author)}"
        ),
        f"source_format: {_json_string(parsed.source_format)}",
        f"source_file: {_json_string(markdown_literal(source_file, single_line=True))}",
        "source_type: original",
        "---",
        "",
        f"# {display_title}",
        "",
    ]
    for passage in passages:
        lines.extend(
            [
                f"## {_location_heading(passage)}",
                "",
                markdown_literal(passage.text),
                "",
                f"^{passage.anchor}",
                "",
            ]
        )
    return "\n".join(lines)


def _remove_temp(directory_fd: int, name: str | None) -> None:
    if name is None:
        return
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def _write_complete(file_descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(file_descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Incomplete parsed Markdown write")
        offset += written
    os.fsync(file_descriptor)


def render_parsed_book(
    destination: str | Path,
    book_id: str,
    parsed: ParsedBook,
    source_file: str | Path,
    passages: Iterable[Passage],
    *,
    managed_root: str | Path | None = None,
    expected_root_identity: tuple[int, int] | None = None,
) -> Path:
    destination = Path(os.path.abspath(os.fspath(Path(destination).expanduser())))
    root = Path(destination.anchor) if managed_root is None else Path(managed_root)
    content = _render(book_id, parsed, source_file, passages)

    cleanup_fd: int | None = None
    published_info: os.stat_result | None = None
    try:
        with _managed_directory_beneath(
            root,
            destination.parent,
            "parsed book",
            create=True,
            root_label="managed root",
            create_root=False,
            expected_root_identity=expected_root_identity,
        ) as (_, directory_fd):
            cleanup_fd = os.dup(directory_fd)
            try:
                destination_info = os.stat(
                    destination.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISLNK(destination_info.st_mode):
                    raise ValueError(
                        "Parsed Markdown destination must not be a symlink"
                    )

            temp_name = f".render-{secrets.token_hex(12)}"
            temp_fd = os.open(
                temp_name,
                _secure_create_open_flags(),
                0o600,
                dir_fd=directory_fd,
            )
            try:
                try:
                    _write_complete(temp_fd, content.encode("utf-8"))
                finally:
                    os.close(temp_fd)
                try:
                    os.replace(
                        temp_name,
                        destination.name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                except (TypeError, NotImplementedError) as exc:
                    raise RuntimeError(
                        "This platform cannot atomically publish parsed Markdown "
                        "by directory FD"
                    ) from exc
                temp_name = None
                published_info = os.stat(
                    destination.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            finally:
                _remove_temp(directory_fd, temp_name)
    except BaseException:
        if cleanup_fd is not None and published_info is not None:
            VaultManager._unlink_if_same(
                cleanup_fd,
                destination.name,
                published_info,
            )
        raise
    finally:
        if cleanup_fd is not None:
            try:
                os.close(cleanup_fd)
            except OSError:
                pass
    return destination
