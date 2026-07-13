from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from book_agent.models import ParsedBook, Passage


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _location_heading(passage: Passage) -> str:
    parts: list[str] = []
    if passage.section:
        parts.append(passage.section)

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
    display_title = " ".join(parsed.title.splitlines()).strip()
    lines = [
        "---",
        f"book_id: {_json_string(book_id)}",
        f"title: {_json_string(parsed.title)}",
        f"source_format: {_json_string(parsed.source_format)}",
        f"source_file: {_json_string(str(source_file))}",
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
                passage.text,
                "",
                f"^{passage.anchor}",
                "",
            ]
        )
    return "\n".join(lines)


def _remove_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def render_parsed_book(
    destination: str | Path,
    book_id: str,
    parsed: ParsedBook,
    source_file: str | Path,
    passages: Iterable[Passage],
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _render(book_id, parsed, source_file, passages)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temp_path, destination)
    except BaseException:
        _remove_temp(temp_path)
        raise
    return destination
