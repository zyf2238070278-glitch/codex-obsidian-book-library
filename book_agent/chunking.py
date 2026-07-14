from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from book_agent.models import ParsedBook, Passage


_PARAGRAPH_BREAK = re.compile(r"\r?\n[\t ]*\r?\n+")
_SENTENCE_ENDINGS = frozenset(".!?。！？；;")
_CLOSING_PUNCTUATION = frozenset('"\'”’）)]】》」』')


@dataclass(frozen=True)
class _Fragment:
    text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    page_label: str | None


def _preferred_break(text: str, max_chars: int) -> int:
    minimum = max(1, max_chars // 2)
    best = 0
    window = text[:max_chars]
    for index, character in enumerate(window):
        if character != "\n" and character not in _SENTENCE_ENDINGS:
            continue
        candidate = index + 1
        if character in _SENTENCE_ENDINGS:
            while (
                candidate < max_chars
                and candidate < len(text)
                and text[candidate] in _CLOSING_PUNCTUATION
            ):
                candidate += 1
        if candidate >= minimum:
            best = max(best, candidate)
    return best or max_chars


def _split_oversized_paragraph(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        boundary = _preferred_break(remaining, max_chars)
        pieces.append(remaining[:boundary])
        remaining = remaining[boundary:]
    if remaining:
        pieces.append(remaining)
    return pieces


def _split_paragraph_with_boundary(
    text: str, has_boundary: bool, max_chars: int
) -> list[str]:
    if not has_boundary or max_chars == 1:
        return _split_oversized_paragraph(text, max_chars)

    boundary = "\n\n" if max_chars >= 3 else "\n"
    body_capacity = max_chars - len(boundary)
    if len(text) > body_capacity:
        body_end = _preferred_break(text, body_capacity)
    else:
        body_end = len(text)

    first = boundary + text[:body_end]
    return [first, *_split_oversized_paragraph(text[body_end:], max_chars)]


def _fragments(parsed: ParsedBook, max_chars: int) -> list[_Fragment]:
    fragments: list[_Fragment] = []
    saw_paragraph = False
    for unit in parsed.units:
        paragraphs = (
            paragraph.strip() for paragraph in _PARAGRAPH_BREAK.split(unit.text)
        )
        for paragraph in paragraphs:
            if not paragraph:
                continue
            pieces = _split_paragraph_with_boundary(
                paragraph, saw_paragraph, max_chars
            )
            for piece in pieces:
                fragments.append(
                    _Fragment(
                        text=piece,
                        section=unit.section,
                        page_start=unit.page_start,
                        page_end=unit.page_end,
                        page_label=unit.page_label,
                    )
                )
            saw_paragraph = True
    return fragments


def _combined_metadata(
    fragments: list[_Fragment],
) -> tuple[str | None, int | None, int | None, str | None]:
    page_starts = [
        fragment.page_start
        for fragment in fragments
        if fragment.page_start is not None
    ]
    page_ends = [
        fragment.page_end
        for fragment in fragments
        if fragment.page_end is not None
    ]
    page_label = next(
        (fragment.page_label for fragment in fragments if fragment.page_label),
        None,
    )
    return (
        fragments[0].section,
        min(page_starts) if page_starts else None,
        max(page_ends) if page_ends else None,
        page_label,
    )


def _should_add(current_length: int, combined_length: int, target_chars: int) -> bool:
    if combined_length <= target_chars:
        return True
    return abs(combined_length - target_chars) <= abs(current_length - target_chars)


def chunk_book(
    book_id: str,
    parsed: ParsedBook,
    markdown_path: str | Path,
    target_chars: int = 1500,
    max_chars: int = 2500,
) -> list[Passage]:
    if target_chars <= 0:
        raise ValueError("target_chars must be greater than zero")
    if max_chars < target_chars:
        raise ValueError("max_chars must be greater than or equal to target_chars")

    fragments = _fragments(parsed, max_chars)
    if not fragments:
        return []

    groups: list[list[_Fragment]] = []
    current: list[_Fragment] = []
    current_text = ""

    for fragment in fragments:
        if not current:
            current = [fragment]
            current_text = fragment.text
            continue

        if fragment.section != current[0].section:
            groups.append(current)
            current = [fragment]
            current_text = fragment.text
            continue

        addition = fragment.text
        combined_length = len(current_text) + len(addition)
        if combined_length > max_chars or not _should_add(
            len(current_text), combined_length, target_chars
        ):
            groups.append(current)
            current = [fragment]
            current_text = fragment.text
            continue

        current.append(fragment)
        current_text += addition

    if current:
        groups.append(current)

    passages: list[Passage] = []
    rendered_path = str(markdown_path)
    for ordinal, group in enumerate(groups):
        text = "".join(fragment.text for fragment in group)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        passage_id = hashlib.sha256(
            f"{book_id}:{ordinal}:{digest}".encode("utf-8")
        ).hexdigest()[:24]
        section, page_start, page_end, page_label = _combined_metadata(group)
        passages.append(
            Passage(
                passage_id=passage_id,
                book_id=book_id,
                ordinal=ordinal,
                text=text,
                section=section,
                page_start=page_start,
                page_end=page_end,
                page_label=page_label,
                markdown_path=rendered_path,
                anchor=passage_id,
                text_sha256=digest,
            )
        )
    return passages
