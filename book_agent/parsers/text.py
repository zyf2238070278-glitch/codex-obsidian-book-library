import re
from pathlib import Path
from typing import Optional

from book_agent.models import ParsedBook, SourceUnit
from book_agent.parsers.base import DocumentParseError


_ATX_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}(?:[ \t]+(.*?))?[ \t]*$")
_FENCE_START = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def _read_utf8(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentParseError(
            f"Cannot parse '{path.name}': expected UTF-8 text."
        ) from exc

    content = content.removeprefix("\ufeff")
    if not content.strip():
        raise DocumentParseError(
            f"Cannot parse '{path.name}': document is empty or whitespace-only."
        )
    return content


def _source_unit(text: str, *, section: Optional[str] = None) -> SourceUnit:
    return SourceUnit(
        text=text,
        section=section,
        page_start=None,
        page_end=None,
        page_label=None,
    )


def parse_txt(
    path: Path, title: Optional[str] = None, author: Optional[str] = None
) -> ParsedBook:
    path = Path(path)
    content = _read_utf8(path)
    paragraphs = []
    current_lines = []

    for line in content.splitlines():
        if line.strip():
            current_lines.append(line)
        elif current_lines:
            paragraphs.append("\n".join(current_lines).strip())
            current_lines = []
    if current_lines:
        paragraphs.append("\n".join(current_lines).strip())

    return ParsedBook(
        title=path.stem if title is None else title,
        author=author,
        source_format="txt",
        units=tuple(_source_unit(paragraph) for paragraph in paragraphs),
    )


def parse_markdown(
    path: Path, title: Optional[str] = None, author: Optional[str] = None
) -> ParsedBook:
    path = Path(path)
    content = _read_utf8(path)
    units = []
    current_lines = []
    current_section = None
    fence_character = None
    fence_length = 0

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            units.append(
                _source_unit("\n".join(current_lines).strip(), section=current_section)
            )
            current_lines = []

    for line in content.splitlines():
        if fence_character is not None:
            current_lines.append(line)
            closing_fence = re.fullmatch(
                rf"[ \t]{{0,3}}{re.escape(fence_character)}{{{fence_length},}}[ \t]*",
                line,
            )
            if closing_fence:
                fence_character = None
                fence_length = 0
            continue

        fence_match = _FENCE_START.match(line)
        if fence_match:
            marker = fence_match.group(1)
            current_lines.append(line)
            fence_character = marker[0]
            fence_length = len(marker)
            continue

        heading_match = _ATX_HEADING.match(line)
        if heading_match:
            flush()
            heading_text = heading_match.group(1) or ""
            current_section = re.sub(
                r"(?:^|[ \t]+)#+[ \t]*$", "", heading_text
            ).strip()
            continue

        if line.strip():
            current_lines.append(line)
        else:
            flush()

    flush()
    if not units:
        raise DocumentParseError(
            f"Cannot parse '{path.name}': Markdown contains no body text."
        )

    return ParsedBook(
        title=path.stem if title is None else title,
        author=author,
        source_format="md",
        units=tuple(units),
    )
