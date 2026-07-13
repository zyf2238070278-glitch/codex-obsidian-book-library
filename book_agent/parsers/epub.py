from pathlib import Path
from typing import Any, Optional

import ebooklib
from bs4 import BeautifulSoup, Comment, Declaration, Doctype, ProcessingInstruction
from ebooklib import epub

from book_agent.models import ParsedBook, SourceUnit
from book_agent.parsers.base import DocumentParseError


_BODY_BLOCK_TAGS = (
    "p",
    "li",
    "div",
    "blockquote",
    "td",
    "th",
    "pre",
    "dd",
    "dt",
    "figcaption",
    "body",
)
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_NON_RENDERED_TAGS = ("head", "script", "style", "nav", "template", "noscript")
_NON_TEXT_NODES = (Comment, Declaration, Doctype, ProcessingInstruction)


def _metadata_text(book: epub.EpubBook, name: str) -> Optional[str]:
    for entry in book.get_metadata("DC", name):
        value = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _spine_item(book: epub.EpubBook, entry: Any) -> Any:
    if isinstance(entry, (tuple, list)):
        if not entry:
            return None
        entry = entry[0]
    if isinstance(entry, str):
        return book.get_item_with_id(entry)
    return entry


def _body_blocks(soup: BeautifulSoup) -> list[str]:
    blocks = []
    current_block_id = None
    current_parts = []

    def flush() -> None:
        nonlocal current_parts
        if current_parts:
            blocks.append(" ".join(current_parts))
            current_parts = []

    for text_node in soup.find_all(string=True):
        if isinstance(text_node, _NON_TEXT_NODES):
            continue
        if text_node.find_parent(_HEADING_TAGS) is not None:
            continue
        block = text_node.find_parent(_BODY_BLOCK_TAGS)
        if block is None:
            continue
        normalized = " ".join(str(text_node).split())
        if not normalized:
            continue

        block_id = id(block)
        if block_id != current_block_id:
            flush()
            current_block_id = block_id
        current_parts.append(normalized)

    flush()
    return blocks


def _source_unit(item: Any) -> Optional[SourceUnit]:
    if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
        return None
    if isinstance(item, epub.EpubNav):
        return None

    soup = BeautifulSoup(item.get_content(), "html.parser")
    for unwanted in soup.find_all(_NON_RENDERED_TAGS):
        unwanted.decompose()

    section = None
    for heading in soup.find_all(_HEADING_TAGS):
        heading_text = heading.get_text(" ", strip=True)
        if heading_text:
            section = heading_text
            break

    blocks = _body_blocks(soup)
    if not blocks:
        return None

    return SourceUnit(
        text="\n\n".join(blocks),
        section=section,
        page_start=None,
        page_end=None,
        page_label=None,
    )


def parse_epub(
    path: Path, title: Optional[str] = None, author: Optional[str] = None
) -> ParsedBook:
    path = Path(path)
    try:
        book = epub.read_epub(path)
        units = tuple(
            unit
            for entry in book.spine
            if (unit := _source_unit(_spine_item(book, entry))) is not None
        )
        if not units:
            raise DocumentParseError(
                f"Cannot parse '{path.name}': EPUB contains no body text."
            )

        parsed_title = title
        if parsed_title is None:
            parsed_title = _metadata_text(book, "title") or path.stem
        parsed_author = author
        if parsed_author is None:
            parsed_author = _metadata_text(book, "creator")

        return ParsedBook(
            title=parsed_title,
            author=parsed_author,
            source_format="epub",
            units=units,
        )
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"Cannot parse '{path.name}': EPUB may be corrupt, damaged, encrypted, "
            "or DRM-protected."
        ) from exc
