from pathlib import Path
from typing import Any, Optional

import ebooklib
from bs4 import BeautifulSoup, Tag
from ebooklib import epub

from book_agent.models import ParsedBook, SourceUnit
from book_agent.parsers.base import DocumentParseError


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


def _block_text(block: Tag) -> str:
    parts = []
    for text_node in block.find_all(string=True):
        if text_node.find_parent(["p", "li"]) is not block:
            continue
        normalized = " ".join(str(text_node).split())
        if normalized:
            parts.append(normalized)
    return " ".join(parts)


def _source_unit(item: Any) -> Optional[SourceUnit]:
    if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
        return None
    if isinstance(item, epub.EpubNav):
        return None

    soup = BeautifulSoup(item.get_content(), "html.parser")
    for unwanted in soup.find_all(["script", "style", "nav"]):
        unwanted.decompose()

    section = None
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        heading_text = heading.get_text(" ", strip=True)
        if heading_text:
            section = heading_text
            break

    blocks = []
    for block in soup.find_all(["p", "li"]):
        block_text = _block_text(block)
        if block_text:
            blocks.append(block_text)
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
