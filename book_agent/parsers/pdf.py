import math
from pathlib import Path
from statistics import median
from typing import Optional

import fitz

from book_agent.models import ParsedBook, SourceUnit
from book_agent.parsers.base import DocumentParseError, NeedsOcrError


_OCR_TEXT_LENGTH_THRESHOLD = 20
_OCR_SAMPLE_SIZE = 10


def _representative_page_indices(page_count: int) -> tuple[int, ...]:
    if page_count <= _OCR_SAMPLE_SIZE:
        return tuple(range(page_count))

    last_index = page_count - 1
    return tuple(
        dict.fromkeys(
            round(sample * last_index / (_OCR_SAMPLE_SIZE - 1))
            for sample in range(_OCR_SAMPLE_SIZE)
        )
    )


def _metadata_text(metadata: dict, key: str) -> Optional[str]:
    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _toc_sections(document: fitz.Document, page_count: int) -> tuple[Optional[str], ...]:
    try:
        toc = document.get_toc(simple=True)
    except Exception:
        return (None,) * page_count

    entries = []
    if isinstance(toc, (list, tuple)):
        for order, entry in enumerate(toc):
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue

            title = entry[1]
            if not isinstance(title, str) or not title.strip():
                continue

            raw_page_number = entry[2]
            if isinstance(raw_page_number, bool):
                continue
            try:
                numeric_page_number = float(raw_page_number)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_page_number) or not numeric_page_number.is_integer():
                continue

            page_number = int(numeric_page_number)
            if page_number < 1:
                continue
            entries.append((page_number, order, title.strip()))

    entries.sort(key=lambda item: (item[0], item[1]))
    sections = []
    current_section = None
    entry_index = 0
    for page_number in range(1, page_count + 1):
        while entry_index < len(entries) and entries[entry_index][0] <= page_number:
            current_section = entries[entry_index][2]
            entry_index += 1
        sections.append(current_section)
    return tuple(sections)


def _needs_ocr(path: Path, page_texts: list[str]) -> NeedsOcrError:
    return NeedsOcrError(
        f"Cannot parse '{path.name}': PDF has insufficient extractable text and needs OCR."
    )


def parse_pdf(
    path: Path, title: Optional[str] = None, author: Optional[str] = None
) -> ParsedBook:
    path = Path(path)
    document = None
    try:
        document = fitz.open(path)
        if document.needs_pass and not document.authenticate(""):
            raise DocumentParseError(
                f"Cannot parse '{path.name}': PDF is encrypted and cannot be opened "
                "with an empty password."
            )

        page_count = len(document)
        if page_count == 0:
            raise _needs_ocr(path, [])

        page_texts = [document[index].get_text("text").strip() for index in range(page_count)]
        if not any(page_texts):
            raise _needs_ocr(path, page_texts)

        representative_lengths = [
            len(page_texts[index]) for index in _representative_page_indices(page_count)
        ]
        if median(representative_lengths) < _OCR_TEXT_LENGTH_THRESHOLD:
            raise _needs_ocr(path, page_texts)

        sections = _toc_sections(document, page_count)
        units = []
        for index, page_text in enumerate(page_texts):
            if not page_text:
                continue
            page = document[index]
            page_label = page.get_label()
            normalized_page_label = (
                str(page_label).strip() if page_label is not None else ""
            )
            physical_page_number = index + 1
            units.append(
                SourceUnit(
                    text=page_text,
                    section=sections[index],
                    page_start=physical_page_number,
                    page_end=physical_page_number,
                    page_label=normalized_page_label or None,
                )
            )

        metadata = document.metadata or {}
        parsed_title = title
        if parsed_title is None:
            parsed_title = _metadata_text(metadata, "title") or path.stem
        parsed_author = author
        if parsed_author is None:
            parsed_author = _metadata_text(metadata, "author")

        return ParsedBook(
            title=parsed_title,
            author=parsed_author,
            source_format="pdf",
            units=tuple(units),
        )
    except NeedsOcrError:
        raise
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"Cannot parse '{path.name}': unable to read PDF document."
        ) from exc
    finally:
        if document is not None:
            document.close()
