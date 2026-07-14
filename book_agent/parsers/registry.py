from pathlib import Path
from typing import Optional

from book_agent.models import ParsedBook
from book_agent.parsers.base import DocumentParseError
from book_agent.parsers.text import parse_markdown, parse_txt


SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".md", ".txt"}


def parse_document(
    path: Path, title: Optional[str] = None, author: Optional[str] = None
) -> ParsedBook:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return parse_txt(path, title=title, author=author)
    if suffix == ".md":
        return parse_markdown(path, title=title, author=author)
    if suffix == ".pdf":
        from book_agent.parsers.pdf import parse_pdf

        return parse_pdf(path, title=title, author=author)
    if suffix == ".epub":
        from book_agent.parsers.epub import parse_epub

        return parse_epub(path, title=title, author=author)

    displayed_suffix = suffix or "<none>"
    raise DocumentParseError(
        f"unsupported document type '{displayed_suffix}' for '{path.name}'."
    )
