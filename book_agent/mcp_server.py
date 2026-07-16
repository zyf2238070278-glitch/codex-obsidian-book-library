from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from book_agent.config import MAX_PREVIEWS
from book_agent.ocr.service import DEFAULT_PENDING_LIMIT, DEFAULT_STATUS_LIMIT
from book_agent.tools import build_tools


TOOL_NAMES = (
    "import_book",
    "list_books",
    "library_status",
    "search_books",
    "get_passages",
    "save_reading_note",
    "start_ocr",
    "start_pending_ocr",
    "ocr_status",
    "pause_ocr",
)

ROOT = Path(os.environ.get("BOOK_LIBRARY_ROOT", os.getcwd())).expanduser().resolve()
_OBSIDIAN_VAULT_ENV = os.environ.get("BOOK_LIBRARY_OBSIDIAN_VAULT")
OBSIDIAN_VAULT = (
    None
    if _OBSIDIAN_VAULT_ENV is None or not _OBSIDIAN_VAULT_ENV.strip()
    else Path(_OBSIDIAN_VAULT_ENV).expanduser().absolute()
)
library_tools = build_tools(ROOT, vault_root=OBSIDIAN_VAULT)
mcp = FastMCP("local-book-library")

CodexAttachmentPath = Annotated[
    str,
    Field(
        description=(
            "The absolute local filesystem path of the book attachment uploaded in "
            "Codex."
        )
    ),
]


@mcp.tool()
def import_book(
    file_path: CodexAttachmentPath,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> dict[str, Any]:
    """Import a Codex book attachment using its absolute local filesystem path."""

    return library_tools.import_book(file_path, title=title, author=author)


@mcp.tool()
def list_books(status: Optional[str] = None) -> dict[str, Any]:
    """List library book metadata, optionally filtered by import status."""

    return library_tools.list_books(status=status)


@mcp.tool()
def library_status(book_id: Optional[str] = None) -> dict[str, Any]:
    """Inspect local index health and actionable issues without returning book text."""

    return library_tools.library_status(book_id=book_id)


@mcp.tool()
def search_books(
    query: str,
    mode: str = "auto",
    book_ids: Optional[list[str]] = None,
    limit: int = MAX_PREVIEWS,
) -> dict[str, Any]:
    """Search books and return at most ten explicitly untrusted text previews."""

    return library_tools.search_books(
        query=query,
        mode=mode,
        book_ids=book_ids,
        limit=limit,
    )


@mcp.tool()
def get_passages(
    passage_ids: list[str],
    neighbor_count: int = 1,
) -> dict[str, Any]:
    """Retrieve full evidence for known passage IDs with bounded neighboring context."""

    return library_tools.get_passages(
        passage_ids=passage_ids,
        neighbor_count=neighbor_count,
    )


@mcp.tool()
def save_reading_note(
    title: str,
    markdown: str,
    passage_ids: list[str],
) -> dict[str, Any]:
    """Save an AI-authored reading note citing known passages in the Obsidian vault."""

    return library_tools.save_reading_note(
        title=title,
        markdown=markdown,
        passage_ids=passage_ids,
    )


@mcp.tool()
def start_ocr(book_id: str) -> dict[str, Any]:
    """Explicitly queue or resume OCR for one managed PDF book."""

    return library_tools.start_ocr(book_id)


@mcp.tool()
def start_pending_ocr(
    limit: int = DEFAULT_PENDING_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """Queue a bounded page of pending OCR books."""

    return library_tools.start_pending_ocr(limit=limit, offset=offset)


@mcp.tool()
def ocr_status(
    book_id: Optional[str] = None,
    limit: int = DEFAULT_STATUS_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """Inspect bounded OCR progress metadata without returning recognized text."""

    return library_tools.ocr_status(book_id=book_id, limit=limit, offset=offset)


@mcp.tool()
def pause_ocr(book_id: str) -> dict[str, Any]:
    """Pause one OCR job at a safe page boundary."""

    return library_tools.pause_ocr(book_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
