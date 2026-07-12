from dataclasses import dataclass
from typing import Literal, Optional, Tuple


BookStatus = Literal[
    "processing",
    "ready",
    "keyword_only",
    "needs_ocr",
    "duplicate",
    "failed",
]
RetrievalMode = Literal["auto", "quote", "explain", "compare"]


@dataclass(frozen=True)
class SourceUnit:
    text: str
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    page_label: Optional[str] = None


@dataclass(frozen=True)
class ParsedBook:
    title: str
    author: Optional[str]
    source_format: str
    units: Tuple[SourceUnit, ...]


@dataclass(frozen=True)
class Passage:
    passage_id: str
    book_id: str
    ordinal: int
    text: str
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    page_label: Optional[str]
    markdown_path: str
    anchor: str
    text_sha256: str
    embedding: Optional[bytes] = None


@dataclass(frozen=True)
class SearchHit:
    passage_id: str
    book_id: str
    title: str
    text: str
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    page_label: Optional[str]
    markdown_path: str
    anchor: str
    score: float
