from pathlib import Path

import fitz
import pytest

from book_agent.parsers.base import DocumentParseError, NeedsOcrError
from book_agent.parsers.pdf import parse_pdf


def _body(label: str) -> str:
    return (
        f"{label}: This synthetic PDF page contains enough extractable English "
        "text to represent ordinary book content without requiring OCR."
    )


def _write_pdf(
    path: Path,
    pages: list[str],
    *,
    metadata: dict[str, str] | None = None,
    toc: list[list[object]] | None = None,
    page_labels: list[dict[str, object]] | None = None,
) -> None:
    document = fitz.open()
    try:
        for text in pages:
            page = document.new_page()
            if text:
                page.insert_textbox(
                    fitz.Rect(72, 72, 540, 760),
                    text,
                    fontsize=11,
                )
        if metadata is not None:
            document.set_metadata(metadata)
        if toc is not None:
            document.set_toc(toc)
        if page_labels is not None:
            document.set_page_labels(page_labels)
        document.save(path)
    finally:
        document.close()


def test_parse_pdf_creates_one_page_aware_unit_per_text_page(tmp_path: Path) -> None:
    path = tmp_path / "two-pages.pdf"
    _write_pdf(path, [_body("First"), _body("Second")])

    book = parse_pdf(path, title="Explicit title", author="Explicit author")

    assert book.title == "Explicit title"
    assert book.author == "Explicit author"
    assert book.source_format == "pdf"
    assert [unit.page_start for unit in book.units] == [1, 2]
    assert [unit.page_end for unit in book.units] == [1, 2]
    assert [unit.page_label for unit in book.units] == [None, None]
    assert [unit.section for unit in book.units] == [None, None]
    assert [unit.text.split(":", 1)[0] for unit in book.units] == ["First", "Second"]


def test_parse_pdf_skips_blank_pages_without_compressing_physical_numbers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blank-middle.pdf"
    _write_pdf(path, [_body("First"), "", _body("Third")])

    book = parse_pdf(path)

    assert [unit.page_start for unit in book.units] == [1, 3]
    assert [unit.page_end for unit in book.units] == [1, 3]


def test_parse_pdf_rejects_an_all_blank_document_as_needing_ocr(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blank.pdf"
    _write_pdf(path, ["", "", ""])

    with pytest.raises(NeedsOcrError, match="(?i)OCR"):
        parse_pdf(path)


def test_blank_front_matter_does_not_trigger_ocr_when_later_pages_have_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "front-matter.pdf"
    _write_pdf(path, ["", ""] + [_body(f"Page {number}") for number in range(3, 13)])

    book = parse_pdf(path)

    assert book.units[0].page_start == 3
    assert [unit.page_start for unit in book.units] == list(range(3, 13))


def test_mostly_blank_document_with_only_a_little_text_needs_ocr(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mostly-scanned.pdf"
    pages = [""] * 15
    pages[7] = "Index"
    _write_pdf(path, pages)

    with pytest.raises(NeedsOcrError, match="(?i)OCR"):
        parse_pdf(path)


def test_parse_pdf_uses_metadata_only_when_arguments_are_none(tmp_path: Path) -> None:
    path = tmp_path / "metadata.pdf"
    _write_pdf(
        path,
        [_body("Metadata")],
        metadata={"title": "Metadata title", "author": "Metadata author"},
    )

    from_metadata = parse_pdf(path)
    explicit_empty = parse_pdf(path, title="", author="")

    assert (from_metadata.title, from_metadata.author) == (
        "Metadata title",
        "Metadata author",
    )
    assert (explicit_empty.title, explicit_empty.author) == ("", "")


def test_parse_pdf_falls_back_to_stem_and_none_for_blank_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fallback-name.pdf"
    _write_pdf(
        path,
        [_body("Fallback")],
        metadata={"title": "   ", "author": ""},
    )

    book = parse_pdf(path)

    assert book.title == "fallback-name"
    assert book.author is None


def test_parse_pdf_assigns_the_nearest_preceding_toc_title(tmp_path: Path) -> None:
    path = tmp_path / "toc.pdf"
    _write_pdf(
        path,
        [_body("One"), _body("Two"), _body("Three")],
        toc=[[1, "Part One", 1], [2, "Chapter Two", 2]],
    )

    book = parse_pdf(path)

    assert [unit.section for unit in book.units] == [
        "Part One",
        "Chapter Two",
        "Chapter Two",
    ]


def test_parse_pdf_preserves_nonempty_page_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.pdf"
    _write_pdf(
        path,
        [_body("One"), _body("Two")],
        page_labels=[
            {"startpage": 0, "prefix": "A-", "style": "D", "firstpagenum": 1}
        ],
    )

    book = parse_pdf(path)

    assert [unit.page_label for unit in book.units] == ["A-1", "A-2"]


def test_parse_pdf_rejects_password_protected_documents(tmp_path: Path) -> None:
    path = tmp_path / "secret.pdf"
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), _body("Secret"))
        document.save(
            path,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner-secret",
            user_pw="reader-secret",
        )
    finally:
        document.close()

    with pytest.raises(DocumentParseError, match=r"(?i)secret\.pdf.*encrypt"):
        parse_pdf(path)


@pytest.mark.parametrize("fixture", ["corrupt", "missing", "directory"])
def test_parse_pdf_wraps_open_errors_with_the_filename_and_cause(
    tmp_path: Path, fixture: str
) -> None:
    path = tmp_path / f"{fixture}.pdf"
    if fixture == "corrupt":
        path.write_bytes(b"not a PDF document")
    elif fixture == "directory":
        path.mkdir()

    with pytest.raises(DocumentParseError, match=path.name) as error:
        parse_pdf(path)

    assert error.value.__cause__ is not None
