class DocumentParseError(ValueError):
    """Raised when a document cannot be converted into parsed source units."""


class NeedsOcrError(DocumentParseError):
    """Raised when a document cannot be parsed without OCR."""
