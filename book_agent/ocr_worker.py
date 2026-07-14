"""Minimal detached entry point for the local Apple Vision OCR worker."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from book_agent.config import AppPaths
from book_agent.embeddings import E5EmbeddingProvider
from book_agent.indexing import BookIndexer
from book_agent.ocr.vision import VisionOcrEngine
from book_agent.ocr.worker import OcrWorker
from book_agent.storage import Database


def main() -> int:
    root_text = os.environ.get("BOOK_LIBRARY_ROOT")
    if not root_text:
        raise RuntimeError("BOOK_LIBRARY_ROOT is required")
    root = Path(root_text).expanduser().absolute()
    vault_text = os.environ.get("BOOK_LIBRARY_OBSIDIAN_VAULT")
    paths = AppPaths.from_root(root, Path(vault_text) if vault_text else None)
    paths.ocr.mkdir(parents=True, exist_ok=True)
    database = Database(paths.database, root=paths.root)
    database.initialize()
    provider = E5EmbeddingProvider(paths.models)
    indexer = BookIndexer(paths, database, provider)
    engine = VisionOcrEngine(helper=paths.vision_helper, temp_root=paths.ocr)
    worker = OcrWorker(paths, database, engine, indexer)
    worker.run_until_empty()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        # Keep this entrypoint metadata-only; page text is never printed.
        print(f"OCR worker failed: {exc.__class__.__name__}: {str(exc)[:500]}", file=sys.stderr)
        raise SystemExit(1)

