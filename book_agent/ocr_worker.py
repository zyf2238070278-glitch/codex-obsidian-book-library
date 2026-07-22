"""Minimal detached entry point for the resilient local OCR worker."""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from book_agent.config import AppPaths
from book_agent.embeddings import E5EmbeddingProvider
from book_agent.indexing import BookIndexer
from book_agent.ocr.light import LightOcrEngine
from book_agent.ocr.rapid import RapidOcrEngine
from book_agent.ocr.router import LocalOcrRouter
from book_agent.ocr.vision import VisionOcrEngine
from book_agent.ocr.worker import OcrWorker
from book_agent.storage import Database


def _regular_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)


def build_light_ocr_engine(
    paths: AppPaths,
    *,
    environ: Mapping[str, str] | None = None,
    find_executable: Callable[[str], str | None] = shutil.which,
    run_command: Callable[..., Any] = subprocess.run,
) -> LightOcrEngine | None:
    """Return the optional macOS Light OCR fallback when its runtime is valid."""

    if not _regular_file(paths.light_ocr_worker) or not _regular_file(
        paths.light_ocr_package
    ):
        return None
    environment = os.environ if environ is None else environ
    configured = environment.get("BOOK_LIBRARY_LIGHT_OCR_NODE")
    bundled = paths.root / "runtime" / "node" / "bin" / "node"
    candidate = configured or (
        str(bundled) if bundled.is_file() else find_executable("node")
    )
    if not candidate:
        return None
    node = Path(candidate).expanduser().resolve()
    if not _regular_file(node) or not os.access(node, os.X_OK):
        return None
    try:
        version = run_command(
            [str(node), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.fullmatch(
            r"v(\d+)\.\d+\.\d+",
            (getattr(version, "stdout", "") or "").strip(),
        )
        if match is None or int(match.group(1)) not in {22, 24}:
            return None
        architecture = run_command(
            [str(node), "-p", "process.arch"],
            check=True,
            capture_output=True,
            text=True,
        )
        if (getattr(architecture, "stdout", "") or "").strip() != "arm64":
            return None
    except (OSError, subprocess.CalledProcessError):
        return None
    return LightOcrEngine(node=node, worker=paths.light_ocr_worker)


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
    light = build_light_ocr_engine(paths)
    try:
        router = LocalOcrRouter(
            vision=VisionOcrEngine(helper=paths.vision_helper, temp_root=paths.ocr),
            rapid=RapidOcrEngine(paths.ocr_models / "rapidocr"),
            light=light,
        )
        worker = OcrWorker(paths, database, router, indexer)
        worker.run_until_empty()
    finally:
        if light is not None:
            light.close()
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
