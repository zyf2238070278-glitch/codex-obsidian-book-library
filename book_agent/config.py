import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self


MAX_PREVIEWS = 10
MAX_FULL_PASSAGES = 6
MAX_EVIDENCE_TOKENS = 8000


def _absolute_without_following_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


@dataclass(frozen=True)
class AppPaths:
    root: Path
    vault: Path
    library: Path
    inbox: Path
    originals: Path
    parsed: Path
    notes: Path
    ocr_reports: Path
    catalog_cards: Path
    catalog_base: Path
    database: Path
    models: Path
    ocr_models: Path
    ocr: Path
    ocr_logs: Path
    vision_helper: Path

    @classmethod
    def from_root(cls, root: Path, vault_root: Path | None = None) -> Self:
        resolved_root = root.resolve()
        vault = (
            resolved_root / "vault"
            if vault_root is None
            else _absolute_without_following_symlinks(vault_root)
        )
        library = vault / "书库"

        return cls(
            root=resolved_root,
            vault=vault,
            library=library,
            inbox=library / "00-待导入",
            originals=library / "10-原始书籍",
            parsed=library / "20-解析文本",
            notes=library / "30-AI读书笔记",
            ocr_reports=library / "40-OCR报告",
            catalog_cards=library / "50-书目卡片",
            catalog_base=library / "书库总览.base",
            database=resolved_root / "data" / "library.sqlite3",
            models=resolved_root / "data" / "models",
            ocr_models=resolved_root / "data" / "ocr-models",
            ocr=resolved_root / "data" / "ocr",
            ocr_logs=resolved_root / "data" / "ocr" / "logs",
            vision_helper=resolved_root / "bin" / "book-vision-ocr",
        )
