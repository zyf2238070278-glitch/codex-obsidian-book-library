from pathlib import Path

import pytest

from book_agent.config import AppPaths


def test_app_paths_are_rooted_under_project(tmp_path: Path) -> None:
    project_root = tmp_path / "project"

    paths = AppPaths.from_root(project_root)

    resolved_root = project_root.resolve()
    assert paths == AppPaths(
        root=resolved_root,
        vault=resolved_root / "vault",
        library=resolved_root / "vault" / "书库",
        inbox=resolved_root / "vault" / "书库" / "00-待导入",
        originals=resolved_root / "vault" / "书库" / "10-原始书籍",
        parsed=resolved_root / "vault" / "书库" / "20-解析文本",
        notes=resolved_root / "vault" / "书库" / "30-AI读书笔记",
        ocr_reports=resolved_root / "vault" / "书库" / "40-OCR报告",
        catalog_cards=resolved_root / "vault" / "书库" / "50-书目卡片",
        catalog_base=resolved_root / "vault" / "书库" / "书库总览.base",
        database=resolved_root / "data" / "library.sqlite3",
        models=resolved_root / "data" / "models",
        ocr_models=resolved_root / "data" / "ocr-models",
        ocr=resolved_root / "data" / "ocr",
        ocr_logs=resolved_root / "data" / "ocr" / "logs",
        vision_helper=resolved_root / "bin" / "book-vision-ocr",
        light_ocr_worker=resolved_root / "scripts" / "light_ocr_worker.mjs",
        light_ocr_package=resolved_root / "node_modules" / "@arcships" / "light-ocr" / "package.json",
    )


def test_app_paths_preserve_literal_tilde_in_relative_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    paths = AppPaths.from_root(Path("~/project"))

    assert paths.root == (tmp_path / "~" / "project").resolve()


def test_app_paths_keep_external_obsidian_files_separate_from_project_data(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    obsidian_vault = tmp_path / "current-obsidian"

    paths = AppPaths.from_root(project_root, vault_root=obsidian_vault)

    resolved_root = project_root.resolve()
    absolute_vault = obsidian_vault.absolute()
    assert paths == AppPaths(
        root=resolved_root,
        vault=absolute_vault,
        library=absolute_vault / "书库",
        inbox=absolute_vault / "书库" / "00-待导入",
        originals=absolute_vault / "书库" / "10-原始书籍",
        parsed=absolute_vault / "书库" / "20-解析文本",
        notes=absolute_vault / "书库" / "30-AI读书笔记",
        ocr_reports=absolute_vault / "书库" / "40-OCR报告",
        catalog_cards=absolute_vault / "书库" / "50-书目卡片",
        catalog_base=absolute_vault / "书库" / "书库总览.base",
        database=resolved_root / "data" / "library.sqlite3",
        models=resolved_root / "data" / "models",
        ocr_models=resolved_root / "data" / "ocr-models",
        ocr=resolved_root / "data" / "ocr",
        ocr_logs=resolved_root / "data" / "ocr" / "logs",
        vision_helper=resolved_root / "bin" / "book-vision-ocr",
        light_ocr_worker=resolved_root / "scripts" / "light_ocr_worker.mjs",
        light_ocr_package=resolved_root / "node_modules" / "@arcships" / "light-ocr" / "package.json",
    )


def test_app_paths_do_not_follow_an_external_vault_symlink(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    target = tmp_path / "target"
    alias = tmp_path / "alias"
    target.mkdir()
    alias.symlink_to(target, target_is_directory=True)

    paths = AppPaths.from_root(project_root, vault_root=alias)

    assert paths.vault == alias.absolute()
    assert paths.vault != target.resolve()


def test_app_paths_places_ocr_runtime_under_project_data(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)

    assert paths.ocr == tmp_path.resolve() / "data" / "ocr"
    assert paths.ocr_logs == paths.ocr / "logs"
    assert paths.ocr_models == tmp_path.resolve() / "data" / "ocr-models"
    assert paths.vision_helper == tmp_path.resolve() / "bin" / "book-vision-ocr"
    assert paths.light_ocr_worker == tmp_path.resolve() / "scripts" / "light_ocr_worker.mjs"
    assert paths.light_ocr_package == (
        tmp_path.resolve() / "node_modules" / "@arcships" / "light-ocr" / "package.json"
    )


def test_paths_expose_non_evidence_ocr_reports_directory(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)

    assert paths.ocr_reports == paths.library / "40-OCR报告"


def test_paths_expose_catalog_cards_and_base(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)

    assert paths.catalog_cards == paths.library / "50-书目卡片"
    assert paths.catalog_base == paths.library / "书库总览.base"
