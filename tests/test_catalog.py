from pathlib import Path

import pytest

from book_agent.catalog import CatalogService, classify_book
from book_agent.config import AppPaths
from book_agent.storage import Database
from book_agent.vault import VaultManager


def test_classifier_uses_curated_taxonomy() -> None:
    assert classify_book("世界摄影史", "内奥米·罗森布拉姆", "") == "摄影艺术与史论"
    assert classify_book("调色师手册", "Alexis Van Hurkman", "") == "色彩科学与调色"
    assert classify_book("现代电视原理", "姜秀华", "") == "电视与视频工程"
    assert classify_book("我操作的少女正在成为旧日之主", None, "") == "小说"


def test_classifier_assigns_all_active_library_titles_before_preview_noise() -> None:
    expected = {
        "照片的本质": "摄影艺术与史论",
        "摄影师之眼": "摄影艺术与史论",
        "世界摄影史": "摄影艺术与史论",
        "调色师手册 电影和视频调色专业技法 第2版": "色彩科学与调色",
        "调色师手册 -电影和视频调色专业技法": "色彩科学与调色",
        "Colour Sense & Measurement - Chinese GPT Translation Same Layout": "色彩科学与调色",
        "视频技术基础 插图版": "影视制作与技术",
        "电影制作技术手册 (刘戈三) (z-library.sk, 1lib.sk, z-lib.sk)": "影视制作与技术",
        "影视技术基础 插图修订 第3版": "影视制作与技术",
        "现代电视原理": "电视与视频工程",
        "虚拟现实（VR）影像拍摄与制作（数字媒体艺术与技术丛书）": "虚拟现实与数字媒体",
        "艺术学概论（第5版）": "艺术理论",
        "我操作的少女正在成为旧日之主": "小说",
    }
    noisy_preview = "第1章 小说 color correction 摄影 虚拟现实 艺术理论"

    assert {
        title: classify_book(title, None, noisy_preview)
        for title in expected
    } == expected


def test_classifier_uses_bounded_preview_and_safe_fallback() -> None:
    assert classify_book("新书", None, "虚拟现实影像制作") == "虚拟现实与数字媒体"
    assert classify_book("完全未知主题", None, "没有匹配词") == "待分类"
    assert classify_book("新书", None, "x" * 4000 + "摄影") == "待分类"


def _catalog(tmp_path: Path) -> tuple[CatalogService, AppPaths, Database]:
    paths = AppPaths.from_root(tmp_path / "project", vault_root=tmp_path / "vault")
    VaultManager(paths).ensure_layout()
    database = Database(paths.database, root=paths.root)
    database.initialize()
    return CatalogService(paths, database), paths, database


def _book(paths: AppPaths, *, status: str = "ready") -> dict[str, object]:
    original = paths.originals / "世界摄影史.pdf"
    original.write_bytes(b"book")
    parsed = paths.parsed / ("a" * 24) / "正文.md"
    parsed.parent.mkdir(parents=True, exist_ok=True)
    parsed.write_text("# 世界摄影史\n", encoding="utf-8")
    return {
        "book_id": "a" * 24,
        "title": "世界摄影史",
        "author": "内奥米·罗森布拉姆",
        "source_format": "pdf",
        "original_path": str(original),
        "parsed_path": str(parsed),
        "status": status,
        "created_at": "2026-07-22 01:02:03",
        "updated_at": "2026-07-22 01:02:03",
    }


def test_sync_book_creates_links_and_initial_category(tmp_path: Path) -> None:
    service, paths, _ = _catalog(tmp_path)

    card = service.sync_book(_book(paths))

    text = card.read_text(encoding="utf-8")
    assert card.parent == paths.catalog_cards
    assert card.name.endswith(f"-{'a' * 24}.md")
    assert 'primary_category: "摄影艺术与史论"' in text
    assert "custom_categories: []" in text
    assert 'source_link: "[[书库/10-原始书籍/世界摄影史.pdf]]"' in text
    assert f'parsed_link: "[[书库/20-解析文本/{"a" * 24}/正文.md]]"' in text
    assert "[[书库/10-原始书籍/世界摄影史.pdf|打开原始书籍]]" in text


def test_sync_book_preserves_user_categories_and_updates_system_fields(
    tmp_path: Path,
) -> None:
    service, paths, _ = _catalog(tmp_path)
    card = service.sync_book(_book(paths))
    text = card.read_text(encoding="utf-8").replace(
        'primary_category: "摄影艺术与史论"',
        'primary_category: "我的摄影研究"',
    ).replace(
        "custom_categories: []",
        'custom_categories:\n  - "必读"\n  - "视觉文化"',
    )
    card.write_text(text, encoding="utf-8")

    same_card = service.sync_book(_book(paths, status="keyword_only"))

    updated = same_card.read_text(encoding="utf-8")
    assert same_card == card
    assert 'primary_category: "我的摄影研究"' in updated
    assert '  - "必读"' in updated
    assert '  - "视觉文化"' in updated
    assert 'library_status: "keyword_only"' in updated


def test_sync_book_rejects_malformed_existing_user_categories(tmp_path: Path) -> None:
    service, paths, _ = _catalog(tmp_path)
    card = service.sync_book(_book(paths))
    card.write_text("---\nprimary_category:\n---\n", encoding="utf-8")

    try:
        service.sync_book(_book(paths))
    except ValueError as exc:
        assert "primary_category" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("malformed category must be rejected")


def test_sync_book_rejects_catalog_directory_symlink(tmp_path: Path) -> None:
    service, paths, _ = _catalog(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.catalog_cards.rmdir()
    paths.catalog_cards.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|safely"):
        service.sync_book(_book(paths))

    assert list(outside.iterdir()) == []


def test_clean_ocr_report_is_linked_without_warning_status(tmp_path: Path) -> None:
    service, paths, _ = _catalog(tmp_path)
    book_id = "a" * 24
    paths.ocr_reports.mkdir(parents=True, exist_ok=True)
    (paths.ocr_reports / f"{book_id}-OCR处理报告.md").write_text(
        "---\nskipped_pages: 0\n---\n# OCR 处理报告\n",
        encoding="utf-8",
    )

    card = service.sync_book(_book(paths))

    text = card.read_text(encoding="utf-8")
    assert 'ocr_status: "not_required"' in text
    assert "OCR处理报告" in text


def test_sync_all_is_idempotent_and_writes_four_base_views(tmp_path: Path) -> None:
    service, paths, database = _catalog(tmp_path)
    for index, (title, source_format) in enumerate(
        (("世界摄影史", "pdf"), ("我操作的少女正在成为旧日之主", "txt"))
    ):
        book_id = f"{index + 1:024x}"
        original = paths.originals / f"{title}.{source_format}"
        original.write_bytes(title.encode("utf-8"))
        database.create_book(
            book_id,
            title,
            None,
            source_format,
            f"{index + 100:064x}",
            str(original),
            status="ready",
        )

    first = service.sync_all()
    second = service.sync_all()

    assert first.total == 2
    assert first.created == 2
    assert first.updated == 0
    assert second.total == 2
    assert second.created == 0
    assert second.updated == 2
    assert len(list(paths.catalog_cards.glob("*.md"))) == 2
    base = paths.catalog_base.read_text(encoding="utf-8")
    assert 'file.inFolder("书库/50-书目卡片")' in base
    for name in ("按主分类", "全部书籍", "待 OCR", "OCR 有警告"):
        assert f'name: "{name}"' in base
    assert "property: note.primary_category" in base


def test_sync_all_uses_only_bounded_parsed_preview_for_classification(
    tmp_path: Path,
) -> None:
    service, paths, database = _catalog(tmp_path)
    book_id = "b" * 24
    original = paths.originals / "新书.txt"
    original.write_text("source", encoding="utf-8")
    parsed = paths.parsed / book_id / "正文.md"
    parsed.parent.mkdir(parents=True)
    parsed.write_text("虚拟现实" + "x" * 10000, encoding="utf-8")
    database.create_book(
        book_id,
        "新书",
        None,
        "txt",
        "c" * 64,
        str(original),
        status="ready",
        parsed_path=str(parsed),
    )

    service.sync_all()

    card = next(paths.catalog_cards.glob(f"*-{book_id}.md"))
    assert 'primary_category: "虚拟现实与数字媒体"' in card.read_text(encoding="utf-8")
