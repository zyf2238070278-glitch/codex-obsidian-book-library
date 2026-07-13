from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = PROJECT_ROOT / "docs" / "USER_GUIDE.md"


def _guide() -> str:
    return GUIDE_PATH.read_text(encoding="utf-8")


def test_user_guide_has_the_required_exact_sections() -> None:
    guide = _guide()

    for section in (
        "首次设置",
        "在 Codex 中上传书籍",
        "引用原文",
        "通俗解释",
        "保存到 Obsidian",
        "扫描版 PDF",
        "隐私边界",
    ):
        assert re.search(rf"^## {re.escape(section)}$", guide, re.MULTILINE)


def test_user_guide_covers_setup_formats_and_codex_reload() -> None:
    guide = _guide()

    for phrase in (
        "uv sync --extra dev --extra semantic",
        "信任项目",
        "重新加载 Codex",
        "PDF",
        "EPUB",
        "Markdown",
        "TXT",
    ):
        assert phrase in guide


def test_user_guide_explains_status_recovery_and_source_locations() -> None:
    guide = _guide()

    for phrase in (
        "library_status",
        "needs_ocr",
        "failed",
        "keyword_only",
        "duplicate",
        "PDF 阅读器页码",
        "EPUB 章节",
    ):
        assert phrase in guide


def test_user_guide_sets_privacy_token_and_obsidian_browsing_boundaries() -> None:
    guide = _guide()

    for phrase in (
        "完整书籍",
        "索引",
        "向量",
        "保留在本机",
        "选中少量段落",
        "Codex 上下文",
        "vault/",
        "作为 Obsidian vault",
        "仅用于浏览",
    ):
        assert phrase in guide
