from pathlib import Path
import re

from book_agent.parsers.registry import SUPPORTED_EXTENSIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = PROJECT_ROOT / "docs" / "USER_GUIDE.md"


def _guide() -> str:
    return GUIDE_PATH.read_text(encoding="utf-8")


def _bash_commands(guide: str) -> list[str]:
    blocks = re.findall(r"```bash\n(.*?)\n```", guide, re.DOTALL)
    return [
        line.strip()
        for block in blocks
        for line in block.splitlines()
        if line.strip()
    ]


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


def test_user_guide_covers_setup_and_codex_reload() -> None:
    guide = _guide()

    for phrase in (
        "uv sync --extra dev --extra semantic",
        "信任项目",
        "重新加载 Codex",
    ):
        assert phrase in guide


def test_user_guide_formats_match_parser_registry_exactly() -> None:
    guide = _guide()
    formats = guide.split("当前支持的文件类型是：", 1)[1].split(
        "不支持的格式", 1
    )[0]
    documented_extensions = set(re.findall(r"`(\.[a-z0-9]+)`", formats))

    assert documented_extensions == SUPPORTED_EXTENSIONS
    assert ".markdown" not in guide


def test_user_guide_downloads_model_after_dependency_install() -> None:
    guide = _guide()
    commands = _bash_commands(guide)
    download_pattern = re.compile(
        r"SentenceTransformer\(\s*(['\"])intfloat/multilingual-e5-small\1,\s*"
        r"cache_folder\s*=\s*(['\"])data/models\2\s*\)"
    )
    download_command = next(
        command
        for command in commands
        if "uv run python -c" in command and download_pattern.search(command)
    )

    assert guide.index("uv sync --extra dev --extra semantic") < guide.index(
        download_command
    )


def test_user_guide_has_forced_offline_semantic_verification() -> None:
    guide = _guide()
    command = next(
        command
        for command in _bash_commands(guide)
        if command.startswith("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ")
    )

    assert "uv run --offline python -c" in command
    assert re.search(
        r"E5EmbeddingProvider\(Path\((['\"])data/models\1\)\)", command
    )
    assert "print(p.available)" in command
    assert "p.embed_query(" in command
    assert ".shape" in command
    assert re.search(r"```text\nTrue\n\(384,\)\n```", guide)


def test_user_guide_keyword_only_recovery_order_matches_startup_behavior() -> None:
    guide = _guide()
    keyword_only_section = guide.split("`keyword_only`：", 1)[1].split("\n- `", 1)[0]
    subbranches = [
        line.strip()
        for line in keyword_only_section.splitlines()
        if line.startswith("  - ")
    ]
    model_unavailable = next(
        line
        for line in subbranches
        if "语义模型未启用" in line and "缓存缺失" in line
    )
    index_failure = next(
        line for line in subbranches if "语义索引失败" in line
    )

    assert "关键词检索仍可用" in keyword_only_section
    download = model_unavailable.index("下载模型")
    reload_process = model_unavailable.index("重新加载 Codex/MCP")
    reimport = model_unavailable.index("重新导入")
    inspect_error = index_failure.index("error")
    repair = index_failure.index("修复")
    retry = index_failure.index("重新导入")

    assert download < reload_process < reimport
    assert inspect_error < repair < retry
    assert "下载模型" not in index_failure


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


def test_user_guide_uses_actual_vault_paths_and_readable_spacing() -> None:
    guide = _guide()

    for path in (
        "vault/书库/30-AI读书笔记/",
        "vault/书库/10-原始书籍/",
        "vault/书库/20-解析文本/",
    ):
        assert f"`{path}`" in guide

    assert "`vault/30-AI读书笔记/`" not in guide
    assert "`vault/10-原始书籍/`" not in guide
    assert "`vault/20-解析文本/`" not in guide
    assert "Codex依据" not in guide
    assert "Codex 依据" in guide
