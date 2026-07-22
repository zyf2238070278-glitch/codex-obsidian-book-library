from pathlib import Path
import re

from book_agent.parsers.registry import SUPPORTED_EXTENSIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = PROJECT_ROOT / "docs" / "USER_GUIDE.md"
POLICY_PATH = PROJECT_ROOT / "AGENTS.md"

CODEX_FIRST_FLOW = (
    "git clone https://github.com/zyf2238070278-glitch/codex-obsidian-book-library.git",
    "请安装并检查这个书库",
    "完整退出并重启 Codex",
    "检查书库状态",
)


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

    assert (
        "git clone https://github.com/zyf2238070278-glitch/"
        "codex-obsidian-book-library.git && "
        "cd codex-obsidian-book-library && ./install-from-github.command"
    ) in guide

    positions = [guide.index(phrase) for phrase in CODEX_FIRST_FLOW]
    assert positions == sorted(positions)
    for phrase in (
        "打开并信任整个项目",
        "首次安装需要联网",
        "约 500 MB",
        "Python 包",
        "项目本地 Python",
        "锁定版本",
        "无需预装 Homebrew、Python、Xcode 或 uv",
        "无需预装 Node.js",
    ):
        assert phrase in guide
    for stale in (
        "uv sync --extra dev --extra semantic",
        "uv run python -c",
        "/Users/" + "zhaoyunfei/",
    ):
        assert stale not in guide


def test_user_guide_formats_match_parser_registry_exactly() -> None:
    guide = _guide()
    formats = guide.split("当前支持的文件类型是：", 1)[1].split(
        "不支持的格式", 1
    )[0]
    documented_extensions = set(re.findall(r"`(\.[a-z0-9]+)`", formats))

    assert documented_extensions == SUPPORTED_EXTENSIONS
    assert ".markdown" not in guide


def test_user_guide_explains_local_runtime_and_repair() -> None:
    guide = _guide()

    for phrase in (
        "项目自带固定版本的 uv",
        "语义模型和 OCR 模型",
        "Apple Vision",
        "RapidOCR",
        "Light OCR",
        "无需预装 Node.js",
        "本机运行",
        "移动项目目录",
        "重新运行",
        "不会删除已有书籍或笔记",
        "删除整个项目目录",
        "先备份",
    ):
        assert phrase in guide


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
    rerun_installer = model_unavailable.index("请安装并检查这个书库")
    reload_process = model_unavailable.index("完整退出并重启 Codex")
    reimport = model_unavailable.index("重新导入")
    inspect_error = index_failure.index("error")
    repair = index_failure.index("修复")
    retry = index_failure.index("重新导入")

    assert rerun_installer < reload_process < reimport
    assert inspect_error < repair < retry
    assert "请安装并检查这个书库" not in index_failure


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
        "当前 Obsidian 仓库",
        "不需要切换",
    ):
        assert phrase in guide


def test_guides_use_active_obsidian_vault_and_project_data_paths() -> None:
    guide = _guide()

    for phrase in (
        "当前 Obsidian 仓库",
        "`书库/10-原始书籍/`",
        "`书库/20-解析文本/`",
        "`书库/30-AI读书笔记/`",
        "项目的 `data/`",
    ):
        assert phrase in guide
    for obsolete in (
        "/Users/",
        "作为 Obsidian vault",
        "Open folder as vault",
        "vault/书库/",
    ):
        assert obsolete not in guide

    assert "Codex依据" not in guide
    assert "Codex 依据" in guide


def test_agents_policy_names_ai_notes_in_the_current_obsidian_vault() -> None:
    policy = POLICY_PATH.read_text(encoding="utf-8")

    assert "当前 Obsidian 仓库 `书库/30-AI读书笔记`" in policy
    assert "vault/书库/30-AI读书笔记" not in policy
