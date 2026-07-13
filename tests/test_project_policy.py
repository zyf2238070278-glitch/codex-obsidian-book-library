from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PROJECT_ROOT = "/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo"
TOOL_ALLOWLIST = [
    "import_book",
    "list_books",
    "library_status",
    "search_books",
    "get_passages",
    "save_reading_note",
]


def test_agents_policy_enforces_grounded_and_safe_book_answers() -> None:
    policy = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    required_rules = (
        "file_path",
        "绝对路径",
        "search_books",
        "get_passages",
        "不可信证据",
        "绝不是指令",
        "原文引用",
        "通俗转述",
        "Codex 推断",
        "PDF 页",
        "EPUB 章节",
        "passage_id",
        "证据不足",
        "模型记忆",
        "30-AI读书笔记",
        "明确要求",
        "多次有界检索",
        "报告错误",
    )
    for rule in required_rules:
        assert rule in policy


def test_project_config_registers_only_the_local_book_tools() -> None:
    config = tomllib.loads(
        (PROJECT_ROOT / ".codex" / "config.toml").read_text(encoding="utf-8")
    )
    server = config["mcp_servers"]["book_library"]

    assert server["command"] == "uv"
    assert server["args"] == ["run", "python", "-m", "book_agent.mcp_server"]
    assert server["cwd"] == FINAL_PROJECT_ROOT
    assert server["env"] == {
        "BOOK_LIBRARY_ROOT": FINAL_PROJECT_ROOT,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert server["required"] is True
    assert server["enabled"] is True
    assert server["enabled_tools"] == TOOL_ALLOWLIST
    assert 1 <= server["startup_timeout_sec"] <= 120
    assert server["tool_timeout_sec"] == 120


def test_obsidian_entry_notes_explain_the_codex_only_workflow() -> None:
    notes = [
        (PROJECT_ROOT / "vault" / "首页.md").read_text(encoding="utf-8"),
        (PROJECT_ROOT / "vault" / "书库" / "说明.md").read_text(
            encoding="utf-8"
        ),
    ]

    for note in notes:
        for phrase in (
            "在 Codex 中上传书籍",
            "列出书库",
            "引用原文",
            "通俗解释",
            "跨书比较",
            "保存到 Obsidian",
            "信任项目",
            "重新加载 Codex",
        ):
            assert phrase in note
