from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_ALLOWLIST = [
    "import_book",
    "list_books",
    "library_status",
    "search_books",
    "get_passages",
    "save_reading_note",
    "start_ocr",
    "start_pending_ocr",
    "ocr_status",
    "pause_ocr",
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
        "短引文",
        "保持简短",
        "除非用户明确要求原文",
        "优先简洁转述",
        "减少 token 消耗",
        "报告错误",
    )
    for rule in required_rules:
        assert rule in policy


def test_generated_project_config_registers_only_local_book_tools(tmp_path: Path) -> None:
    from installer.install_macos import render_codex_config

    project_root = tmp_path / "portable-project"
    vault = tmp_path / "portable-vault"
    config = tomllib.loads(
        render_codex_config(
            project_root=project_root,
            vault=vault,
            python=project_root / ".venv" / "bin" / "python",
        )
    )
    server = config["mcp_servers"]["book_library"]

    assert server["command"] == str(project_root / ".venv" / "bin" / "python")
    assert server["args"] == ["-m", "book_agent.mcp_server"]
    assert server["cwd"] == str(project_root)
    assert server["env"] == {
        "BOOK_LIBRARY_ROOT": str(project_root),
        "BOOK_LIBRARY_OBSIDIAN_VAULT": str(vault),
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
