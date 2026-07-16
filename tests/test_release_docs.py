from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_PATHS = (
    Path("README.md"),
    Path("docs/安装说明.md"),
    Path("docs/使用说明.md"),
    Path("docs/常见问题.md"),
    Path("docs/隐私与数据存放.md"),
)

PUBLIC_INSTALL_DOCS = (
    Path("README.md"),
    Path("docs/安装说明.md"),
    Path("docs/USER_GUIDE.md"),
    Path("docs/常见问题.md"),
)

CODEX_FIRST_FLOW = (
    "git clone https://github.com/zyf2238070278-glitch/codex-obsidian-book-library.git",
    "请安装并检查这个书库",
    "完整退出并重启 Codex",
    "检查书库状态",
)

STALE_INSTALL_TEXT = (
    "v0.2.0-beta.1",
    "约 292 MB",
    "下载全量 ZIP",
    "uv sync --extra dev --extra semantic",
    "/Users/" + "zhaoyunfei/",
)

RELEASE_TEXT_PATHS = (
    *DOCUMENT_PATHS,
    Path("LICENSE"),
    Path("THIRD_PARTY_NOTICES.md"),
    Path("third_party/uv/LICENSE-APACHE"),
    Path("third_party/uv/LICENSE-MIT"),
    Path("third_party/model/LICENSE-MIT"),
)

MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
PUBLISHER_HOME_PREFIX = "/Users/" + "zhaoyunfei/"
SHELL_PATH_PLACEHOLDERS = (
    "<PROJECT_ROOT>",
    "<OBSIDIAN_VAULT>",
    "<OUTPUT_DIR>",
    "<UV_BINARY>",
)


def _read(relative: str | Path) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def _assert_contains(text: str, phrases: tuple[str, ...]) -> None:
    for phrase in phrases:
        assert phrase in text, phrase


def _contains_publisher_home(path: Path) -> bool:
    return PUBLISHER_HOME_PREFIX.encode() in path.read_bytes()


def test_publisher_home_detector_checks_non_utf8_bytes(tmp_path: Path) -> None:
    binary = tmp_path / "binary.dat"
    binary.write_bytes(b"\xff\xfe\x00" + PUBLISHER_HOME_PREFIX.encode())

    assert _contains_publisher_home(binary) is True


def test_all_tracked_regular_files_exclude_publisher_home_path() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    offenders: list[str] = []
    for encoded_path in tracked:
        if not encoded_path:
            continue
        path = PROJECT_ROOT / encoded_path.decode("utf-8", errors="surrogateescape")
        if not stat.S_ISREG(path.lstat().st_mode):
            continue
        if _contains_publisher_home(path):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert offenders == []


def test_historical_bash_examples_have_no_path_placeholders() -> None:
    plan_paths = (
        PROJECT_ROOT / "docs" / "plans",
        PROJECT_ROOT / "docs" / "superpowers" / "plans",
    )
    offenders: list[str] = []
    for plan_root in plan_paths:
        for path in plan_root.glob("*.md"):
            for block in re.findall(
                r"```(?:bash|sh|zsh)\n(.*?)\n```",
                path.read_text(encoding="utf-8"),
                re.DOTALL,
            ):
                if any(placeholder in block for placeholder in SHELL_PATH_PLACEHOLDERS):
                    offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert offenders == []


def test_quick_start_uses_clone_and_install_selected_vault_paths() -> None:
    quick_start = _read("outputs/书库RAG快速开始.md")

    _assert_contains(
        quick_start,
        (
            "当前打开并信任的 Git 克隆目录",
            "安装时可以选择已有 Vault",
            "默认使用项目内的 `Obsidian书库/`",
        ),
    )
    assert "<PROJECT_ROOT>" not in quick_start
    assert "<OBSIDIAN_VAULT>" not in quick_start


def test_release_document_and_license_set_is_complete() -> None:
    missing = [
        path.as_posix()
        for path in RELEASE_TEXT_PATHS
        if not (PROJECT_ROOT / path).is_file()
    ]

    assert missing == []


@pytest.mark.parametrize("path", PUBLIC_INSTALL_DOCS)
def test_public_install_docs_use_codex_first_flow(path: Path) -> None:
    text = _read(path)
    positions = [text.index(item) for item in CODEX_FIRST_FLOW]

    assert positions == sorted(positions)
    for stale in STALE_INSTALL_TEXT:
        assert stale not in text


def test_readme_gives_the_complete_shortest_macos_path() -> None:
    readme = _read("README.md")
    install = readme[readme.index("## Git 安装") :]

    steps = (
        *CODEX_FIRST_FLOW,
        "导入这本书",
    )
    _assert_contains(
        install,
        (
            *steps,
            "Apple Silicon",
            "macOS 16",
            "约 500 MB",
            "首次安装需要联网",
            "无需预装 Homebrew、Python、Xcode 或 uv",
            "项目本地 Python",
            "锁定版本",
            "Apple Vision",
            "RapidOCR",
            "打开并信任",
            ".codex/config.toml",
            "PDF",
            "EPUB",
            "Markdown",
            "TXT",
            "Obsidian 只用来浏览",
            "<项目目录>/Obsidian书库",
            './install-macos.command --vault "<已有 Vault 的绝对路径>"',
            "书库/00-待导入",
            "书库/10-原始书籍",
            "书库/20-解析文本",
            "书库/30-AI读书笔记",
            "移动项目目录",
            "重新运行",
            "备份",
        ),
    )
    assert [install.index(step) for step in steps] == sorted(
        install.index(step) for step in steps
    )


def test_public_docs_explain_online_local_and_recovery_boundaries() -> None:
    combined = "\n".join(_read(path) for path in PUBLIC_INSTALL_DOCS)

    _assert_contains(
        combined,
        (
            "约 500 MB",
            "语义模型",
            "Python 包",
            "首次安装需要联网",
            "锁定版本",
            "项目自带固定版本的 uv",
            "项目本地 Python",
            "无需预装 Homebrew、Python、Xcode 或 uv",
            "Apple Silicon",
            "macOS 16",
            "Apple Vision",
            "RapidOCR",
            "本机运行",
            "移动项目目录",
            "重新运行",
            "不会删除已有书籍或笔记",
            "删除整个项目目录",
            "先备份",
        ),
    )


def test_readme_gives_copyable_git_clone_command() -> None:
    readme = _read("README.md")
    install = _read("docs/安装说明.md")

    for text in (readme, install):
        _assert_contains(
            text,
            (
                CODEX_FIRST_FLOW[0],
                "cd codex-obsidian-book-library",
                "打开并信任",
                "完整退出并重启 Codex",
            ),
        )


def test_readme_preserves_post_install_book_workflow() -> None:
    readme = _read("README.md")
    steps = (
        "完整退出并重启 Codex",
        "检查书库状态",
        "导入这本书",
    )
    assert [readme.index(step) for step in steps] == sorted(
        readme.index(step) for step in steps
    )


def test_gatekeeper_help_uses_only_safe_scoped_actions() -> None:
    public_docs = "\n".join(_read(path) for path in DOCUMENT_PATHS)

    _assert_contains(
        public_docs,
        (
            "Finder",
            "右键或按住 Control 点按",
            "系统设置",
            "隐私与安全性",
            "仍要打开",
        ),
    )
    for unsafe in ("spctl", "xattr", "全局关闭 Gatekeeper", "允许任何来源"):
        assert unsafe not in public_docs


def test_usage_guide_covers_grounded_workflows_and_bounded_retrieval() -> None:
    usage = _read("docs/使用说明.md")

    _assert_contains(
        usage,
        (
            "检查书库状态",
            "导入这本书",
            "列出书库里的书",
            "引用两段短原文",
            "《书名》",
            "PDF 页",
            "EPUB 章节",
            "用通俗的话解释",
            "比较《",
            "保存为读书笔记",
            "只有明确说",
            "按需检索",
            "不会把整本书塞进上下文",
            "节省 token",
        ),
    )


def test_install_and_privacy_docs_state_runtime_and_data_boundaries() -> None:
    install = _read("docs/安装说明.md")
    privacy = _read("docs/隐私与数据存放.md")
    combined = install + "\n" + privacy

    _assert_contains(
        combined,
        (
            "Apple Silicon",
            "macOS 16",
            "不支持 Intel Mac",
            "不支持 Windows",
            "首次安装需要联网",
            "约 500 MB",
            "Python 包",
            "锁定版本",
            "项目本地 Python",
            "语义检索在本机运行",
            "不代表完全离线",
            "不是零 token",
            "不是零内容传输",
            "选中的少量短段落会进入 Codex 对话",
            "原书、数据库和笔记默认保留在本机",
            "data/library.sqlite3",
            "data/models",
            ".venv",
            ".codex/config.toml",
            "Obsidian书库",
            "备份",
            "卸载",
        ),
    )


def test_faq_answers_the_expected_first_run_questions() -> None:
    faq = _read("docs/常见问题.md")

    _assert_contains(
        faq,
        (
            "MCP 面板里没有 `book_library`",
            "也可能是正常的",
            "以“检查书库状态”的结果为准",
            "新任务不需要每次输入命令",
            "打开并信任整个项目",
            "0 本",
            "正常但还没有导入书",
            "更换 Vault",
            "移动项目目录",
            "Gatekeeper",
            "PDF、EPUB、Markdown 和 TXT",
            "扫描版 PDF",
            "明确说“开始 OCR 这本书”",
            "Apple Vision",
            "RapidOCR",
            "首次安装",
            "比较慢",
        ),
    )


def test_release_text_contains_no_private_paths_or_live_credentials() -> None:
    forbidden_literals = [
        "/Users/",
        "/home/",
        "C:\\Users\\",
    ]
    home_account = Path.home().name
    if home_account.casefold() not in {"admin", "guest", "root", "runner", "user"}:
        forbidden_literals.append(home_account)
    credential_patterns = (
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}"),
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"open\.feishu\.cn/open-apis/bot/v2/hook/[^\s)]+"),
    )

    for path in RELEASE_TEXT_PATHS:
        text = _read(path)
        assert not any(marker in text for marker in forbidden_literals), path
        assert not any(pattern.search(text) for pattern in credential_patterns), path


def test_mit_and_apache_license_files_are_full_non_placeholder_texts() -> None:
    mit_paths = (
        Path("LICENSE"),
        Path("third_party/uv/LICENSE-MIT"),
        Path("third_party/model/LICENSE-MIT"),
    )
    for path in mit_paths:
        text = _read(path)
        _assert_contains(
            text,
            (
                "MIT License",
                "Copyright (c)",
                "Permission is hereby granted, free of charge",
                "The above copyright notice and this permission notice",
                'THE SOFTWARE IS PROVIDED "AS IS"',
                "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE",
            ),
        )
        assert len(text) > 900
        assert "<YEAR>" not in text
        assert "<COPYRIGHT HOLDER>" not in text

    apache = _read("third_party/uv/LICENSE-APACHE")
    _assert_contains(
        apache,
        (
            "Apache License",
            "Version 2.0, January 2004",
            "1. Definitions.",
            "2. Grant of Copyright License.",
            "3. Grant of Patent License.",
            "4. Redistribution.",
            "9. Accepting Warranty or Additional Liability.",
            "END OF TERMS AND CONDITIONS",
            "APPENDIX: How to apply the Apache License to your work.",
        ),
    )
    assert len(apache) > 10_000


def test_third_party_notice_pins_exact_components_and_license_paths() -> None:
    notice = _read("THIRD_PARTY_NOTICES.md")

    _assert_contains(
        notice,
        (
            "uv 0.11.26",
            "https://github.com/astral-sh/uv/tree/0.11.26",
            "Apache-2.0 OR MIT",
            "third_party/uv/LICENSE-APACHE",
            "third_party/uv/LICENSE-MIT",
            "intfloat/multilingual-e5-small",
            MODEL_REVISION,
            f"https://huggingface.co/intfloat/multilingual-e5-small/tree/{MODEL_REVISION}",
            "MIT",
            "third_party/model/LICENSE-MIT",
            "模型权重随 all-in-one ZIP 分发",
            "does not contain a separate LICENSE file",
            "standard MIT license text",
        ),
    )
