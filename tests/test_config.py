from pathlib import Path

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
        database=resolved_root / "data" / "library.sqlite3",
        models=resolved_root / "data" / "models",
    )
