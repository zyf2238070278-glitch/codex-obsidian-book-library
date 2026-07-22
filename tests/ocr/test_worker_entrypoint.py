from __future__ import annotations

import subprocess
from pathlib import Path

from book_agent.config import AppPaths
from book_agent.ocr.light import LightOcrEngine
from book_agent.ocr.service import OcrService
from book_agent.ocr_worker import build_light_ocr_engine
from book_agent.storage import Database


def _runtime(paths: AppPaths) -> None:
    paths.light_ocr_worker.parent.mkdir(parents=True, exist_ok=True)
    paths.light_ocr_worker.write_text("// worker", encoding="utf-8")
    paths.light_ocr_package.parent.mkdir(parents=True, exist_ok=True)
    paths.light_ocr_package.write_text("{}", encoding="utf-8")


def test_build_light_ocr_engine_uses_supported_arm64_node(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    _runtime(paths)
    node = tmp_path / "bin" / "node"
    node.parent.mkdir()
    node.write_text("#!/bin/sh\n", encoding="utf-8")
    node.chmod(0o755)

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output = "v24.4.1\n" if argv[-1] == "--version" else "arm64\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    engine = build_light_ocr_engine(
        paths,
        environ={},
        find_executable=lambda name: str(node) if name == "node" else None,
        run_command=runner,
    )

    assert isinstance(engine, LightOcrEngine)
    assert engine.node == node.resolve()
    assert engine.worker == paths.light_ocr_worker


def test_build_light_ocr_engine_is_optional_when_runtime_is_unavailable(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")

    assert build_light_ocr_engine(
        paths,
        environ={},
        find_executable=lambda _: None,
    ) is None


def test_build_light_ocr_engine_rejects_unsupported_node_version(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path / "project")
    _runtime(paths)
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\n", encoding="utf-8")
    node.chmod(0o755)

    assert build_light_ocr_engine(
        paths,
        environ={},
        find_executable=lambda _: str(node),
        run_command=lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, "v23.0.0\n", ""
        ),
    ) is None


def test_detached_service_environment_reaches_light_ocr_composition(
    tmp_path: Path, monkeypatch
) -> None:
    paths = AppPaths.from_root(tmp_path / "project", tmp_path / "vault")
    paths.root.mkdir(parents=True)
    paths.vault.mkdir()
    _runtime(paths)
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\n", encoding="utf-8")
    node.chmod(0o755)
    monkeypatch.setenv("BOOK_LIBRARY_LIGHT_OCR_NODE", str(node))
    database = Database(paths.database, root=paths.root)
    database.initialize()
    environment = OcrService(paths, database)._worker_environment()

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output = "v24.4.1\n" if argv[-1] == "--version" else "arm64\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    engine = build_light_ocr_engine(
        paths,
        environ=environment,
        find_executable=lambda _: None,
        run_command=runner,
    )

    assert isinstance(engine, LightOcrEngine)
    assert engine.node == node.resolve()
