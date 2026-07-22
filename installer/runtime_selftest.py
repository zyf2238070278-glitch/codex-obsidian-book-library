#!/usr/bin/env python3
"""Verify that every packaged offline runtime component can start."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SelfTestResult:
    embedding_dimensions: int


class SelfTestError(RuntimeError):
    pass


def _probe_imports() -> None:
    for module_name in (
        "fitz",
        "ebooklib",
        "mcp",
        "numpy",
        "onnxruntime",
        "rapidocr",
        "sentence_transformers",
    ):
        importlib.import_module(module_name)


def _embedding_provider_class() -> type:
    from book_agent.embeddings import E5EmbeddingProvider

    return E5EmbeddingProvider


def _probe_embedding(model_root: Path) -> int:
    import numpy as np

    previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        provider = _embedding_provider_class()(model_root)
        if not provider.available:
            raise SelfTestError("语义模型不可用。")
        vector = provider.embed_query("安装自检")
    finally:
        if previous_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_hf_offline
        if previous_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers_offline

    array = np.asarray(vector)
    if array.ndim != 1:
        raise SelfTestError("语义模型必须返回一维向量。")
    return int(array.shape[0])


def _probe_rapidocr(model_root: Path) -> None:
    from book_agent.ocr.rapid import REQUIRED_MODEL_FILES

    for filename in REQUIRED_MODEL_FILES:
        model = model_root / filename
        try:
            info = model.lstat()
        except OSError as exc:
            raise SelfTestError(f"RapidOCR 模型缺失或为空：{filename}") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or model.is_symlink()
            or info.st_size <= 0
        ):
            raise SelfTestError(f"RapidOCR 模型缺失或为空：{filename}")


def _probe_vision(helper: Path) -> None:
    try:
        info = helper.lstat()
    except OSError as exc:
        raise SelfTestError(f"Vision OCR helper 不可用：{helper}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or helper.is_symlink()
        or info.st_size <= 0
        or not os.access(helper, os.X_OK)
    ):
        raise SelfTestError(f"Vision OCR helper 不可用：{helper}")

    try:
        completed = subprocess.run(
            [str(helper), "--capabilities"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SelfTestError(f"Vision OCR helper 无法运行：{exc}") from exc

    if completed.returncode != 0:
        raise SelfTestError(
            f"Vision OCR helper capabilities 失败：退出码 {completed.returncode}"
        )
    if completed.stderr:
        raise SelfTestError("Vision OCR helper capabilities 输出了错误信息。")
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SelfTestError("Vision OCR helper capabilities 不是有效 JSON。") from exc
    if type(payload) is not dict or set(payload) != {"schema_version", "languages"}:
        raise SelfTestError("Vision OCR helper capabilities schema 不受支持。")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 2:
        raise SelfTestError("Vision OCR helper schema 版本不受支持。")
    languages = payload["languages"]
    if (
        type(languages) is not list
        or any(type(language) is not str or not language for language in languages)
        or len(set(languages)) != len(languages)
        or not {"zh-Hans", "en-US"}.issubset(languages)
    ):
        raise SelfTestError("Vision OCR helper 不支持中文或英文识别。")


def _probe_light_ocr(project_root: Path) -> None:
    node = project_root / "runtime" / "node" / "bin" / "node"
    worker = project_root / "scripts" / "light_ocr_worker.mjs"
    for path, label, executable in (
        (node, "Node.js", True),
        (worker, "worker", False),
    ):
        try:
            info = path.lstat()
        except OSError as exc:
            raise SelfTestError(f"Light OCR {label} 不可用：{path}") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or path.is_symlink()
            or info.st_size <= 0
            or (executable and not os.access(path, os.X_OK))
        ):
            raise SelfTestError(f"Light OCR {label} 不可用：{path}")

    try:
        completed = subprocess.run(
            [str(node), str(worker)],
            cwd=project_root,
            input='{"op":"close"}\n',
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SelfTestError(f"Light OCR 启动自检无法运行：{exc}") from exc
    if completed.returncode != 0:
        raise SelfTestError(
            f"Light OCR 启动自检失败：退出码 {completed.returncode}"
        )
    if completed.stdout or completed.stderr:
        raise SelfTestError("Light OCR 启动自检产生了意外输出。")


def _load_mcp_runtime() -> tuple[object, type, type, Callable[..., object]]:
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    return anyio, ClientSession, StdioServerParameters, stdio_client


def _validate_mcp_library_status(response: object) -> None:
    if getattr(response, "isError", None) is not False:
        raise SelfTestError("MCP library_status 调用返回了协议错误。")
    if getattr(response, "error", None) is not None:
        raise SelfTestError("MCP library_status 调用返回了处理错误。")

    payload = getattr(response, "structuredContent", None)
    if payload is None:
        content = getattr(response, "content", None)
        if type(content) is not list:
            raise SelfTestError("MCP library_status 响应格式无效。")
        text_items = [
            getattr(item, "text", None)
            for item in content
            if getattr(item, "type", None) == "text"
        ]
        if len(text_items) != 1 or type(text_items[0]) is not str:
            raise SelfTestError("MCP library_status 文本响应格式无效。")
        try:
            payload = json.loads(text_items[0])
        except json.JSONDecodeError as exc:
            raise SelfTestError("MCP library_status 文本响应不是有效 JSON。") from exc
    if type(payload) is not dict:
        raise SelfTestError("MCP library_status 结构化响应格式无效。")
    if payload.get("ok") is not True:
        raise SelfTestError("MCP library_status 业务自检失败。")


async def _probe_mcp_async(project_root: Path, vault: Path) -> None:
    anyio, client_session, server_parameters, stdio = _load_mcp_runtime()
    try:
        vault.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SelfTestError(f"MCP Obsidian Vault 无法创建：{exc}") from exc
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "BOOK_LIBRARY_ROOT": str(project_root),
            "BOOK_LIBRARY_OBSIDIAN_VAULT": str(vault),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    parameters = server_parameters(
        command=sys.executable,
        args=["-m", "book_agent.mcp_server"],
        cwd=str(project_root),
        env=child_environment,
    )
    with anyio.fail_after(30):
        async with stdio(parameters) as (read_stream, write_stream):
            async with client_session(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                if "library_status" not in {tool.name for tool in tools.tools}:
                    raise SelfTestError("MCP 服务缺少 library_status 工具。")
                response = await session.call_tool("library_status")
                _validate_mcp_library_status(response)


def _probe_mcp(project_root: Path, vault: Path) -> None:
    try:
        anyio, _, _, _ = _load_mcp_runtime()
        anyio.run(_probe_mcp_async, project_root, vault)
    except SelfTestError:
        raise
    except Exception as exc:
        raise SelfTestError(f"MCP stdio 自检失败：{exc}") from exc


def run_selftest(
    *,
    project_root: Path,
    vault: Path | None = None,
    import_probe: Callable[[], None] = _probe_imports,
    embedding_probe: Callable[[Path], int] = _probe_embedding,
    rapidocr_probe: Callable[[Path], None] = _probe_rapidocr,
    vision_probe: Callable[[Path], None] = _probe_vision,
    light_ocr_probe: Callable[[Path], None] = _probe_light_ocr,
    mcp_probe: Callable[[Path, Path], None] = _probe_mcp,
) -> SelfTestResult:
    selected_vault = vault if vault is not None else project_root / "Obsidian书库"
    try:
        import_probe()
        dimensions = embedding_probe(project_root / "data" / "models")
        if dimensions != 384:
            raise SelfTestError(f"语义模型维度错误：预期 384，实际 {dimensions}")
        rapidocr_probe(project_root / "data" / "ocr-models" / "rapidocr")
        vision_probe(project_root / "bin" / "book-vision-ocr")
        light_ocr_probe(project_root)
        mcp_probe(project_root, selected_vault)
    except SelfTestError:
        raise
    except Exception as exc:
        raise SelfTestError(f"安装自检失败：{exc}") from exc
    return SelfTestResult(embedding_dimensions=dimensions)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查离线书库安装运行时")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--vault", type=Path)
    return parser


def _absolute_vault_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_selftest(
            project_root=args.project_root.resolve(),
            vault=(
                _absolute_vault_without_symlink_resolution(args.vault)
                if args.vault is not None
                else None
            ),
        )
    except SelfTestError as exc:
        message = str(exc)
        if not message.startswith("安装自检失败"):
            message = f"安装自检失败：{message}"
        print(message, file=sys.stderr)
        return 1
    print(f"安装自检通过：语义模型维度 {result.embedding_dimensions}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
