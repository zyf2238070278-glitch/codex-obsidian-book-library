from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import anyio
import numpy as np
import pytest

from installer import runtime_selftest
from book_agent.ocr.rapid import REQUIRED_MODEL_FILES


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _passing_probes(events: list[str]):
    def imports() -> None:
        events.append("imports")

    def embedding(path: Path) -> int:
        events.append(f"embedding:{path}")
        return 384

    def rapidocr(path: Path) -> None:
        events.append(f"rapidocr:{path}")

    def vision(path: Path) -> None:
        events.append(f"vision:{path}")

    def light_ocr(path: Path) -> None:
        events.append(f"light_ocr:{path}")

    def mcp(path: Path, vault: Path) -> None:
        events.append(f"mcp:{path}:{vault}")

    return imports, embedding, rapidocr, vision, light_ocr, mcp


def test_run_selftest_calls_probes_in_exact_order_and_returns_dimensions(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    imports, embedding, rapidocr, vision, light_ocr, mcp = _passing_probes(events)

    result = runtime_selftest.run_selftest(
        project_root=tmp_path,
        vault=tmp_path / "Custom Vault",
        import_probe=imports,
        embedding_probe=embedding,
        rapidocr_probe=rapidocr,
        vision_probe=vision,
        light_ocr_probe=light_ocr,
        mcp_probe=mcp,
    )

    assert result == runtime_selftest.SelfTestResult(embedding_dimensions=384)
    assert events == [
        "imports",
        f"embedding:{tmp_path / 'data' / 'models'}",
        f"rapidocr:{tmp_path / 'data' / 'ocr-models' / 'rapidocr'}",
        f"vision:{tmp_path / 'bin' / 'book-vision-ocr'}",
        f"light_ocr:{tmp_path}",
        f"mcp:{tmp_path}:{tmp_path / 'Custom Vault'}",
    ]


def test_run_selftest_rejects_wrong_embedding_dimension_without_later_probes(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    imports, _, rapidocr, vision, light_ocr, mcp = _passing_probes(events)

    with pytest.raises(
        runtime_selftest.SelfTestError,
        match="^语义模型维度错误：预期 384，实际 768$",
    ):
        runtime_selftest.run_selftest(
            project_root=tmp_path,
            vault=tmp_path / "Vault",
            import_probe=imports,
            embedding_probe=lambda path: events.append(f"embedding:{path}") or 768,
            rapidocr_probe=rapidocr,
            vision_probe=vision,
            light_ocr_probe=light_ocr,
            mcp_probe=mcp,
        )

    assert events == ["imports", f"embedding:{tmp_path / 'data' / 'models'}"]


def test_run_selftest_defaults_mcp_to_project_obsidian_vault(tmp_path: Path) -> None:
    observed: list[tuple[Path, Path]] = []

    runtime_selftest.run_selftest(
        project_root=tmp_path,
        import_probe=lambda: None,
        embedding_probe=lambda _: 384,
        rapidocr_probe=lambda _: None,
        vision_probe=lambda _: None,
        light_ocr_probe=lambda _: None,
        mcp_probe=lambda root, vault: observed.append((root, vault)),
    )

    assert observed == [(tmp_path, tmp_path / "Obsidian书库")]


@pytest.mark.parametrize("failed_index", range(6))
def test_run_selftest_maps_each_probe_exception_and_stops(
    tmp_path: Path, failed_index: int
) -> None:
    events: list[str] = []

    def probe(index: int, result: object = None):
        def call(*_: object) -> object:
            events.append(str(index))
            if index == failed_index:
                raise ValueError(f"probe {index} broke")
            return result

        return call

    with pytest.raises(
        runtime_selftest.SelfTestError,
        match=rf"^安装自检失败：probe {failed_index} broke$",
    ):
        runtime_selftest.run_selftest(
            project_root=tmp_path,
            import_probe=probe(0),
            embedding_probe=probe(1, 384),
            rapidocr_probe=probe(2),
            vision_probe=probe(3),
            light_ocr_probe=probe(4),
            mcp_probe=probe(5),
        )

    assert events == [str(index) for index in range(failed_index + 1)]


def test_run_selftest_preserves_selftest_errors(tmp_path: Path) -> None:
    expected = runtime_selftest.SelfTestError("具体错误")

    def fail() -> None:
        raise expected

    with pytest.raises(runtime_selftest.SelfTestError) as captured:
        runtime_selftest.run_selftest(project_root=tmp_path, import_probe=fail)

    assert captured.value is expected


def test_embedding_probe_forces_offline_mode_and_requires_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    class Provider:
        def __init__(self, root: Path) -> None:
            observed["root"] = root

        @property
        def available(self) -> bool:
            observed["offline"] = (
                os.environ.get("HF_HUB_OFFLINE"),
                os.environ.get("TRANSFORMERS_OFFLINE"),
            )
            return True

        def embed_query(self, text: str) -> np.ndarray:
            observed["text"] = text
            return np.zeros(384, dtype=np.float32)

    monkeypatch.setattr(runtime_selftest, "_embedding_provider_class", lambda: Provider)

    assert runtime_selftest._probe_embedding(tmp_path) == 384
    assert observed == {
        "root": tmp_path,
        "offline": ("1", "1"),
        "text": "安装自检",
    }


def test_embedding_probe_rejects_unavailable_or_non_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Unavailable:
        def __init__(self, _: Path) -> None:
            pass

        available = False

    monkeypatch.setattr(
        runtime_selftest, "_embedding_provider_class", lambda: Unavailable
    )
    with pytest.raises(runtime_selftest.SelfTestError, match="语义模型不可用"):
        runtime_selftest._probe_embedding(tmp_path)

    class Matrix(Unavailable):
        available = True

        def embed_query(self, _: str) -> np.ndarray:
            return np.zeros((1, 384), dtype=np.float32)

    monkeypatch.setattr(runtime_selftest, "_embedding_provider_class", lambda: Matrix)
    with pytest.raises(runtime_selftest.SelfTestError, match="一维向量"):
        runtime_selftest._probe_embedding(tmp_path)


def test_rapidocr_probe_requires_nonempty_regular_model_files(tmp_path: Path) -> None:
    root = tmp_path / "rapidocr"
    root.mkdir()
    for filename in REQUIRED_MODEL_FILES:
        (root / filename).write_bytes(b"model")

    runtime_selftest._probe_rapidocr(root)

    (root / REQUIRED_MODEL_FILES[0]).write_bytes(b"")
    with pytest.raises(runtime_selftest.SelfTestError, match="RapidOCR 模型缺失或为空"):
        runtime_selftest._probe_rapidocr(root)


def test_vision_probe_runs_exact_capabilities_argv_and_validates_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = tmp_path / "book-vision-ocr"
    helper.write_bytes(b"helper")
    helper.chmod(0o755)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps({"schema_version": 2, "languages": ["zh-Hans", "en-US"]}),
            "",
        )

    monkeypatch.setattr(runtime_selftest.subprocess, "run", run)
    runtime_selftest._probe_vision(helper)

    assert calls == [
        (
            [str(helper), "--capabilities"],
            {"capture_output": True, "text": True, "check": False, "timeout": 15},
        )
    ]


@pytest.mark.parametrize(
    "completed",
    [
        subprocess.CompletedProcess(["helper"], 4, "", ""),
        subprocess.CompletedProcess(["helper"], 0, "{}", "warning"),
        subprocess.CompletedProcess(["helper"], 0, "not-json", ""),
        subprocess.CompletedProcess(
            ["helper"], 0, '{"schema_version":1,"languages":["zh-Hans","en-US"]}', ""
        ),
        subprocess.CompletedProcess(
            ["helper"], 0, '{"schema_version":2,"languages":["zh-Hans"]}', ""
        ),
    ],
)
def test_vision_probe_rejects_invalid_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed: subprocess.CompletedProcess[str],
) -> None:
    helper = tmp_path / "book-vision-ocr"
    helper.write_bytes(b"helper")
    helper.chmod(0o755)
    monkeypatch.setattr(runtime_selftest.subprocess, "run", lambda *a, **k: completed)

    with pytest.raises(runtime_selftest.SelfTestError, match="Vision OCR"):
        runtime_selftest._probe_vision(helper)


def test_light_ocr_probe_starts_worker_and_closes_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = tmp_path / "runtime" / "node" / "bin" / "node"
    worker = tmp_path / "scripts" / "light_ocr_worker.mjs"
    node.parent.mkdir(parents=True)
    worker.parent.mkdir(parents=True)
    node.write_bytes(b"node")
    node.chmod(0o755)
    worker.write_text("// worker", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(runtime_selftest.subprocess, "run", run)

    runtime_selftest._probe_light_ocr(tmp_path)

    assert calls == [
        (
            [str(node), str(worker)],
            {
                "cwd": tmp_path,
                "input": '{"op":"close"}\n',
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": 120,
            },
        )
    ]


@pytest.mark.parametrize(
    "completed",
    [
        subprocess.CompletedProcess(["node"], 7, "", "engine failed"),
        subprocess.CompletedProcess(["node"], 0, "unexpected", ""),
    ],
)
def test_light_ocr_probe_rejects_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed: subprocess.CompletedProcess[str],
) -> None:
    node = tmp_path / "runtime" / "node" / "bin" / "node"
    worker = tmp_path / "scripts" / "light_ocr_worker.mjs"
    node.parent.mkdir(parents=True)
    worker.parent.mkdir(parents=True)
    node.write_bytes(b"node")
    node.chmod(0o755)
    worker.write_text("// worker", encoding="utf-8")
    monkeypatch.setattr(runtime_selftest.subprocess, "run", lambda *a, **k: completed)

    with pytest.raises(runtime_selftest.SelfTestError, match="Light OCR"):
        runtime_selftest._probe_light_ocr(tmp_path)


def test_mcp_probe_calls_library_status_with_actual_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    events: list[str] = []

    @asynccontextmanager
    async def fake_stdio(parameters: object):
        captured["parameters"] = parameters
        yield object(), object()

    class Session:
        def __init__(self, read_stream: object, write_stream: object) -> None:
            captured["streams"] = (read_stream, write_stream)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            events.append("closed")

        async def initialize(self) -> None:
            events.append("initialize")

        async def list_tools(self) -> object:
            events.append("list_tools")
            return SimpleNamespace(tools=[SimpleNamespace(name="library_status")])

        async def call_tool(self, name: str) -> object:
            events.append(f"call_tool:{name}")
            return SimpleNamespace(
                isError=False,
                error=None,
                structuredContent={"ok": True},
                content=[],
            )

    monkeypatch.setenv("SELFTEST_SENTINEL", "preserved")
    from mcp import StdioServerParameters

    monkeypatch.setattr(
        runtime_selftest,
        "_load_mcp_runtime",
        lambda: (anyio, Session, StdioServerParameters, fake_stdio),
    )

    vault = tmp_path / "Custom Vault"
    vault.mkdir()
    anyio.run(runtime_selftest._probe_mcp_async, tmp_path, vault)

    parameters = captured["parameters"]
    assert parameters.command == sys.executable
    assert parameters.args == ["-m", "book_agent.mcp_server"]
    assert parameters.cwd == str(tmp_path)
    assert parameters.env["BOOK_LIBRARY_ROOT"] == str(tmp_path)
    assert parameters.env["BOOK_LIBRARY_OBSIDIAN_VAULT"] == str(vault)
    assert parameters.env["HF_HUB_OFFLINE"] == "1"
    assert parameters.env["TRANSFORMERS_OFFLINE"] == "1"
    assert parameters.env["SELFTEST_SENTINEL"] == "preserved"
    assert not (tmp_path / "Obsidian书库").exists()
    assert events == [
        "initialize",
        "list_tools",
        "call_tool:library_status",
        "closed",
    ]


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            isError=True,
            error=None,
            structuredContent=None,
            content=[SimpleNamespace(type="text", text="database error")],
        ),
        SimpleNamespace(
            isError=False,
            error={"message": "handler failed"},
            structuredContent={"ok": True},
            content=[],
        ),
        SimpleNamespace(
            isError=False,
            error=None,
            structuredContent={"ok": False, "error": "database failed"},
            content=[],
        ),
        SimpleNamespace(
            isError=False,
            error=None,
            structuredContent=["not", "an", "object"],
            content=[],
        ),
        SimpleNamespace(
            isError=False,
            error=None,
            structuredContent=None,
            content=[SimpleNamespace(type="text", text="not-json")],
        ),
    ],
)
def test_mcp_library_status_rejects_transport_business_and_malformed_errors(
    response: object,
) -> None:
    with pytest.raises(runtime_selftest.SelfTestError, match="MCP.*library_status"):
        runtime_selftest._validate_mcp_library_status(response)


def test_mcp_library_status_accepts_json_text_business_result() -> None:
    response = SimpleNamespace(
        isError=False,
        error=None,
        structuredContent=None,
        content=[SimpleNamespace(type="text", text='{"ok": true}')],
    )

    runtime_selftest._validate_mcp_library_status(response)


def test_real_mcp_rejects_symlink_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_vault = tmp_path / "Real Vault"
    real_vault.mkdir()
    symlink_vault = tmp_path / "Linked Vault"
    symlink_vault.symlink_to(real_vault, target_is_directory=True)
    monkeypatch.setenv("PYTHONPATH", str(PROJECT_ROOT))

    with pytest.raises(runtime_selftest.SelfTestError, match="MCP stdio"):
        runtime_selftest._probe_mcp(tmp_path, symlink_vault)


def test_cli_success_prints_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        runtime_selftest,
        "run_selftest",
        lambda **_: runtime_selftest.SelfTestResult(embedding_dimensions=384),
    )

    assert runtime_selftest.main(
        ["--project-root", str(tmp_path), "--vault", str(tmp_path / "Vault")]
    ) == 0
    captured = capsys.readouterr()
    assert "384" in captured.out
    assert captured.err == ""


def test_cli_failure_is_chinese_stderr_and_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(**_: object) -> runtime_selftest.SelfTestResult:
        raise runtime_selftest.SelfTestError("安装自检失败：fixture")

    monkeypatch.setattr(runtime_selftest, "run_selftest", fail)

    assert runtime_selftest.main(["--project-root", str(tmp_path)]) != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "安装自检失败：fixture" in captured.err


@pytest.mark.parametrize("blocked_module", ["fitz", "mcp", "numpy"])
def test_module_cli_reports_missing_checked_dependency_without_traceback(
    tmp_path: Path, blocked_module: str
) -> None:
    (tmp_path / "sitecustomize.py").write_text(
        f"""import sys
class _BlockedModuleFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {blocked_module!r} or fullname.startswith({(blocked_module + '.')!r}):
            raise ImportError("controlled missing {blocked_module}")
        return None
sys.meta_path.insert(0, _BlockedModuleFinder())
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "installer.runtime_selftest",
            "--project-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(PROJECT_ROOT))),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "安装自检失败" in completed.stderr
    assert completed.stderr.count("安装自检失败") == 1
    assert f"controlled missing {blocked_module}" in completed.stderr
    assert "Traceback" not in completed.stderr
