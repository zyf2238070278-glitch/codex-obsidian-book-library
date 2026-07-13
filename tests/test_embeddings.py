import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from book_agent.embeddings import E5EmbeddingProvider


def _complete_model(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "modules.json").write_text("[]", encoding="utf-8")
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


@pytest.mark.parametrize("layout", ["empty", "unrelated", "partial", "hf-partial"])
def test_incomplete_caches_are_unavailable_without_importing_sentence_transformers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    if layout == "unrelated":
        (tmp_path / "notes.txt").write_text("not a model", encoding="utf-8")
    elif layout == "partial":
        (tmp_path / "modules.json").write_text("[]", encoding="utf-8")
    elif layout == "hf-partial":
        partial = (
            tmp_path
            / "models--intfloat--multilingual-e5-small"
            / "snapshots"
            / "partial-rev"
        )
        partial.mkdir(parents=True)
        (partial / "modules.json").write_text("[]", encoding="utf-8")
        (partial / "config.json").write_text("", encoding="utf-8")

    attempted_imports: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            attempted_imports.append(name)
            raise AssertionError("cache detection must not import sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    provider = E5EmbeddingProvider(tmp_path)

    assert provider.available is False
    with pytest.raises(RuntimeError, match="本地|模型|cache"):
        provider.embed_query("查询")
    assert attempted_imports == []


def test_available_recognizes_a_direct_complete_model_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _complete_model(tmp_path)
    attempted_imports: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            attempted_imports.append(name)
            raise AssertionError("available must be an import-free filesystem check")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    provider = E5EmbeddingProvider(tmp_path)

    assert provider.MODEL_NAME == "intfloat/multilingual-e5-small"
    assert provider.available is True
    assert provider._model is None
    assert attempted_imports == []


def test_load_uses_the_first_complete_sorted_hf_snapshot_and_e5_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = (
        tmp_path
        / "models--intfloat--multilingual-e5-small"
        / "snapshots"
    )
    selected = _complete_model(snapshots / "a-revision")
    _complete_model(snapshots / "z-revision")
    constructed: list[tuple[object, dict[str, object]]] = []
    encoded: list[tuple[object, bool]] = []

    class FakeSentenceTransformer:
        def __init__(self, source, **kwargs) -> None:
            constructed.append((source, kwargs))

        def encode(self, inputs, *, normalize_embeddings: bool):
            encoded.append((inputs, normalize_embeddings))
            if isinstance(inputs, str):
                return np.arange(6, dtype=np.float64)[::2]
            return np.arange(12, dtype=np.float64).reshape(2, 6)[:, ::2]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    provider = E5EmbeddingProvider(tmp_path)

    query = provider.embed_query("库存为什么波动")
    passages = provider.embed_passages(["第一段", "第二段"])

    assert len(constructed) == 1
    assert Path(constructed[0][0]) == selected
    assert constructed[0][1]["local_files_only"] is True
    assert encoded == [
        ("query: 库存为什么波动", True),
        (["passage: 第一段", "passage: 第二段"], True),
    ]
    assert query.dtype == np.float32
    assert query.shape == (3,)
    assert query.flags.c_contiguous
    assert passages.dtype == np.float32
    assert passages.shape == (2, 3)
    assert passages.flags.c_contiguous


def test_complete_cache_reports_a_clear_missing_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _complete_model(tmp_path)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    real_import = builtins.__import__

    def missing_dependency(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ModuleNotFoundError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_dependency)

    with pytest.raises(RuntimeError, match="sentence-transformers|依赖"):
        E5EmbeddingProvider(tmp_path).embed_query("查询")


def test_local_model_load_failures_are_wrapped_as_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _complete_model(tmp_path)

    class BrokenSentenceTransformer:
        def __init__(self, source, **kwargs) -> None:
            raise OSError("bad local files")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = BrokenSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.raises(RuntimeError, match="加载|load"):
        E5EmbeddingProvider(tmp_path).embed_query("查询")
