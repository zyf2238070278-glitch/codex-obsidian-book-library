import builtins
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from book_agent.embeddings import E5EmbeddingProvider


def _modules() -> list[dict[str, object]]:
    return [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
        {
            "idx": 2,
            "name": "2",
            "path": "2_Normalize",
            "type": "sentence_transformers.models.Normalize",
        },
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _complete_model(path: Path, *, sharded: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "modules.json", _modules())
    _write_json(path / "config.json", {"model_type": "bert"})
    _write_json(path / "tokenizer_config.json", {"model_max_length": 512})
    _write_json(path / "tokenizer.json", {"version": "1.0"})
    _write_json(path / "1_Pooling" / "config.json", {"word_embedding_dimension": 384})
    if sharded:
        _write_json(
            path / "model.safetensors.index.json",
            {
                "weight_map": {
                    "encoder.layer.0": "model-00001-of-00002.safetensors",
                    "encoder.layer.1": "model-00002-of-00002.safetensors",
                }
            },
        )
        (path / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
        (path / "model-00002-of-00002.safetensors").write_bytes(b"shard-two")
    else:
        (path / "model.safetensors").write_bytes(b"weights")
    return path


def _two_file_partial(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "modules.json", _modules())
    _write_json(path / "config.json", {"model_type": "bert"})
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


def test_two_file_model_marker_is_not_a_complete_local_model(tmp_path: Path) -> None:
    _two_file_partial(tmp_path)

    assert E5EmbeddingProvider(tmp_path).available is False


@pytest.mark.parametrize(
    "missing",
    [
        "invalid-modules-json",
        "weight",
        "tokenizer-config",
        "tokenizer-artifact",
        "pooling-config",
    ],
)
def test_required_sentence_transformer_assets_must_be_complete(
    tmp_path: Path, missing: str
) -> None:
    _complete_model(tmp_path)
    if missing == "invalid-modules-json":
        (tmp_path / "modules.json").write_text("{broken", encoding="utf-8")
    elif missing == "weight":
        (tmp_path / "model.safetensors").unlink()
    elif missing == "tokenizer-config":
        (tmp_path / "tokenizer_config.json").unlink()
    elif missing == "tokenizer-artifact":
        (tmp_path / "tokenizer.json").unlink()
    elif missing == "pooling-config":
        (tmp_path / "1_Pooling" / "config.json").unlink()

    assert E5EmbeddingProvider(tmp_path).available is False


def test_sharded_weights_require_every_indexed_shard(tmp_path: Path) -> None:
    _complete_model(tmp_path, sharded=True)
    (tmp_path / "model-00002-of-00002.safetensors").unlink()

    assert E5EmbeddingProvider(tmp_path).available is False


def test_load_prefers_the_valid_refs_main_snapshot_and_uses_e5_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_cache = tmp_path / "models--intfloat--multilingual-e5-small"
    snapshots = model_cache / "snapshots"
    _two_file_partial(snapshots / "a-old-partial")
    selected = _complete_model(snapshots / "z-main-revision")
    (model_cache / "refs").mkdir(parents=True)
    (model_cache / "refs" / "main").write_text("z-main-revision\n", encoding="utf-8")
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


def test_invalid_refs_main_falls_back_to_another_complete_snapshot(
    tmp_path: Path,
) -> None:
    model_cache = tmp_path / "models--intfloat--multilingual-e5-small"
    snapshots = model_cache / "snapshots"
    _two_file_partial(snapshots / "a-main-partial")
    selected = _complete_model(snapshots / "z-valid-fallback")
    (model_cache / "refs").mkdir(parents=True)
    (model_cache / "refs" / "main").write_text("a-main-partial", encoding="utf-8")

    assert E5EmbeddingProvider(model_cache)._find_local_model() == selected


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
