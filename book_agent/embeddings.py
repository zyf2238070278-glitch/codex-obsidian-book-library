from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np


def encode_vector(vector: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(vector, dtype=np.float32)
    return contiguous.tobytes()


def decode_vector(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.float32).copy()


class NullEmbeddingProvider:
    available = False

    @staticmethod
    def _unavailable() -> RuntimeError:
        return RuntimeError("语义模型未启用，当前无法生成语义向量。")

    def embed_query(self, text: str) -> np.ndarray:
        raise self._unavailable()

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        raise self._unavailable()


class E5EmbeddingProvider:
    MODEL_NAME = "intfloat/multilingual-e5-small"

    _HF_CACHE_NAME = "models--intfloat--multilingual-e5-small"
    _WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
    _WEIGHT_INDEX_FILES = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    _TOKENIZER_FILES = (
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "vocab.txt",
    )

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._model: Any | None = None

    @staticmethod
    def _is_nonempty_file(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    @classmethod
    def _read_json(cls, path: Path) -> object | None:
        if not cls._is_nonempty_file(path):
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    @classmethod
    def _is_json_object(cls, path: Path) -> bool:
        return isinstance(cls._read_json(path), dict)

    @staticmethod
    def _safe_child(root: Path, relative: str) -> Path | None:
        candidate = Path(relative or ".")
        if candidate.is_absolute() or ".." in candidate.parts:
            return None
        return root / candidate

    @classmethod
    def _has_complete_weights(cls, transformer_path: Path) -> bool:
        if any(
            cls._is_nonempty_file(transformer_path / filename)
            for filename in cls._WEIGHT_FILES
        ):
            return True

        for filename in cls._WEIGHT_INDEX_FILES:
            index = cls._read_json(transformer_path / filename)
            if not isinstance(index, dict):
                continue
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                continue
            shard_paths: set[Path] = set()
            valid_index = True
            for shard in weight_map.values():
                if not isinstance(shard, str) or not shard:
                    valid_index = False
                    break
                shard_path = cls._safe_child(transformer_path, shard)
                if shard_path is None:
                    valid_index = False
                    break
                shard_paths.add(shard_path)
            if valid_index and shard_paths and all(
                cls._is_nonempty_file(shard_path) for shard_path in shard_paths
            ):
                return True
        return False

    @classmethod
    def _has_complete_tokenizer(cls, transformer_path: Path) -> bool:
        if not cls._is_json_object(transformer_path / "tokenizer_config.json"):
            return False
        if cls._is_json_object(transformer_path / "tokenizer.json"):
            return True
        return any(
            cls._is_nonempty_file(transformer_path / filename)
            for filename in cls._TOKENIZER_FILES
        )

    @classmethod
    def _is_complete_transformer(cls, transformer_path: Path) -> bool:
        return (
            cls._is_json_object(transformer_path / "config.json")
            and cls._has_complete_weights(transformer_path)
            and cls._has_complete_tokenizer(transformer_path)
        )

    @classmethod
    def _is_complete_model(cls, path: Path) -> bool:
        modules = cls._read_json(path / "modules.json")
        if not isinstance(modules, list) or not modules:
            return False

        found_transformer = False
        found_pooling = False
        for module in modules:
            if not isinstance(module, dict):
                return False
            module_type = module.get("type")
            module_relative_path = module.get("path")
            if not isinstance(module_type, str) or not isinstance(
                module_relative_path, str
            ):
                return False
            module_path = cls._safe_child(path, module_relative_path)
            if module_path is None:
                return False

            if module_type.endswith(".Transformer"):
                found_transformer = True
                if not cls._is_complete_transformer(module_path):
                    return False
            elif module_type.endswith(".Pooling"):
                found_pooling = True
                if not cls._is_json_object(module_path / "config.json"):
                    return False
            elif module_type.endswith(".Normalize"):
                continue
            elif module_relative_path and not cls._is_json_object(
                module_path / "config.json"
            ):
                return False

        return found_transformer and found_pooling

    @staticmethod
    def _is_directory(path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False

    @classmethod
    def _main_snapshot(cls, hf_cache: Path, snapshots: Path) -> Path | None:
        main_ref = hf_cache / "refs" / "main"
        if not cls._is_nonempty_file(main_ref):
            return None
        try:
            revision = main_ref.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        if not revision:
            return None
        revision_path = Path(revision)
        if revision_path.is_absolute() or len(revision_path.parts) != 1:
            return None
        candidate = snapshots / revision
        if cls._is_directory(candidate) and cls._is_complete_model(candidate):
            return candidate
        return None

    def _find_local_model(self) -> Path | None:
        if self._is_complete_model(self.cache_dir):
            return self.cache_dir

        hf_cache = self.cache_dir / self._HF_CACHE_NAME
        if self.cache_dir.name == self._HF_CACHE_NAME:
            hf_cache = self.cache_dir
        snapshots = hf_cache / "snapshots"

        main_snapshot = self._main_snapshot(hf_cache, snapshots)
        if main_snapshot is not None:
            return main_snapshot

        try:
            candidates = sorted(snapshots.iterdir(), key=lambda path: path.name)
        except OSError:
            return None
        for candidate in candidates:
            if self._is_directory(candidate) and self._is_complete_model(candidate):
                return candidate
        return None

    @property
    def available(self) -> bool:
        return self._find_local_model() is not None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        model_path = self._find_local_model()
        if model_path is None:
            raise RuntimeError(
                "未找到完整的本地 E5 模型缓存；语义检索保持关闭，且不会尝试联网下载。"
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少可选依赖 sentence-transformers，无法加载本地 E5 模型。"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "导入 sentence-transformers 依赖失败，无法加载本地 E5 模型。"
            ) from exc

        try:
            self._model = SentenceTransformer(
                str(model_path),
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(f"加载本地 E5 模型失败：{exc}") from exc
        return self._model

    def embed_query(self, text: str) -> np.ndarray:
        encoded = self._load().encode(
            f"query: {text}",
            normalize_embeddings=True,
        )
        vector = np.asarray(encoded, dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(vector, dtype=np.float32)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        materialized = list(texts)
        if not materialized:
            return np.empty((0, 0), dtype=np.float32)

        encoded = self._load().encode(
            [f"passage: {text}" for text in materialized],
            normalize_embeddings=True,
        )
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim == 1 and len(materialized) == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] != len(materialized):
            raise RuntimeError("E5 模型返回了无效的段落向量形状。")
        return np.ascontiguousarray(matrix, dtype=np.float32)
