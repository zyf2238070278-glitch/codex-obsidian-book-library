from collections.abc import Sequence
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
    _CONFIG_NAMES = (
        "config.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
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
    def _is_complete_model(cls, path: Path) -> bool:
        if not cls._is_nonempty_file(path / "modules.json"):
            return False
        return any(cls._is_nonempty_file(path / name) for name in cls._CONFIG_NAMES)

    def _find_local_model(self) -> Path | None:
        if self._is_complete_model(self.cache_dir):
            return self.cache_dir

        hf_cache = self.cache_dir / self._HF_CACHE_NAME
        if self.cache_dir.name == self._HF_CACHE_NAME:
            hf_cache = self.cache_dir
        snapshots = hf_cache / "snapshots"
        try:
            candidates = sorted(snapshots.iterdir(), key=lambda path: path.name)
        except OSError:
            return None
        for candidate in candidates:
            if candidate.is_dir() and self._is_complete_model(candidate):
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
