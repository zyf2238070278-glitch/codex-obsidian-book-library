from collections.abc import Sequence

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
