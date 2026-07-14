from collections.abc import Mapping, Sequence

import numpy as np


class DeterministicEmbeddingProvider:
    """Small deterministic embedding fake shared by retrieval tests."""

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float] | np.ndarray],
        *,
        available: bool = True,
    ) -> None:
        self.available = available
        self.vectors = {
            text: np.asarray(vector, dtype=np.float32) for text, vector in vectors.items()
        }
        self.query_calls: list[str] = []
        self.passage_calls: list[list[str]] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        return self.vectors[text].copy()

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        materialized = list(texts)
        self.passage_calls.append(materialized)
        return np.asarray([self.vectors[text] for text in materialized], dtype=np.float32)
