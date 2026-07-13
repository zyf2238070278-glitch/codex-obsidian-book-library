from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from book_agent.config import MAX_PREVIEWS
from book_agent.embeddings import decode_vector
from book_agent.models import RetrievalMode, SearchHit
from book_agent.storage import Database


_VALID_MODES = {"auto", "quote", "explain", "compare"}
_CANDIDATE_LIMIT = 20
_RRF_K = 60


class Retriever:
    def __init__(self, database: Database, embedding_provider: object) -> None:
        self.database = database
        self.embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        mode: RetrievalMode = "auto",
        book_ids: Sequence[str] | None = None,
        limit: int = MAX_PREVIEWS,
    ) -> list[SearchHit]:
        normalized = query.strip()
        if not normalized:
            raise ValueError("查询内容不能为空。")
        if mode not in _VALID_MODES:
            raise ValueError(f"不支持的检索模式：{mode}")
        if book_ids is not None and not book_ids:
            return []

        safe_limit = max(1, min(int(limit), MAX_PREVIEWS))
        keyword_hits = self.database.keyword_search(
            normalized,
            _CANDIDATE_LIMIT,
            book_ids,
        )
        if mode == "quote" and keyword_hits:
            return keyword_hits[:safe_limit]

        semantic_hits = self._semantic_search(normalized, book_ids)
        if mode == "quote":
            return semantic_hits[:safe_limit]

        return self._rrf(keyword_hits, semantic_hits)[:safe_limit]

    def _semantic_search(
        self,
        query: str,
        book_ids: Sequence[str] | None,
    ) -> list[SearchHit]:
        try:
            if not self.embedding_provider.available:  # type: ignore[attr-defined]
                return []
            query_vector = np.asarray(
                self.embedding_provider.embed_query(query),  # type: ignore[attr-defined]
                dtype=np.float32,
            )
            if (
                query_vector.ndim != 1
                or query_vector.size == 0
                or not np.all(np.isfinite(query_vector))
            ):
                return []
            query64 = query_vector.astype(np.float64, copy=False)
            query_norm = float(np.linalg.norm(query64))
            if not np.isfinite(query_norm) or query_norm <= 0.0:
                return []
        except Exception:
            return []

        ranked: list[SearchHit] = []
        for hit, payload in self.database.iter_embeddings(book_ids):
            try:
                passage_vector = decode_vector(payload)
                if (
                    passage_vector.ndim != 1
                    or passage_vector.shape != query_vector.shape
                    or not np.all(np.isfinite(passage_vector))
                ):
                    continue
                passage64 = passage_vector.astype(np.float64, copy=False)
                passage_norm = float(np.linalg.norm(passage64))
                if not np.isfinite(passage_norm) or passage_norm <= 0.0:
                    continue
                score = float(
                    np.dot(query64, passage64) / (query_norm * passage_norm)
                )
                if not np.isfinite(score):
                    continue
            except Exception:
                continue
            ranked.append(replace(hit, score=score))

        ranked.sort(key=lambda hit: (-hit.score, hit.passage_id))
        return ranked[:_CANDIDATE_LIMIT]

    @staticmethod
    def _rrf(
        keyword_hits: Sequence[SearchHit],
        semantic_hits: Sequence[SearchHit],
    ) -> list[SearchHit]:
        metadata: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for ranking in (keyword_hits, semantic_hits):
            seen: set[str] = set()
            for rank, hit in enumerate(ranking, start=1):
                if hit.passage_id in seen:
                    continue
                seen.add(hit.passage_id)
                metadata.setdefault(hit.passage_id, hit)
                scores[hit.passage_id] = scores.get(hit.passage_id, 0.0) + 1 / (
                    _RRF_K + rank
                )

        fused = [
            replace(metadata[passage_id], score=score)
            for passage_id, score in scores.items()
        ]
        fused.sort(key=lambda hit: (-hit.score, hit.passage_id))
        return fused
