import re
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from book_agent.config import MAX_EVIDENCE_TOKENS, MAX_FULL_PASSAGES, MAX_PREVIEWS
from book_agent.embeddings import decode_vector
from book_agent.models import RetrievalMode, SearchHit
from book_agent.storage import Database


_VALID_MODES = {"auto", "quote", "explain", "compare"}
_CANDIDATE_LIMIT = 20
_RRF_K = 60
MIN_SEMANTIC_SCORE = 0.20


def estimate_tokens(text: str) -> int:
    ascii_chars = sum(ord(char) <= 0x7F for char in text)
    bmp_non_ascii = sum(0x7F < ord(char) <= 0xFFFF for char in text)
    astral_chars = len(text) - ascii_chars - bmp_non_ascii
    return (ascii_chars + 3) // 4 + bmp_non_ascii + astral_chars * 2


def _text_fingerprint(text: str) -> str:
    return re.sub(r"\s+", "", text)


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

    def get_passages(
        self,
        passage_ids: Sequence[str],
        neighbor_count: int = 1,
    ) -> list[dict[str, object]]:
        if isinstance(passage_ids, (str, bytes)) or not isinstance(
            passage_ids, Sequence
        ):
            raise ValueError("passage_ids 必须是字符串 ID 的序列，不能是 str/bytes。")
        raw_ids = list(passage_ids)
        if not 1 <= len(raw_ids) <= MAX_FULL_PASSAGES:
            raise ValueError(f"passage_ids 必须包含 1 到 {MAX_FULL_PASSAGES} 个 ID。")
        if any(
            not isinstance(passage_id, str) or not passage_id.strip()
            for passage_id in raw_ids
        ):
            raise ValueError("passage_ids 中每个 ID 必须是非空白字符串。")
        if type(neighbor_count) is not int or neighbor_count not in (0, 1):
            raise ValueError("neighbor_count 必须是整数 0 或 1。")

        requested = list(dict.fromkeys(raw_ids))
        selected = self.database.get_passages(requested)
        available_by_id = {hit.passage_id: hit for hit in selected}
        missing = [passage_id for passage_id in requested if passage_id not in available_by_id]
        if missing:
            raise ValueError(
                "未知或当前不可检索的 passage_id：" + ", ".join(missing)
            )

        selected_by_id: dict[str, SearchHit] = {}
        selected_fingerprints: set[str] = set()
        for hit in selected:
            fingerprint = _text_fingerprint(hit.text)
            if fingerprint in selected_fingerprints:
                continue
            selected_fingerprints.add(fingerprint)
            selected_by_id[hit.passage_id] = hit

        selected_tokens = {
            passage_id: estimate_tokens(hit.text)
            for passage_id, hit in selected_by_id.items()
        }
        oversized = [
            passage_id
            for passage_id, token_count in selected_tokens.items()
            if token_count > MAX_EVIDENCE_TOKENS
        ]
        if oversized:
            raise ValueError(
                f"所选 passage {oversized[0]} 超过 {MAX_EVIDENCE_TOKENS} tokens；"
                "请拆成多次调用。"
            )
        selected_total = sum(selected_tokens.values())
        if selected_total > MAX_EVIDENCE_TOKENS:
            raise ValueError(
                f"所选 passage 总计超过 {MAX_EVIDENCE_TOKENS} tokens；"
                "请拆成多次调用。"
            )

        candidates = []
        seen_candidate_ids: set[str] = set()
        for hit in selected:
            ordinal = self.database.get_ordinal(hit.passage_id)
            if ordinal is None:
                raise ValueError(
                    f"未知或当前不可检索的 passage_id：{hit.passage_id}"
                )
            neighbors = self.database.get_neighbors(
                hit.book_id, ordinal, neighbor_count
            )
            if not any(
                candidate.passage_id == hit.passage_id for candidate in neighbors
            ):
                raise ValueError(
                    f"未知或当前不可检索的 passage_id：{hit.passage_id}"
                )
            for candidate in neighbors:
                if candidate.passage_id in seen_candidate_ids:
                    continue
                seen_candidate_ids.add(candidate.passage_id)
                candidates.append(candidate)

        included_ids = set(selected_by_id)
        included_fingerprints = set(selected_fingerprints)
        total_tokens = selected_total
        for candidate in candidates:
            if candidate.passage_id in included_ids:
                continue
            fingerprint = _text_fingerprint(candidate.text)
            if fingerprint in included_fingerprints:
                continue
            candidate_tokens = estimate_tokens(candidate.text)
            if (
                len(included_ids) >= MAX_FULL_PASSAGES
                or total_tokens + candidate_tokens > MAX_EVIDENCE_TOKENS
            ):
                continue
            included_ids.add(candidate.passage_id)
            included_fingerprints.add(fingerprint)
            total_tokens += candidate_tokens

        return [
            self._evidence_dict(candidate)
            for candidate in candidates
            if candidate.passage_id in included_ids
        ]

    @staticmethod
    def _evidence_dict(hit: SearchHit) -> dict[str, object]:
        location_parts: list[str] = []
        if hit.section:
            location_parts.append(hit.section)
        first_page = hit.page_start
        last_page = hit.page_end
        if first_page is not None or last_page is not None:
            first_page = first_page if first_page is not None else last_page
            last_page = last_page if last_page is not None else first_page
            if first_page == last_page:
                location_parts.append(f"PDF 页 {first_page}")
            else:
                location_parts.append(f"PDF 页 {first_page}–{last_page}")

        return {
            "passage_id": hit.passage_id,
            "book_id": hit.book_id,
            "title": hit.title,
            "text": hit.text,
            "section": hit.section,
            "page_start": hit.page_start,
            "page_end": hit.page_end,
            "page_label": hit.page_label,
            "location": " · ".join(location_parts) or hit.passage_id,
            "obsidian_link": f"[[{hit.markdown_path}#^{hit.anchor}]]",
            "untrusted_content": True,
            "estimated_tokens": estimate_tokens(hit.text),
        }

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
                if not np.isfinite(score) or score < MIN_SEMANTIC_SCORE:
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
