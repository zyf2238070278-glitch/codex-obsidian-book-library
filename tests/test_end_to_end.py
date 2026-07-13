import json
from pathlib import Path

from book_agent.config import MAX_EVIDENCE_TOKENS, MAX_FULL_PASSAGES, MAX_PREVIEWS
from book_agent.tools import build_tools
from fakes import DeterministicEmbeddingProvider


EXACT_PHRASE = "蓝杉钟声只在清晨响起"
SEMANTIC_TEXT = "晶圆厂扩建通常需要多年，订单突然增加时，新产能无法立刻响应。"
UNRELATED_TEXT = "品牌广告可以帮助消费者识别新产品。"
SEMANTIC_QUERY = "为什么市场突然想买更多商品时容易出现供给短缺"


def _json_round_trip(payload: object) -> object:
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def test_codex_tool_boundary_runs_the_complete_grounded_reading_workflow(
    tmp_path: Path,
) -> None:
    source = tmp_path / "合成测试书.md"
    source.write_text(
        "# 供给周期\n\n"
        f"{SEMANTIC_TEXT}\n\n"
        "# 校验短语\n\n"
        f"{EXACT_PHRASE}。\n\n"
        "# 传播\n\n"
        f"{UNRELATED_TEXT}\n",
        encoding="utf-8",
    )
    provider = DeterministicEmbeddingProvider(
        {
            SEMANTIC_TEXT: [1.0, 0.0],
            f"\n\n{EXACT_PHRASE}。": [0.0, 1.0],
            f"\n\n{UNRELATED_TEXT}": [0.0, -1.0],
            SEMANTIC_QUERY: [1.0, 0.0],
        }
    )
    tools = build_tools(tmp_path / "library", embedding_provider=provider)

    imported = tools.import_book(
        file_path=str(source.absolute()),
        title="供给周期测试书",
        author="本地测试作者",
    )
    assert imported["ok"] is True
    assert imported["status"] == "ready"
    assert imported["passage_count"] == 3

    quoted = tools.search_books(EXACT_PHRASE, mode="quote", limit=2)
    assert quoted["ok"] is True
    assert quoted["count"] == 1
    assert EXACT_PHRASE in quoted["results"][0]["preview"]
    assert quoted["results"][0]["untrusted_content"] is True
    assert "text" not in quoted["results"][0]

    assert tools.database.keyword_search(SEMANTIC_QUERY, 20) == []
    paraphrased = tools.search_books(SEMANTIC_QUERY, mode="explain", limit=2)
    assert paraphrased["ok"] is True
    assert paraphrased["count"] == 1
    selected = paraphrased["results"][0]
    assert SEMANTIC_TEXT in selected["preview"]
    assert selected["section"] == "供给周期"
    assert selected["untrusted_content"] is True

    expanded = tools.get_passages([selected["passage_id"]], neighbor_count=0)
    assert expanded["ok"] is True
    assert len(expanded["evidence"]) == 1
    evidence = expanded["evidence"][0]
    assert evidence["passage_id"] == selected["passage_id"]
    assert evidence["title"] == "供给周期测试书"
    assert evidence["location"] == "供给周期"
    assert evidence["text"] == SEMANTIC_TEXT
    assert evidence["untrusted_content"] is True

    saved = tools.save_reading_note(
        "供给周期通俗解释",
        "扩产很慢，所以突然增加的购买需求可能暂时得不到满足。",
        [evidence["passage_id"]],
    )
    assert saved["ok"] is True
    note = Path(saved["path"])
    note_text = note.read_text(encoding="utf-8")
    assert note.parent.name == "30-AI读书笔记"
    assert "index_for_evidence: false" in note_text
    assert "《供给周期测试书》：供给周期" in note_text
    assert f"#^{evidence['passage_id']}]]" in note_text

    assert quoted["count"] <= MAX_PREVIEWS
    assert paraphrased["count"] <= MAX_PREVIEWS
    assert len(expanded["evidence"]) <= MAX_FULL_PASSAGES
    assert sum(item["estimated_tokens"] for item in expanded["evidence"]) <= (
        MAX_EVIDENCE_TOKENS
    )
    for payload in (imported, quoted, paraphrased, expanded, saved):
        assert _json_round_trip(payload) == payload
