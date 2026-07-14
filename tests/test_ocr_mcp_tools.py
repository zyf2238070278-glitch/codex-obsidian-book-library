import json
from dataclasses import dataclass

import pytest

from book_agent.tools import LibraryTools


@dataclass(frozen=True)
class FakeSummary:
    book_id: str
    title: str = "合成书"
    status: str = "queued"
    total_pages: int = 2
    completed_pages: int = 0
    current_page: int | None = None
    queue_position: int | None = 1
    updated_at: str | None = None
    error: str | None = None
    estimated_remaining_seconds: int | None = None

    @property
    def percent_complete(self) -> float:
        return 0.0


class FakeOcr:
    def start_ocr(self, book_id: str) -> FakeSummary:
        return FakeSummary(book_id)

    def start_pending_ocr(self, *, limit: int = 25, offset: int = 0) -> dict[str, object]:
        return {"count": 1, "jobs": [FakeSummary("a" * 24)], "limit": limit, "offset": offset}

    def status(self, book_id: str | None = None, *, limit: int = 20, offset: int = 0) -> FakeSummary | dict[str, object]:
        if book_id:
            return FakeSummary(book_id)
        return {"count": 0, "jobs": [], "limit": limit, "offset": offset}

    def pause(self, book_id: str) -> FakeSummary:
        return FakeSummary(book_id, status="paused")


def _tools(fake: FakeOcr) -> LibraryTools:
    return LibraryTools(
        paths=object(), database=object(), importer=object(), retriever=object(),
        notes=object(), embedding_provider=object(), ocr_service=fake,
    )


def test_ocr_tools_return_bounded_json_metadata_without_text() -> None:
    tools = _tools(FakeOcr())
    result = tools.start_ocr("a" * 24)
    assert result["ok"] is True
    assert result["book_id"] == "a" * 24
    assert "text" not in result
    json.dumps(result, ensure_ascii=False, allow_nan=False)

    pending = tools.start_pending_ocr(limit=3, offset=2)
    assert pending["ok"] is True
    assert pending["limit"] == 3
    assert pending["jobs"][0]["book_id"] == "a" * 24

    status = tools.ocr_status()
    assert status == {"ok": True, "count": 0, "jobs": [], "limit": 20, "offset": 0}

    paused = tools.pause_ocr("a" * 24)
    assert paused["ok"] is True
    assert paused["status"] == "paused"


@pytest.mark.parametrize(
    "call",
    [
        lambda tools: tools.start_ocr(True),
        lambda tools: tools.start_pending_ocr(limit=True),
        lambda tools: tools.start_pending_ocr(limit=float("nan")),
        lambda tools: tools.ocr_status(offset=-1),
        lambda tools: tools.pause_ocr(" "),
    ],
)
def test_ocr_tool_invalid_inputs_are_json_errors(call) -> None:
    result = call(_tools(FakeOcr()))
    assert result["ok"] is False
    json.dumps(result, ensure_ascii=False, allow_nan=False)
