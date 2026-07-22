from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from book_agent.ocr.light import LightOcrEngine, LightOcrError


class _Process:
    def __init__(self, responses: list[dict[str, object] | str]) -> None:
        encoded = "".join(
            (item if isinstance(item, str) else json.dumps(item)) + "\n"
            for item in responses
        )
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(encoded)
        self.stderr = io.StringIO()
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    path.write_bytes(b"png fixture")
    return path


def _engine(process: _Process) -> LightOcrEngine:
    return LightOcrEngine(
        node=Path("/fake/node"),
        worker=Path("/fake/light_ocr_worker.mjs"),
        process_factory=lambda *args, **kwargs: process,
        response_reader=lambda stream, timeout: stream.readline(),
        request_id_factory=lambda: "request-1",
    )


def test_light_adapter_returns_engine_neutral_lines(tmp_path: Path) -> None:
    process = _Process(
        [
            {
                "id": "request-1",
                "ok": True,
                "lines": [
                    {
                        "text": "测试文字",
                        "confidence": 0.93,
                        "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1},
                    }
                ],
            }
        ]
    )
    engine = _engine(process)

    result = engine.recognize_image(_image(tmp_path))

    assert result.engine == "light_ocr"
    assert result.ordered_text() == "测试文字"
    assert result.lines[0].confidence == 0.93
    assert result.lines[0].box.y == pytest.approx(0.7)
    request = json.loads(process.stdin.getvalue().splitlines()[0])
    assert request == {
        "id": "request-1",
        "op": "recognize",
        "image": str((tmp_path / "page.png").resolve()),
    }


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("not-json", "valid JSON"),
        ({"id": "wrong", "ok": True, "lines": []}, "request ID"),
        ({"id": "request-1", "ok": False, "error": "model unavailable"}, "model unavailable"),
        ({"id": "request-1", "ok": True, "lines": "bad"}, "lines"),
    ],
)
def test_light_adapter_rejects_invalid_or_failed_responses(
    tmp_path: Path,
    response: dict[str, object] | str,
    message: str,
) -> None:
    engine = _engine(_Process([response]))

    with pytest.raises(LightOcrError, match=message):
        engine.recognize_image(_image(tmp_path))


def test_light_adapter_stops_dead_or_silent_worker(tmp_path: Path) -> None:
    process = _Process([])
    engine = _engine(process)

    with pytest.raises(LightOcrError, match="closed|response"):
        engine.recognize_image(_image(tmp_path))

    assert process.terminated is True


def test_light_adapter_close_is_idempotent() -> None:
    process = _Process([])
    engine = _engine(process)

    engine.close()
    engine.close()

    assert process.stdin.getvalue() == ""


def test_light_adapter_defaults_worker_to_cpu_provider(tmp_path: Path) -> None:
    process = _Process([{"id": "request-1", "ok": True, "lines": []}])
    observed_environment: dict[str, str] = {}

    def factory(*args: object, **kwargs: object) -> _Process:
        observed_environment.update(kwargs["env"])  # type: ignore[arg-type]
        return process

    engine = LightOcrEngine(
        node=Path("/fake/node"),
        worker=Path("/fake/light_ocr_worker.mjs"),
        process_factory=factory,
        response_reader=lambda stream, timeout: stream.readline(),
        request_id_factory=lambda: "request-1",
    )

    engine.recognize_image(_image(tmp_path))

    assert observed_environment["LIGHT_OCR_EXECUTION"] == "cpu"
