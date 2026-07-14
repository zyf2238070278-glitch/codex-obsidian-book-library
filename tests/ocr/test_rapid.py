from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from book_agent.ocr.rapid import RapidOcrEngine, RapidOcrError


def _image(path: Path) -> Path:
    pixmap = fitz.Pixmap(fitz.csGRAY, 100, 100, b"\xff" * 10_000, False)
    pixmap.save(path)
    return path


def _models(root: Path) -> Path:
    root.mkdir()
    for name in ("det.onnx", "rec.onnx", "cls.onnx"):
        (root / name).write_bytes(b"model")
    return root


def test_rapid_engine_rejects_missing_pinned_model(tmp_path: Path) -> None:
    engine = RapidOcrEngine(model_root=tmp_path / "missing")

    with pytest.raises(RapidOcrError, match="RapidOCR model is missing"):
        engine.recognize_image(_image(tmp_path / "page.png"))


def test_rapid_engine_normalizes_polygons_to_boxes(tmp_path: Path) -> None:
    class FakeRapid:
        def __call__(self, image_path: str) -> list[object]:
            assert Path(image_path).name == "page.png"
            return [
                [
                    [[10, 20], [40, 20], [40, 30], [10, 30]],
                    "有效文字",
                    0.9,
                ],
                [
                    [[0, 0], [0, 0], [0, 0], [0, 0]],
                    "坏框",
                    0.8,
                ],
            ]

    result = RapidOcrEngine(
        _models(tmp_path / "models"),
        factory=lambda **_: FakeRapid(),
    ).recognize_image(_image(tmp_path / "page.png"))

    assert result.engine == "rapidocr"
    assert result.ordered_text() == "有效文字"
    assert result.lines[0].box.x == pytest.approx(0.1)
    assert result.lines[0].box.y == pytest.approx(0.2)
    assert result.lines[0].box.width == pytest.approx(0.3)
    assert result.lines[0].box.height == pytest.approx(0.1)
    assert result.discarded_observations == 1
