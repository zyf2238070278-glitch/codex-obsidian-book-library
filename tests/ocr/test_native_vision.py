from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "native" / "book_vision_ocr" / "main.swift"
HELPER = PROJECT_ROOT / "bin" / "book-vision-ocr"


def _strict_json(text: str) -> object:
    return json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def _run_helper(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HELPER), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _require_built_helper() -> None:
    if (
        sys.platform != "darwin"
        or not HELPER.is_file()
        or not os.access(HELPER, os.X_OK)
    ):
        pytest.skip("requires macOS and a locally built book-vision-ocr helper")


def _write_synthetic_image(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page(width=1200, height=400)
    page.insert_text(
        (80, 220),
        "VISION TEST 123",
        fontsize=72,
        fontname="helv",
        color=(0, 0, 0),
    )
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = tmp_path / "synthetic-vision.png"
    pixmap.save(image)
    document.close()
    return image


def test_swift_source_uses_required_vision_and_imageio_contract() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    for required in (
        "CGImageSourceCreateWithURL",
        "VNImageRequestHandler",
        "VNRecognizeTextRequest",
        ".accurate",
        "usesLanguageCorrection = true",
        "supportedRecognitionLanguages",
        "JSONEncoder",
        "maximumRecognizedCharacters = 100_000",
    ):
        assert required in source


@pytest.mark.macos_vision
def test_native_helper_reports_version_and_required_capabilities() -> None:
    _require_built_helper()

    version = _run_helper("--version")
    capabilities = _run_helper("--capabilities")

    assert version.returncode == 0
    assert version.stderr == ""
    version_payload = _strict_json(version.stdout)
    assert version_payload == {"schema_version": 1, "version": "0.1.0"}
    assert capabilities.returncode == 0
    assert capabilities.stderr == ""
    payload = _strict_json(capabilities.stdout)
    assert type(payload) is dict
    assert set(payload) == {"schema_version", "languages"}
    assert type(payload["schema_version"]) is int
    assert payload["schema_version"] == 1
    assert type(payload["languages"]) is list
    assert all(type(language) is str for language in payload["languages"])
    assert {"zh-Hans", "en-US"}.issubset(payload["languages"])


@pytest.mark.macos_vision
def test_native_helper_recognizes_synthetic_image_with_normalized_native_json(
    tmp_path: Path,
) -> None:
    _require_built_helper()
    image = _write_synthetic_image(tmp_path)

    completed = _run_helper(
        "--image",
        str(image.resolve()),
        "--languages",
        "zh-Hans,en-US",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = _strict_json(completed.stdout)
    assert type(payload) is dict
    assert set(payload) == {"schema_version", "lines"}
    assert type(payload["schema_version"]) is int
    assert payload["schema_version"] == 1
    assert type(payload["lines"]) is list
    assert "VISION" in " ".join(line["text"] for line in payload["lines"]).upper()
    previous_key: tuple[float, float, str] | None = None
    for line in payload["lines"]:
        assert type(line) is dict
        assert set(line) == {"text", "confidence", "box"}
        assert type(line["text"]) is str and line["text"]
        assert type(line["confidence"]) is float
        assert math.isfinite(line["confidence"])
        assert 0.0 <= line["confidence"] <= 1.0
        box = line["box"]
        assert type(box) is dict
        assert set(box) == {"x", "y", "width", "height"}
        for key in ("x", "y", "width", "height"):
            assert type(box[key]) is float
            assert math.isfinite(box[key])
            assert 0.0 <= box[key] <= 1.0
        assert box["x"] + box["width"] <= 1.000001
        assert box["y"] + box["height"] <= 1.000001
        key = (-box["y"], box["x"], line["text"])
        if previous_key is not None:
            assert previous_key <= key
        previous_key = key


@pytest.mark.macos_vision
def test_native_helper_rejects_symlink_directory_and_unsupported_language(
    tmp_path: Path,
) -> None:
    _require_built_helper()
    image = _write_synthetic_image(tmp_path)
    symlink = tmp_path / "linked.png"
    symlink.symlink_to(image)

    for arguments in (
        ("--image", str(symlink), "--languages", "en-US"),
        ("--image", str(tmp_path), "--languages", "en-US"),
        ("--image", str(image), "--languages", "xx-INVALID"),
    ):
        completed = _run_helper(*arguments)
        assert completed.returncode != 0
        assert completed.stdout == ""
        assert completed.stderr.strip()


@pytest.mark.macos_vision
@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("--image", "relative.png", "--languages", "zh-Hans,en-US"),
        (
            "--image",
            "https://example.invalid/a.png",
            "--languages",
            "zh-Hans,en-US",
        ),
        ("--image", "/definitely/missing.png", "--languages", "zh-Hans,en-US"),
        ("--image", "/tmp/a.png", "--languages", ""),
        ("--image", "/tmp/a.png", "--languages", "xx-INVALID"),
        ("--image", "/tmp/a.png", "--languages", "en-US,en-US"),
        ("--image", "/tmp/a.png", "--image", "/tmp/b.png"),
        ("--unknown",),
        ("--version", "--version"),
    ],
)
def test_native_helper_rejects_invalid_arguments_without_stdout_json(
    arguments: tuple[str, ...],
) -> None:
    _require_built_helper()

    completed = _run_helper(*arguments)

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.strip()
