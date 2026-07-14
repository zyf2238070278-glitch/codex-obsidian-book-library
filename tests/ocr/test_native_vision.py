from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from scripts.build_vision_helper import build_vision_helper


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "native" / "book_vision_ocr" / "main.swift"
MAXIMUM_IMAGE_FILE_BYTES = 64 * 1024 * 1024
MAXIMUM_IMAGE_DIMENSION = 12_000


def _strict_json(text: str) -> object:
    return json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


@pytest.fixture(scope="session")
def vision_helper(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if sys.platform != "darwin":
        pytest.skip("Apple Vision helper requires macOS")
    output = tmp_path_factory.mktemp("native-vision-helper") / "book-vision-ocr"
    return build_vision_helper(source=SOURCE, output=output)


def _run_helper(
    helper: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(helper), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _png_header(width: int, height: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    compressor = zlib.compressobj(level=9)
    row = b"\x00" * (1 + ((width + 7) // 8))
    compressed = bytearray()
    for _ in range(height):
        compressed.extend(compressor.compress(row))
    compressed.extend(compressor.flush())
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", bytes(compressed))
        + chunk(b"IEND", b"")
    )


def _write_synthetic_image(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page(width=1200, height=600)
    page.insert_text(
        (80, 200),
        "VISION TEST 123",
        fontsize=72,
        fontname="helv",
        color=(0, 0, 0),
    )
    page.insert_text(
        (80, 420),
        "中文识别测试",
        fontsize=72,
        fontname="china-s",
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
        "O_NOFOLLOW",
        "O_CLOEXEC",
        "fstat",
        "CGImageSourceCreateWithData",
        "CGImageSourceCopyPropertiesAtIndex",
        "VNImageRequestHandler",
        "VNRecognizeTextRequest",
        ".accurate",
        "usesLanguageCorrection = true",
        "supportedRecognitionLanguages",
        "JSONEncoder",
        "maximumRecognizedUnicodeScalars = 100_000",
        "maximumRecognizedUTF8Bytes = 400_000",
        "maximumImageFileBytes = 64 * 1024 * 1024",
        "maximumImageDimension = 12_000",
        "maximumImagePixels = 40_000_000",
    ):
        assert required in source
    assert "CGImageSourceCreateWithURL" not in source


@pytest.mark.macos_vision
def test_native_helper_reports_version_and_required_capabilities(
    vision_helper: Path,
) -> None:
    version = _run_helper(vision_helper, "--version")
    capabilities = _run_helper(vision_helper, "--capabilities")

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
    vision_helper: Path,
) -> None:
    image = _write_synthetic_image(tmp_path)

    completed = _run_helper(
        vision_helper,
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
    recognized_text = " ".join(line["text"] for line in payload["lines"])
    assert "VISION" in recognized_text.upper()
    assert sum("\u4e00" <= character <= "\u9fff" for character in recognized_text) >= 2
    previous_key: tuple[float, float, str] | None = None
    for line in payload["lines"]:
        assert type(line) is dict
        assert set(line) == {"text", "confidence", "box"}
        assert type(line["text"]) is str and line["text"]
        assert isinstance(line["confidence"], (int, float))
        assert not isinstance(line["confidence"], bool)
        confidence = float(line["confidence"])
        assert math.isfinite(confidence)
        assert 0.0 <= confidence <= 1.0
        box = line["box"]
        assert type(box) is dict
        assert set(box) == {"x", "y", "width", "height"}
        for key in ("x", "y", "width", "height"):
            assert isinstance(box[key], (int, float))
            assert not isinstance(box[key], bool)
            coordinate = float(box[key])
            assert math.isfinite(coordinate)
            assert 0.0 <= coordinate <= 1.0
        assert box["x"] + box["width"] <= 1.000001
        assert box["y"] + box["height"] <= 1.000001
        key = (-box["y"], box["x"], line["text"])
        if previous_key is not None:
            assert previous_key <= key
        previous_key = key


@pytest.mark.macos_vision
def test_native_helper_rejects_symlink_directory_and_unsupported_language(
    tmp_path: Path,
    vision_helper: Path,
) -> None:
    image = _write_synthetic_image(tmp_path)
    symlink = tmp_path / "linked.png"
    symlink.symlink_to(image)

    for arguments in (
        ("--image", str(symlink), "--languages", "en-US"),
        ("--image", str(tmp_path), "--languages", "en-US"),
        ("--image", str(image), "--languages", "xx-INVALID"),
    ):
        completed = _run_helper(vision_helper, *arguments)
        assert completed.returncode != 0
        assert completed.stdout == ""
        assert completed.stderr.strip()


@pytest.mark.macos_vision
def test_native_helper_rejects_oversized_and_malformed_images_before_decode(
    tmp_path: Path,
    vision_helper: Path,
) -> None:
    oversized_side = tmp_path / "oversized-side.png"
    oversized_side.write_bytes(_png_header(MAXIMUM_IMAGE_DIMENSION + 1, 1))
    oversized_pixels = tmp_path / "oversized-pixels.png"
    oversized_pixels.write_bytes(_png_header(8_000, 6_000))
    boundary = tmp_path / "boundary-dimension.png"
    boundary.write_bytes(_png_header(MAXIMUM_IMAGE_DIMENSION, 1))
    malformed = tmp_path / "malformed.png"
    malformed.write_bytes(b"not an image")
    oversized_file = tmp_path / "oversized-file.png"
    with oversized_file.open("wb") as stream:
        stream.truncate(MAXIMUM_IMAGE_FILE_BYTES + 1)

    cases = (
        (oversized_side, "exceeds the 12,000 pixel side limit"),
        (oversized_pixels, "pixel count exceeds"),
        (malformed, "image source"),
        (oversized_file, "file size"),
    )
    for image, diagnostic in cases:
        completed = _run_helper(
            vision_helper,
            "--image",
            str(image),
            "--languages",
            "en-US",
        )
        assert completed.returncode != 0
        assert completed.stdout == ""
        assert diagnostic in completed.stderr.lower()

    boundary_result = _run_helper(
        vision_helper,
        "--image",
        str(boundary),
        "--languages",
        "en-US",
    )
    assert boundary_result.returncode != 0
    assert boundary_result.stdout == ""
    assert "exceeds the 12,000 pixel side limit" not in boundary_result.stderr
    assert "pixel count exceeds" not in boundary_result.stderr


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
    vision_helper: Path,
) -> None:
    completed = _run_helper(vision_helper, *arguments)

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.strip()
