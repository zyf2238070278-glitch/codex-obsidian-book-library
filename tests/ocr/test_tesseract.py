from __future__ import annotations

import subprocess
from pathlib import Path

import fitz

from book_agent.ocr.tesseract import TesseractEngine


def _image(path: Path) -> Path:
    pixmap = fitz.Pixmap(fitz.csGRAY, 100, 100, b"\xff" * 10_000, False)
    pixmap.save(path)
    return path


def _binary(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _tessdata(path: Path) -> Path:
    path.mkdir()
    for language in ("chi_sim", "chi_tra", "eng", "osd"):
        (path / f"{language}.traineddata").write_bytes(b"data")
    return path


VALID_AND_INVALID_ROWS = """level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t95\t有效行
5\t1\t1\t1\t2\t1\t0\t0\t0\t0\t80\t坏框
5\t1\t1\t1\t3\t1\t40\t40\t20\t10\t-1\t无效置信度
"""


def test_tesseract_uses_only_packaged_binary_and_tessdata(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, VALID_AND_INVALID_ROWS, "")

    binary = _binary(tmp_path / "tesseract")
    tessdata = _tessdata(tmp_path / "tessdata")
    image = _image(tmp_path / "page.png")
    TesseractEngine(binary=binary, tessdata=tessdata, runner=runner).recognize_image(image)

    assert captured["argv"] == [
        str(binary),
        str(image),
        "stdout",
        "--tessdata-dir",
        str(tessdata),
        "-l",
        "chi_sim+chi_tra+eng",
        "tsv",
    ]
    assert captured["kwargs"] == {
        "shell": False,
        "check": False,
        "text": True,
        "capture_output": True,
        "timeout": 120,
        "env": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    }


def test_tesseract_discards_invalid_tsv_rows(tmp_path: Path) -> None:
    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, VALID_AND_INVALID_ROWS, "")

    result = TesseractEngine(
        binary=_binary(tmp_path / "tesseract"),
        tessdata=_tessdata(tmp_path / "tessdata"),
        runner=runner,
    ).recognize_image(_image(tmp_path / "page.png"))

    assert result.ordered_text() == "有效行"
    assert result.discarded_observations == 2
