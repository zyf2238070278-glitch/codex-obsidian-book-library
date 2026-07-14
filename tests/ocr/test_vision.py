from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import fitz
import pytest

from book_agent.ocr import vision as vision_module
from book_agent.ocr.vision import (
    HELPER_TIMEOUT_SECONDS,
    MAXIMUM_LONG_EDGE_PIXELS,
    MAXIMUM_PAGE_PIXELS,
    MAXIMUM_STDOUT_BYTES,
    VisionOcrEngine,
    VisionOcrError,
)


def _write_pdf(
    path: Path,
    *,
    width: float = 612,
    height: float = 792,
    pages: int = 1,
) -> Path:
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=width, height=height)
        page.insert_text((30, 50), f"synthetic page {index + 1}")
    document.save(path)
    document.close()
    return path


def _helper(tmp_path: Path) -> Path:
    helper = tmp_path / "book-vision-ocr"
    helper.write_bytes(b"#!/bin/sh\nexit 0\n")
    helper.chmod(0o700)
    return helper


def _payload(lines: list[dict[str, object]] | None = None) -> bytes:
    return json.dumps(
        {"schema_version": 1, "lines": lines or []},
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class RecordingRunner:
    def __init__(
        self,
        stdout: bytes | str | None = None,
        *,
        stderr: bytes | str = b"",
        returncode: int = 0,
        error: BaseException | None = None,
    ) -> None:
        self.stdout = _payload() if stdout is None else stdout
        self.stderr = stderr
        self.returncode = returncode
        self.error = error
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        argv: list[str],
        *,
        environment: Mapping[str, str],
        cwd: Path,
        timeout: float,
        executable: Path,
    ) -> subprocess.CompletedProcess[bytes | str]:
        image = Path(argv[2])
        pixmap = fitz.Pixmap(image)
        self.calls.append(
            {
                "argv": list(argv),
                "environment": dict(environment),
                "cwd": cwd,
                "timeout": timeout,
                "executable": executable,
                "executable_identity": (
                    executable.stat().st_dev,
                    executable.stat().st_ino,
                ),
                "image": image,
                "mode": stat.S_IMODE(image.stat().st_mode),
                "width": pixmap.width,
                "height": pixmap.height,
                "components": pixmap.n,
                "xres": pixmap.xres,
                "yres": pixmap.yres,
            }
        )
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(
            argv,
            self.returncode,
            self.stdout,
            self.stderr,
        )


def _engine(
    tmp_path: Path,
    runner: RecordingRunner,
    *,
    helper: Path | None = None,
) -> VisionOcrEngine:
    return VisionOcrEngine(
        helper=_helper(tmp_path) if helper is None else helper,
        temp_root=tmp_path / "ocr-temp",
        run_helper=runner,
    )


def test_recognizes_one_physical_page_at_300_dpi_in_grayscale(tmp_path: Path) -> None:
    pdf = tmp_path / "two-pages.pdf"
    document = fitz.open()
    document.new_page(width=144, height=144)
    document.new_page(width=612, height=792)
    document.save(pdf)
    document.close()
    runner = RecordingRunner(
        _payload(
            [
                {
                    "text": "第二行",
                    "confidence": 0.8,
                    "box": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.1},
                },
                {
                    "text": "First",
                    "confidence": 0.9,
                    "box": {"x": 0.1, "y": 0.8, "width": 0.2, "height": 0.1},
                },
            ]
        )
    )

    result = _engine(tmp_path, runner).recognize_page(pdf, page_index=1)

    assert result.ordered_text() == "First\n第二行"
    assert tuple(line.text for line in result.lines) == ("First", "第二行")
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["width"] == 2550
    assert call["height"] == 3300
    assert call["components"] == 1
    assert call["xres"] == 300
    assert call["yres"] == 300


def test_default_process_runner_explicitly_disables_shell_and_inheritance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self) -> None:
            stdout_read, stdout_write = os.pipe()
            stderr_read, stderr_write = os.pipe()
            os.write(stdout_write, _payload())
            os.close(stdout_write)
            os.close(stderr_write)
            self.stdout = os.fdopen(stdout_read, "rb", buffering=0)
            self.stderr = os.fdopen(stderr_read, "rb", buffering=0)
            self.pid = 999_999

        def wait(self, timeout: float) -> int:
            return 0

    def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(vision_module.subprocess, "Popen", fake_popen)

    result = vision_module._default_run_helper(
        ["/absolute/helper", "--image", "/absolute/page.png"],
        environment={"PATH": "/usr/bin:/bin"},
        cwd=Path("/"),
        timeout=120,
        executable=Path("/absolute/helper-pinned"),
    )

    assert result.returncode == 0
    assert captured["shell"] is False
    assert captured["close_fds"] is True
    assert captured["start_new_session"] is True
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["executable"] == "/absolute/helper-pinned"


def test_default_runner_executes_only_the_pinned_helper(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = tmp_path / "helper with spaces"
    helper.write_text(
        "#!/bin/sh\nprintf '%s' '{\"schema_version\":1,\"lines\":[]}'\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)

    result = VisionOcrEngine(
        helper=helper,
        temp_root=tmp_path / "ocr-temp",
    ).recognize_page(pdf, page_index=0)

    assert result.lines == ()
    assert list((tmp_path / "ocr-temp").iterdir()) == []


def test_default_runner_kills_timed_out_process_group_and_cleans_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = tmp_path / "slow-helper"
    helper.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    helper.chmod(0o700)
    monkeypatch.setattr(vision_module, "HELPER_TIMEOUT_SECONDS", 0.05)
    started = time.monotonic()

    with pytest.raises(VisionOcrError, match="timed out"):
        VisionOcrEngine(
            helper=helper,
            temp_root=tmp_path / "ocr-temp",
        ).recognize_page(pdf, page_index=0)

    assert time.monotonic() - started < 2
    assert list((tmp_path / "ocr-temp").iterdir()) == []


def test_caps_long_edge_and_twenty_million_pixels_proportionally(tmp_path: Path) -> None:
    long_pdf = _write_pdf(tmp_path / "long.pdf", width=4000, height=100)
    large_pdf = _write_pdf(tmp_path / "large.pdf", width=2000, height=2000)

    for pdf in (long_pdf, large_pdf):
        runner = RecordingRunner()
        _engine(tmp_path, runner).recognize_page(pdf, page_index=0)
        width = int(runner.calls[0]["width"])
        height = int(runner.calls[0]["height"])
        assert max(width, height) <= MAXIMUM_LONG_EDGE_PIXELS
        assert width * height <= MAXIMUM_PAGE_PIXELS
        assert width > 0 and height > 0

    long_call = RecordingRunner()
    _engine(tmp_path, long_call).recognize_page(long_pdf, page_index=0)
    assert int(long_call.calls[0]["width"]) == MAXIMUM_LONG_EDGE_PIXELS
    assert int(long_call.calls[0]["height"]) == 300


@pytest.mark.parametrize("page_index", [-1, 1, True, 0.0, "0"])
def test_rejects_invalid_or_out_of_range_physical_page(
    tmp_path: Path, page_index: object
) -> None:
    pdf = _write_pdf(tmp_path / "one.pdf")
    runner = RecordingRunner()

    with pytest.raises((TypeError, VisionOcrError), match="page_index|page"):
        _engine(tmp_path, runner).recognize_page(pdf, page_index=page_index)  # type: ignore[arg-type]

    assert runner.calls == []


def test_rejects_corrupt_and_encrypted_pdf(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a pdf")
    source = _write_pdf(tmp_path / "source.pdf")
    encrypted = tmp_path / "encrypted.pdf"
    with fitz.open(source) as document:
        document.save(
            encrypted,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="secret",
        )

    for pdf in (corrupt, encrypted):
        runner = RecordingRunner()
        with pytest.raises(VisionOcrError, match="PDF|encrypted"):
            _engine(tmp_path, runner).recognize_page(pdf, page_index=0)
        assert runner.calls == []


@pytest.mark.parametrize("dimension", [0.0, float("inf"), float("nan")])
def test_rejects_zero_or_non_finite_page_dimensions_before_rasterizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimension: float,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")

    class FakeDocument:
        needs_pass = False

        def __len__(self) -> int:
            return 1

        def load_page(self, page_index: int) -> object:
            return SimpleNamespace(rect=SimpleNamespace(width=dimension, height=792.0))

        def close(self) -> None:
            pass

    monkeypatch.setattr(vision_module.fitz, "open", lambda path: FakeDocument())

    with pytest.raises(VisionOcrError, match="dimensions"):
        _engine(tmp_path, RecordingRunner()).recognize_page(pdf, page_index=0)


def test_invokes_absolute_helper_and_png_with_fixed_contract(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    runner = RecordingRunner()

    _engine(tmp_path, runner).recognize_page(pdf, page_index=0)

    call = runner.calls[0]
    argv = call["argv"]
    assert argv == [
        str((tmp_path / "book-vision-ocr").absolute()),
        "--image",
        str(Path(argv[2]).absolute()),
        "--languages",
        "zh-Hans,en-US",
    ]
    assert call["timeout"] == HELPER_TIMEOUT_SECONDS == 120.0
    assert call["cwd"] == Path("/")
    assert call["environment"] == {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
    }
    assert call["mode"] == 0o600
    assert Path(call["executable"]).is_absolute()
    assert Path(call["executable"]) != tmp_path / "book-vision-ocr"
    assert call["executable_identity"] == (
        (tmp_path / "book-vision-ocr").stat().st_dev,
        (tmp_path / "book-vision-ocr").stat().st_ino,
    )
    assert not Path(call["executable"]).exists()
    assert Path(argv[2]).parent == (tmp_path / "ocr-temp").absolute()
    assert not Path(argv[2]).exists()
    assert list((tmp_path / "ocr-temp").glob("*.png")) == []


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (RecordingRunner(error=subprocess.TimeoutExpired(["helper"], 120)), "timed out"),
        (RecordingRunner(returncode=9, stderr=b"native failure"), "native failure"),
        (RecordingRunner(stdout=b"\xff"), "UTF-8"),
        (RecordingRunner(stdout=b"{"), "JSON"),
        (RecordingRunner(stdout=b"x" * (MAXIMUM_STDOUT_BYTES + 1)), "output"),
    ],
)
def test_reports_bounded_helper_failures_and_always_cleans_png(
    tmp_path: Path, runner: RecordingRunner, message: str
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")

    with pytest.raises(VisionOcrError, match=message):
        _engine(tmp_path, runner).recognize_page(pdf, page_index=0)

    assert list((tmp_path / "ocr-temp").glob("*.png")) == []


def test_keyboard_interrupt_propagates_and_cleans_png(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    runner = RecordingRunner(error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        _engine(tmp_path, runner).recognize_page(pdf, page_index=0)

    assert list((tmp_path / "ocr-temp").glob("*.png")) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": True, "lines": []},
        {"schema_version": 2, "lines": []},
        {"schema_version": 1, "lines": [], "extra": 1},
        {"schema_version": 1},
        {"schema_version": 1, "lines": {}},
        {"schema_version": 1, "lines": [{"text": "x", "confidence": 1, "box": {}, "extra": 1}]},
        {"schema_version": 1, "lines": [{"text": "", "confidence": 1, "box": {"x": 0, "y": 0, "width": 1, "height": 1}}]},
        {"schema_version": 1, "lines": [{"text": "x", "confidence": True, "box": {"x": 0, "y": 0, "width": 1, "height": 1}}]},
        {"schema_version": 1, "lines": [{"text": "x", "confidence": 1.1, "box": {"x": 0, "y": 0, "width": 1, "height": 1}}]},
        {"schema_version": 1, "lines": [{"text": "x", "confidence": 1, "box": {"x": True, "y": 0, "width": 1, "height": 1}}]},
        {"schema_version": 1, "lines": [{"text": "x", "confidence": 1, "box": {"x": 0, "y": 0, "width": 0, "height": 1}}]},
        {"schema_version": 1, "lines": [{"text": "x", "confidence": 1, "box": {"x": 0.8, "y": 0, "width": 0.3, "height": 1}}]},
        {"schema_version": 1, "lines": [{"text": "x", "confidence": 1, "box": {"x": 0, "y": 0, "width": float("nan"), "height": 1}}]},
        {"schema_version": 1, "lines": [{"text": "x", "confidence": float("inf"), "box": {"x": 0, "y": 0, "width": 1, "height": 1}}]},
        {"schema_version": 1, "lines": [{"text": "x", "confidence": 10**400, "box": {"x": 0, "y": 0, "width": 1, "height": 1}}]},
    ],
)
def test_rejects_malformed_or_non_finite_native_schema(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    raw = json.dumps(payload, allow_nan=True).encode("utf-8")

    with pytest.raises(VisionOcrError, match="schema|line|box|confidence|JSON"):
        _engine(tmp_path, RecordingRunner(raw)).recognize_page(pdf, page_index=0)


def test_tolerates_only_tiny_bbox_rounding_error(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    line = {
        "text": "edge",
        "confidence": 1,
        "box": {"x": 0.8, "y": -0.0000002, "width": 0.2000005, "height": 1.0000001},
    }

    result = _engine(tmp_path, RecordingRunner(_payload([line]))).recognize_page(
        pdf, page_index=0
    )

    assert result.lines[0].box.x == 0.8
    assert result.lines[0].box.y == 0.0
    assert result.lines[0].box.width == pytest.approx(0.2)
    assert result.lines[0].box.height == 1.0


def test_rejects_aggregate_text_over_native_unicode_budget(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    line = {
        "text": "文" * 100_001,
        "confidence": 1,
        "box": {"x": 0, "y": 0, "width": 1, "height": 1},
    }

    with pytest.raises(VisionOcrError, match="text budget"):
        _engine(tmp_path, RecordingRunner(_payload([line]))).recognize_page(
            pdf, page_index=0
        )


def test_rejects_helper_and_temp_symlinks(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = _helper(tmp_path)
    linked_helper = tmp_path / "linked-helper"
    linked_helper.symlink_to(helper)

    with pytest.raises(VisionOcrError, match="symlink"):
        _engine(tmp_path, RecordingRunner(), helper=linked_helper).recognize_page(
            pdf, page_index=0
        )

    real_temp = tmp_path / "real-temp"
    real_temp.mkdir()
    temp_link = tmp_path / "temp-link"
    temp_link.symlink_to(real_temp, target_is_directory=True)
    engine = VisionOcrEngine(
        helper=helper,
        temp_root=temp_link,
        run_helper=RecordingRunner(),
    )
    with pytest.raises(VisionOcrError, match="symlink"):
        engine.recognize_page(pdf, page_index=0)


def test_detects_helper_path_exchange_and_cleans_png(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = _helper(tmp_path)

    class SwappingRunner(RecordingRunner):
        def __call__(self, *args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes | str]:
            result = super().__call__(*args, **kwargs)  # type: ignore[arg-type]
            helper.unlink()
            helper.write_bytes(b"replacement")
            helper.chmod(0o700)
            return result

    with pytest.raises(VisionOcrError, match="changed"):
        _engine(tmp_path, SwappingRunner(), helper=helper).recognize_page(
            pdf, page_index=0
        )
    assert list((tmp_path / "ocr-temp").glob("*.png")) == []


def test_detects_png_path_exchange_without_following_symlink(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")

    class SwappingRunner(RecordingRunner):
        def __call__(self, *args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes | str]:
            result = super().__call__(*args, **kwargs)  # type: ignore[arg-type]
            image = Path(self.calls[-1]["image"])
            image.unlink()
            image.symlink_to(pdf)
            return result

    with pytest.raises(VisionOcrError, match="temporary PNG changed"):
        _engine(tmp_path, SwappingRunner()).recognize_page(pdf, page_index=0)

    assert pdf.is_file()
    assert list((tmp_path / "ocr-temp").iterdir()) == []


def test_helper_must_be_absolute_regular_and_executable(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    directory = tmp_path / "directory"
    directory.mkdir()
    not_executable = tmp_path / "not-executable"
    not_executable.write_text("x", encoding="utf-8")

    for helper in (Path("relative-helper"), directory, not_executable):
        with pytest.raises(VisionOcrError, match="absolute|regular|executable"):
            VisionOcrEngine(
                helper=helper,
                temp_root=tmp_path / "ocr-temp",
                run_helper=RecordingRunner(),
            ).recognize_page(pdf, page_index=0)
