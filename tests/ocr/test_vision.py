from __future__ import annotations

import json
import os
import signal
import shutil
import stat
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from book_agent.ocr import vision as vision_module
from book_agent.ocr.vision import (
    HELPER_TIMEOUT_SECONDS,
    MAXIMUM_HELPER_BYTES,
    MAXIMUM_LONG_EDGE_PIXELS,
    MAXIMUM_PAGE_PIXELS,
    MAXIMUM_STDOUT_BYTES,
    RunRequest,
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
        {
            "schema_version": 2,
            "lines": lines or [],
            "discarded_observations": 0,
        },
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_parser_accepts_valid_lines_when_one_native_box_is_discarded() -> None:
    result = vision_module._parse_helper_output(
        json.dumps(
            {
                "schema_version": 2,
                "discarded_observations": 1,
                "lines": [
                    {
                        "text": "有效文字",
                        "confidence": 0.9,
                        "box": {"x": 0.1, "y": 0.5, "width": 0.2, "height": 0.1},
                    }
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )

    assert result.ordered_text() == "有效文字"
    assert result.discarded_observations == 1


def _read_process_ids(path: Path) -> tuple[int, int]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if path.is_file():
            values = path.read_text(encoding="utf-8").split()
            if len(values) == 2:
                return int(values[0]), int(values[1])
        time.sleep(0.01)
    raise AssertionError("synthetic helper did not record process IDs")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _assert_processes_gone(*process_ids: int) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not any(_pid_exists(pid) for pid in process_ids):
            return
        time.sleep(0.01)
    assert not [pid for pid in process_ids if _pid_exists(pid)]


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
        self.temp_root: Path | None = None

    def __call__(
        self,
        request: RunRequest,
    ) -> subprocess.CompletedProcess[bytes | str]:
        argv = request.argv
        image = Path(argv[2])
        pixmap = fitz.Pixmap(image)
        image_fd = int(image.name)
        self.calls.append(
            {
                "argv": list(argv),
                "environment": dict(request.environment),
                "cwd": request.cwd,
                "timeout": request.timeout,
                "pass_fds": request.pass_fds,
                "has_executable": hasattr(request, "executable"),
                "image": image,
                "image_fd": image_fd,
                "mode": stat.S_IMODE(os.fstat(image_fd).st_mode),
                "temp_entries": (
                    list(self.temp_root.iterdir())
                    if self.temp_root is not None
                    else None
                ),
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
            list(argv),
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
    runner.temp_root = tmp_path / "ocr-temp"
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = _helper(tmp_path)
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
            assert Path(captured["executable"]).is_file()
            return 0

    def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(vision_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vision_module, "_verify_codesign", lambda snapshot: None)

    result = VisionOcrEngine(
        helper=helper,
        temp_root=tmp_path / "ocr-temp",
    ).recognize_page(pdf, page_index=0)

    assert result.lines == ()
    assert captured["shell"] is False
    assert captured["close_fds"] is True
    assert captured["start_new_session"] is True
    assert captured["stdin"] is subprocess.DEVNULL
    assert Path(captured["executable"]).name == "book-vision-ocr"
    image_fd = int(Path(captured["argv"][2]).name)  # type: ignore[index]
    assert captured["argv"] == [
        str(helper),
        "--image",
        f"/dev/fd/{image_fd}",
        "--languages",
        "zh-Hans,en-US",
    ]
    assert captured["pass_fds"] == (image_fd,)
    assert not Path(captured["executable"]).exists()
    assert list((tmp_path / "ocr-temp").iterdir()) == []


def test_default_runner_executes_only_the_private_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = tmp_path / "helper with spaces"
    shutil.copyfile("/usr/bin/true", helper)
    helper.chmod(0o700)
    helper_bytes = helper.read_bytes()
    helper_metadata = helper.stat()
    descriptors_before = set(os.listdir("/dev/fd"))
    monkeypatch.setattr(vision_module, "_verify_codesign", lambda snapshot: None)

    with pytest.raises(VisionOcrError, match="JSON"):
        VisionOcrEngine(
            helper=helper,
            temp_root=tmp_path / "ocr-temp",
        ).recognize_page(pdf, page_index=0)

    assert list((tmp_path / "ocr-temp").iterdir()) == []
    assert set(os.listdir("/dev/fd")) == descriptors_before
    assert helper.read_bytes() == helper_bytes
    after = helper.stat()
    assert (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_mtime_ns,
    ) == (
        helper_metadata.st_dev,
        helper_metadata.st_ino,
        helper_metadata.st_size,
        helper_metadata.st_mode,
        helper_metadata.st_mtime_ns,
    )


def test_codesign_verification_is_fixed_bounded_and_rechecks_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _helper(tmp_path)
    engine = VisionOcrEngine(helper=helper, temp_root=tmp_path / "ocr-temp")
    source_fd, identity, digest = engine._validate_helper()
    temp_root = engine._prepare_temp_root()
    directory = engine._create_call_directory(temp_root)
    snapshot = engine._snapshot_helper(source_fd, directory, identity, digest)
    captured: dict[str, object] = {}

    def fake_bounded(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(vision_module, "_run_bounded_command", fake_bounded)
    try:
        vision_module._verify_codesign(snapshot)
    finally:
        os.close(snapshot.descriptor)
        os.close(source_fd)
        os.unlink(snapshot.path)
        os.rmdir(snapshot.directory)

    assert captured["argv"] == [
        "/usr/bin/codesign",
        "--verify",
        "--strict",
        str(snapshot.path),
    ]
    assert captured["timeout"] == 30.0
    assert captured["cwd"] == Path("/")
    assert captured["environment"] == {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
    }


def test_codesign_timeout_reports_the_codesign_stage_and_actual_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, source_fd = _private_snapshot_fixture(tmp_path)
    monkeypatch.setattr(
        vision_module,
        "_run_bounded_command",
        lambda argv, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(argv, kwargs["timeout"])
        ),
    )
    try:
        with pytest.raises(VisionOcrError, match="code-sign.*timed out.*30"):
            vision_module._verify_codesign(snapshot)
    finally:
        os.close(snapshot.descriptor)
        os.close(source_fd)
        snapshot.path.unlink()
        snapshot.directory.rmdir()


def test_engine_does_not_misreport_codesign_timeout_as_helper_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = _helper(tmp_path)
    monkeypatch.setattr(
        vision_module,
        "_run_bounded_command",
        lambda argv, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(argv, kwargs["timeout"])
        ),
    )

    with pytest.raises(VisionOcrError) as captured:
        VisionOcrEngine(
            helper=helper,
            temp_root=tmp_path / "ocr-temp",
        ).recognize_page(pdf, page_index=0)

    assert "code-sign verification timed out after 30 seconds" in str(captured.value)
    assert "120" not in str(captured.value)
    assert list((tmp_path / "ocr-temp").iterdir()) == []


def _private_snapshot_fixture(
    tmp_path: Path,
) -> tuple[vision_module._ExecutableSnapshot, int]:
    helper = _helper(tmp_path)
    engine = VisionOcrEngine(helper=helper, temp_root=tmp_path / "ocr-temp")
    source_fd, identity, digest = engine._validate_helper()
    temp_root = engine._prepare_temp_root()
    directory = engine._create_call_directory(temp_root)
    return engine._snapshot_helper(source_fd, directory, identity, digest), source_fd


@pytest.mark.parametrize(
    "stage",
    [
        "missing-pipe",
        "deadline",
        "selector",
        "set-blocking-stdout",
        "set-blocking-stderr",
        "register-stdout",
        "register-stderr",
    ],
)
def test_default_runner_reaps_process_and_closes_pipes_when_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    process_ids = tmp_path / f"{stage}.pids"
    helper = tmp_path / "group-helper"
    helper.write_text(
        "#!/bin/sh\n"
        "/bin/sleep 30 & child=$!\n"
        "/bin/echo \"$$ $child\" > \"$PID_FILE\"\n"
        "wait\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)
    engine = VisionOcrEngine(helper=helper, temp_root=tmp_path / "ocr-temp")
    source_fd, identity, digest = engine._validate_helper()
    directory = engine._create_call_directory(engine._prepare_temp_root())
    snapshot = engine._snapshot_helper(source_fd, directory, identity, digest)
    request = RunRequest(
        argv=(str(helper),),
        environment=(
            ("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
            ("PID_FILE", str(process_ids)),
        ),
        cwd=Path("/"),
        timeout=120.0,
        pass_fds=(),
    )
    real_popen = subprocess.Popen
    real_selector = vision_module.selectors.DefaultSelector
    real_set_blocking = os.set_blocking
    process_holder: dict[str, subprocess.Popen[bytes]] = {}
    selector_holder: dict[str, object] = {}
    popen_returned = False
    deadline_failed = False

    class MissingPipeProcess:
        def __init__(self, process: subprocess.Popen[bytes]) -> None:
            self.pid = process.pid
            self.stdout = process.stdout
            self.stderr = None
            self._process = process

        def wait(self, timeout: float) -> int:
            return self._process.wait(timeout=timeout)

        def kill(self) -> None:
            self._process.kill()

    def synchronized_popen(
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.Popen[bytes] | MissingPipeProcess:
        nonlocal popen_returned
        process = real_popen(argv, **kwargs)  # type: ignore[arg-type]
        process_holder["process"] = process
        _read_process_ids(process_ids)
        popen_returned = True
        if stage == "missing-pipe":
            return MissingPipeProcess(process)
        return process

    class TrackingSelector:
        def __init__(self) -> None:
            self._selector = real_selector()
            self.register_calls = 0
            self.closed = False
            selector_holder["selector"] = self

        def register(self, *args: object, **kwargs: object) -> object:
            self.register_calls += 1
            target = 1 if stage == "register-stdout" else 2
            if stage.startswith("register-") and self.register_calls == target:
                raise RuntimeError(f"synthetic {stage} failure")
            return self._selector.register(*args, **kwargs)

        def close(self) -> None:
            self.closed = True
            self._selector.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self._selector, name)

    blocking_calls = 0

    def failing_set_blocking(descriptor: int, blocking: bool) -> None:
        nonlocal blocking_calls
        blocking_calls += 1
        target = 1 if stage == "set-blocking-stdout" else 2
        if stage.startswith("set-blocking-") and blocking_calls == target:
            raise RuntimeError(f"synthetic {stage} failure")
        real_set_blocking(descriptor, blocking)

    real_monotonic = time.monotonic

    def failing_monotonic() -> float:
        nonlocal deadline_failed
        if stage == "deadline" and popen_returned and not deadline_failed:
            deadline_failed = True
            raise KeyboardInterrupt("synthetic deadline failure")
        return real_monotonic()

    monkeypatch.setattr(vision_module, "_verify_codesign", lambda value: None)
    monkeypatch.setattr(vision_module.subprocess, "Popen", synchronized_popen)
    if stage == "selector":
        monkeypatch.setattr(
            vision_module.selectors,
            "DefaultSelector",
            lambda: (_ for _ in ()).throw(RuntimeError("synthetic selector failure")),
        )
    else:
        monkeypatch.setattr(
            vision_module.selectors,
            "DefaultSelector",
            TrackingSelector,
        )
    monkeypatch.setattr(vision_module.os, "set_blocking", failing_set_blocking)
    monkeypatch.setattr(vision_module.time, "monotonic", failing_monotonic)
    try:
        expected = (
            KeyboardInterrupt
            if stage == "deadline"
            else (RuntimeError, VisionOcrError)
        )
        with pytest.raises(expected, match="synthetic|pipe"):
            vision_module._default_run_helper(request, snapshot)

        leader_pid, child_pid = _read_process_ids(process_ids)
        _assert_processes_gone(leader_pid, child_pid)
        process = process_holder["process"]
        assert process.stdout is not None and process.stdout.closed
        if stage == "missing-pipe":
            assert process.stderr is not None
            process.stderr.close()
        else:
            assert process.stderr is not None and process.stderr.closed
        if "selector" in selector_holder:
            assert selector_holder["selector"].closed is True  # type: ignore[union-attr]
        assert not snapshot.path.exists()
        assert not snapshot.directory.exists()
    finally:
        os.close(snapshot.descriptor)
        os.close(source_fd)
        process = process_holder.get("process")
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            if process.stdout is not None and not process.stdout.closed:
                process.stdout.close()
            if process.stderr is not None and not process.stderr.closed:
                process.stderr.close()
        if snapshot.path.exists():
            snapshot.path.unlink()
        if snapshot.directory.exists():
            snapshot.directory.rmdir()


def test_popen_failure_does_not_attempt_to_kill_an_unstarted_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, source_fd = _private_snapshot_fixture(tmp_path)
    request = RunRequest(
        argv=(str(tmp_path / "book-vision-ocr"), "--version"),
        environment=(("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),),
        cwd=Path("/"),
        timeout=120.0,
        pass_fds=(),
    )
    terminated: list[object] = []
    monkeypatch.setattr(vision_module, "_verify_codesign", lambda value: None)
    monkeypatch.setattr(
        vision_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("synthetic Popen failure")
        ),
    )
    monkeypatch.setattr(
        vision_module,
        "_terminate_process_group",
        lambda process: terminated.append(process),
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="Popen"):
            vision_module._default_run_helper(request, snapshot)

        assert terminated == []
        assert not snapshot.path.exists()
        assert not snapshot.directory.exists()
    finally:
        os.close(snapshot.descriptor)
        os.close(source_fd)
        if snapshot.path.exists():
            snapshot.path.unlink()
        if snapshot.directory.exists():
            snapshot.directory.rmdir()


@pytest.mark.parametrize(
    "outcome",
    ["success", "nonzero", "timeout", "output-limit", "interrupt", "popen"],
)
def test_default_runner_keeps_snapshot_until_bounded_command_finishes_then_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    snapshot, source_fd = _private_snapshot_fixture(tmp_path)
    request = RunRequest(
        argv=(str(tmp_path / "book-vision-ocr"), "--version"),
        environment=(("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),),
        cwd=Path("/"),
        timeout=120.0,
        pass_fds=(),
    )

    def fake_bounded(
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        assert snapshot.path.is_file()
        assert snapshot.directory.is_dir()
        assert "after_spawn" not in kwargs
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, 120)
        if outcome == "output-limit":
            raise VisionOcrError("synthetic output limit")
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        if outcome == "popen":
            raise OSError("synthetic Popen failure")
        return subprocess.CompletedProcess(
            argv,
            9 if outcome == "nonzero" else 0,
            b"",
            b"",
        )

    monkeypatch.setattr(vision_module, "_verify_codesign", lambda value: None)
    monkeypatch.setattr(vision_module, "_run_bounded_command", fake_bounded)
    try:
        if outcome == "timeout":
            with pytest.raises(subprocess.TimeoutExpired):
                vision_module._default_run_helper(request, snapshot)
        elif outcome == "output-limit":
            with pytest.raises(VisionOcrError, match="output limit"):
                vision_module._default_run_helper(request, snapshot)
        elif outcome == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                vision_module._default_run_helper(request, snapshot)
        elif outcome == "popen":
            with pytest.raises(OSError, match="Popen"):
                vision_module._default_run_helper(request, snapshot)
        else:
            vision_module._default_run_helper(request, snapshot)

        assert not snapshot.path.exists()
        assert not snapshot.directory.exists()
    finally:
        os.close(snapshot.descriptor)
        os.close(source_fd)
        if snapshot.path.exists():
            snapshot.path.unlink()
        if snapshot.directory.exists():
            snapshot.directory.rmdir()


def test_default_runner_reports_cleanup_failure_without_masking_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, source_fd = _private_snapshot_fixture(tmp_path)
    request = RunRequest(
        argv=(str(tmp_path / "book-vision-ocr"), "--version"),
        environment=(("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),),
        cwd=Path("/"),
        timeout=120.0,
        pass_fds=(),
    )
    real_unlink = os.unlink

    def fail_snapshot_unlink(path: object, *args: object, **kwargs: object) -> None:
        if Path(path) == snapshot.path:
            raise PermissionError("synthetic cleanup failure")
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(vision_module, "_verify_codesign", lambda value: None)
    monkeypatch.setattr(
        vision_module,
        "_run_bounded_command",
        lambda argv, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(vision_module.os, "unlink", fail_snapshot_unlink)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            vision_module._default_run_helper(request, snapshot)
        assert any(
            "cleanup" in note.lower()
            for note in getattr(captured.value, "__notes__", ())
        )
    finally:
        monkeypatch.setattr(vision_module.os, "unlink", real_unlink)
        os.close(snapshot.descriptor)
        os.close(source_fd)
        if snapshot.path.exists():
            snapshot.path.unlink()
        if snapshot.directory.exists():
            snapshot.directory.rmdir()


def test_private_cleanup_refuses_replaced_snapshot_without_unlinking_it(
    tmp_path: Path,
) -> None:
    snapshot, source_fd = _private_snapshot_fixture(tmp_path)
    replacement = tmp_path / "external-synthetic-file"
    replacement_bytes = b"synthetic external content must survive"
    replacement.write_bytes(replacement_bytes)
    os.replace(replacement, snapshot.path)
    try:
        diagnostic = vision_module._cleanup_private_snapshot(snapshot)

        assert diagnostic is not None and "refused" in diagnostic
        assert snapshot.path.read_bytes() == replacement_bytes
        assert snapshot.directory.is_dir()
    finally:
        os.close(snapshot.descriptor)
        os.close(source_fd)
        if snapshot.path.exists():
            snapshot.path.unlink()
        if snapshot.directory.exists():
            snapshot.directory.rmdir()


def test_nonzero_default_run_preserves_exit_error_and_cleans_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = _helper(tmp_path)
    monkeypatch.setattr(vision_module, "_verify_codesign", lambda value: None)
    monkeypatch.setattr(
        vision_module,
        "_run_bounded_command",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv,
            9,
            b"",
            b"synthetic nonzero",
        ),
    )

    with pytest.raises(VisionOcrError, match="exit code 9.*synthetic nonzero"):
        VisionOcrEngine(
            helper=helper,
            temp_root=tmp_path / "ocr-temp",
        ).recognize_page(pdf, page_index=0)

    assert list((tmp_path / "ocr-temp").iterdir()) == []


def test_successful_default_run_converts_cleanup_failure_to_vision_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = _helper(tmp_path)
    real_unlink = os.unlink
    blocked_paths: list[Path] = []

    def fail_private_snapshot_unlink(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        candidate = Path(path)
        if (
            candidate.name == "book-vision-ocr"
            and candidate.parent.name.startswith("vision-call-")
        ):
            blocked_paths.append(candidate)
            raise PermissionError("synthetic cleanup failure")
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(vision_module, "_verify_codesign", lambda value: None)
    monkeypatch.setattr(
        vision_module,
        "_run_bounded_command",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, _payload(), b""),
    )
    monkeypatch.setattr(vision_module.os, "unlink", fail_private_snapshot_unlink)
    try:
        with pytest.raises(VisionOcrError, match="cleanup"):
            VisionOcrEngine(
                helper=helper,
                temp_root=tmp_path / "ocr-temp",
            ).recognize_page(pdf, page_index=0)
    finally:
        monkeypatch.setattr(vision_module.os, "unlink", real_unlink)
        for path in blocked_paths:
            if path.exists():
                path.unlink()
            if path.parent.exists():
                path.parent.rmdir()


def test_call_directory_creation_failure_does_not_leave_named_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_root = tmp_path / "ocr-temp"
    temp_root.mkdir()
    real_chmod = os.chmod

    def fail_private_directory_chmod(path: object, *args: object, **kwargs: object) -> None:
        if Path(path).name.startswith("vision-call-"):
            raise PermissionError("synthetic chmod failure")
        real_chmod(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(vision_module.os, "chmod", fail_private_directory_chmod)

    with pytest.raises(VisionOcrError, match="call directory"):
        VisionOcrEngine._create_call_directory(temp_root)

    assert list(temp_root.iterdir()) == []


@pytest.mark.parametrize(
    ("stage", "error_type"),
    [
        ("fchmod", OSError),
        ("fchmod", KeyboardInterrupt),
        ("fstat", OSError),
        ("fstat", KeyboardInterrupt),
        ("unlink", OSError),
        ("unlink", KeyboardInterrupt),
    ],
)
def test_anonymous_png_creation_failure_closes_fd_and_removes_named_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_type: type[BaseException],
) -> None:
    document = fitz.open()
    page = document.new_page(width=20, height=20)
    pixmap = page.get_pixmap(colorspace=fitz.csGRAY, alpha=False)
    directory = tmp_path / "private-call"
    directory.mkdir(mode=0o700)
    before_fds = set(os.listdir("/dev/fd"))
    real_fchmod = os.fchmod
    real_fstat = os.fstat
    real_unlink = os.unlink
    failed = False

    def maybe_fail() -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise error_type(f"synthetic {stage} failure")

    def failing_fchmod(descriptor: int, mode: int) -> None:
        if stage == "fchmod":
            maybe_fail()
        real_fchmod(descriptor, mode)

    def failing_fstat(descriptor: int) -> os.stat_result:
        if stage == "fstat":
            maybe_fail()
        return real_fstat(descriptor)

    def failing_unlink(path: object, *args: object, **kwargs: object) -> None:
        if stage == "unlink":
            maybe_fail()
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(vision_module.os, "fchmod", failing_fchmod)
    monkeypatch.setattr(vision_module.os, "fstat", failing_fstat)
    monkeypatch.setattr(vision_module.os, "unlink", failing_unlink)
    expected = KeyboardInterrupt if error_type is KeyboardInterrupt else VisionOcrError
    try:
        with pytest.raises(expected, match="synthetic|temporary PNG"):
            VisionOcrEngine._write_anonymous_png(pixmap, directory)

        assert list(directory.iterdir()) == []
        assert set(os.listdir("/dev/fd")) == before_fds
    finally:
        monkeypatch.setattr(vision_module.os, "fchmod", real_fchmod)
        monkeypatch.setattr(vision_module.os, "fstat", real_fstat)
        monkeypatch.setattr(vision_module.os, "unlink", real_unlink)
        document.close()
        for entry in directory.iterdir():
            entry.unlink()


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, MemoryError, KeyboardInterrupt, SystemExit],
)
def test_png_encoding_failures_have_stable_errors_and_no_named_artifacts(
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    directory = tmp_path / "private-call"
    directory.mkdir(mode=0o700)
    before_fds = set(os.listdir("/dev/fd"))

    class FailingPixmap:
        def tobytes(self, output: str) -> bytes:
            assert output == "png"
            raise error_type("synthetic PNG encoding failure")

    expected = (
        VisionOcrError
        if error_type in (RuntimeError, MemoryError)
        else error_type
    )
    with pytest.raises(expected, match="PNG|encoding"):
        VisionOcrEngine._write_anonymous_png(  # type: ignore[arg-type]
            FailingPixmap(),
            directory,
        )

    assert list(directory.iterdir()) == []
    assert set(os.listdir("/dev/fd")) == before_fds


def test_default_runner_kills_timed_out_process_group_and_cleans_temp_files(
) -> None:
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        vision_module._run_bounded_command(
            ["/bin/sleep", "5"],
            environment={"PATH": "/usr/bin:/bin"},
            cwd=Path("/"),
            timeout=0.05,
        )

    assert time.monotonic() - started < 2


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
    assert int(long_call.calls[0]["width"]) < MAXIMUM_LONG_EDGE_PIXELS
    assert int(long_call.calls[0]["height"]) < 300


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


def test_invokes_public_runner_with_anonymous_png_fd_contract(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    runner = RecordingRunner()

    _engine(tmp_path, runner).recognize_page(pdf, page_index=0)

    call = runner.calls[0]
    argv = call["argv"]
    assert argv == [
        str((tmp_path / "book-vision-ocr").absolute()),
        "--image",
        f"/dev/fd/{call['image_fd']}",
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
    assert call["pass_fds"] == (call["image_fd"],)
    assert call["has_executable"] is False
    assert call["temp_entries"] == []
    assert list((tmp_path / "ocr-temp").iterdir()) == []
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(int(call["image_fd"]))


@pytest.mark.parametrize("bad_fd", [True, -1, 999_999, "3"])
def test_run_request_rejects_bool_negative_closed_and_non_integer_fds(
    bad_fd: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="pass_fds|descriptor"):
        RunRequest(
            argv=("/absolute/helper",),
            environment=(("PATH", "/usr/bin:/bin"),),
            cwd=Path("/"),
            timeout=120.0,
            pass_fds=(bad_fd,),  # type: ignore[arg-type]
        )


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

    assert list((tmp_path / "ocr-temp").iterdir()) == []
    if runner.calls:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(int(runner.calls[0]["image_fd"]))


def test_keyboard_interrupt_propagates_and_cleans_png(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    runner = RecordingRunner(error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        _engine(tmp_path, runner).recognize_page(pdf, page_index=0)

    assert list((tmp_path / "ocr-temp").iterdir()) == []
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(int(runner.calls[0]["image_fd"]))


@pytest.mark.parametrize("outcome", ["success", "failure", "interrupt"])
def test_all_ocr_descriptors_close_on_every_outcome(
    tmp_path: Path,
    outcome: str,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    error: BaseException | None = None
    if outcome == "failure":
        error = subprocess.TimeoutExpired(["helper"], 120)
    elif outcome == "interrupt":
        error = KeyboardInterrupt()
    runner = RecordingRunner(error=error)
    before = set(os.listdir("/dev/fd"))

    if outcome == "success":
        _engine(tmp_path, runner).recognize_page(pdf, page_index=0)
    else:
        expected = KeyboardInterrupt if outcome == "interrupt" else VisionOcrError
        with pytest.raises(expected):
            _engine(tmp_path, runner).recognize_page(pdf, page_index=0)

    assert set(os.listdir("/dev/fd")) == before
    assert list((tmp_path / "ocr-temp").iterdir()) == []


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


def test_private_helper_snapshot_is_immune_to_in_place_source_rewrite(
    tmp_path: Path,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = _helper(tmp_path)
    original = helper.read_bytes()
    replacement = original.replace(b"exit 0", b"exit 9")
    before = helper.stat()

    class RewritingRunner(RecordingRunner):
        def __call__(self, request: RunRequest) -> subprocess.CompletedProcess[bytes | str]:
            helper.write_bytes(replacement)
            helper.chmod(stat.S_IMODE(before.st_mode))
            os.utime(helper, ns=(before.st_atime_ns, before.st_mtime_ns))
            return super().__call__(request)

    runner = RewritingRunner()
    with pytest.raises(VisionOcrError, match="helper changed|helper.*changed"):
        _engine(tmp_path, runner, helper=helper).recognize_page(pdf, page_index=0)

    assert helper.read_bytes() == replacement
    assert list((tmp_path / "ocr-temp").iterdir()) == []


def test_rejects_helper_larger_than_bounded_snapshot_limit(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = tmp_path / "oversized-helper"
    with helper.open("wb") as stream:
        stream.truncate(MAXIMUM_HELPER_BYTES + 1)
    helper.chmod(0o700)
    runner = RecordingRunner()

    with pytest.raises(VisionOcrError, match="helper.*size|helper.*large"):
        _engine(tmp_path, runner, helper=helper).recognize_page(pdf, page_index=0)

    assert runner.calls == []


@pytest.mark.parametrize("valid_output", [True, False])
def test_original_pdf_bytes_and_metadata_are_never_modified(
    tmp_path: Path,
    valid_output: bool,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf", pages=2)
    before_bytes = pdf.read_bytes()
    before = pdf.stat()
    runner = RecordingRunner(_payload() if valid_output else b"{")

    if valid_output:
        _engine(tmp_path, runner).recognize_page(pdf, page_index=1)
    else:
        with pytest.raises(VisionOcrError, match="JSON"):
            _engine(tmp_path, runner).recognize_page(pdf, page_index=1)

    after = pdf.stat()
    assert pdf.read_bytes() == before_bytes
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mode, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mode,
        before.st_mtime_ns,
    )


def test_deeply_nested_json_becomes_stable_vision_error(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    deeply_nested = b"[" * 10_000 + b"0" + b"]" * 10_000

    with pytest.raises(VisionOcrError, match="invalid JSON"):
        _engine(tmp_path, RecordingRunner(deeply_nested)).recognize_page(
            pdf, page_index=0
        )


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


def test_helper_execute_permission_uses_effective_owner_credentials(
    tmp_path: Path,
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf")
    helper = _helper(tmp_path)
    runner = RecordingRunner()

    helper.chmod(0o401)
    with pytest.raises(VisionOcrError, match="executable"):
        _engine(tmp_path, runner, helper=helper).recognize_page(pdf, page_index=0)
    assert runner.calls == []

    helper.chmod(0o500)
    result = _engine(tmp_path, runner, helper=helper).recognize_page(
        pdf, page_index=0
    )
    assert result.lines == ()
