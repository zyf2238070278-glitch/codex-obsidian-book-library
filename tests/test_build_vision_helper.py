from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts import build_vision_helper as vision_builder
from scripts.build_vision_helper import (
    VisionHelperBuildError,
    build_vision_helper,
)


MACHO_ARM64 = b"\xcf\xfa\xed\xfe" + (b"\x00" * 28)


class FakeRunner:
    def __init__(
        self,
        *,
        binary: bytes = MACHO_ARM64,
        architectures: str = "arm64\n",
        capabilities: object | None = None,
        failures: dict[str, tuple[int, str]] | None = None,
        make_output: bool = True,
    ) -> None:
        self.binary = binary
        self.architectures = architectures
        self.capabilities = (
            {
                "schema_version": 1,
                "languages": ["zh-Hans", "en-US"],
            }
            if capabilities is None
            else capabilities
        )
        self.failures = failures or {}
        self.make_output = make_output
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        assert type(argv) is list
        assert all(type(item) is str for item in argv)
        self.calls.append(list(argv))
        command = (
            Path(argv[1]).name
            if argv[:1] == ["xcrun"]
            else Path(argv[0]).name
        )
        failure = self.failures.get(command)
        if failure is not None:
            return subprocess.CompletedProcess(argv, failure[0], "", failure[1])
        if argv[:2] == ["xcrun", "swiftc"]:
            if self.make_output:
                Path(argv[argv.index("-o") + 1]).write_bytes(self.binary)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if command == "lipo":
            return subprocess.CompletedProcess(argv, 0, self.architectures, "")
        if argv[-1:] == ["--capabilities"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps(self.capabilities, allow_nan=True),
                "",
            )
        return subprocess.CompletedProcess(argv, 0, "", "")


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "native" / "main.swift"
    source.parent.mkdir(parents=True)
    source.write_text("// source fixture\n", encoding="utf-8")
    return source


def test_default_runner_is_non_shell_and_uses_writable_module_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(vision_builder.subprocess, "run", fake_run)

    result = vision_builder._default_run_command(["xcrun", "swiftc", "--version"])

    assert result.returncode == 0
    assert captured["argv"] == ["xcrun", "swiftc", "--version"]
    assert captured["shell"] is False
    assert captured["check"] is False
    assert captured["capture_output"] is True
    assert captured["text"] is True
    environment = captured["env"]
    assert type(environment) is dict
    clang_cache = Path(environment["CLANG_MODULE_CACHE_PATH"])
    swift_cache = Path(environment["SWIFT_MODULECACHE_PATH"])
    assert clang_cache == swift_cache
    assert clang_cache.is_absolute()
    assert clang_cache.is_dir()


def test_builder_uses_exact_non_shell_arm64_pipeline_and_atomically_installs(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "bin" / "book-vision-ocr"
    runner = FakeRunner()

    result = build_vision_helper(source=source, output=output, run_command=runner)

    assert result == output.resolve()
    assert output.read_bytes() == MACHO_ARM64
    assert stat.S_IMODE(output.stat().st_mode) == 0o755
    assert len(runner.calls) == 5
    compile_command = runner.calls[0]
    temporary_output = Path(compile_command[-1])
    assert compile_command == [
        "xcrun",
        "swiftc",
        "-O",
        "-target",
        "arm64-apple-macos13.0",
        str(source.resolve()),
        "-o",
        str(temporary_output),
    ]
    assert temporary_output != output
    assert runner.calls[1] == [
        "codesign",
        "--force",
        "--sign",
        "-",
        str(temporary_output),
    ]
    assert runner.calls[2] == ["lipo", "-archs", str(temporary_output)]
    assert runner.calls[3] == [
        "codesign",
        "--verify",
        "--strict",
        str(temporary_output),
    ]
    assert runner.calls[4] == [str(temporary_output), "--capabilities"]


def test_builder_rejects_existing_symlink_without_running_commands(tmp_path: Path) -> None:
    source = _source(tmp_path)
    target = tmp_path / "target"
    target.write_bytes(b"old")
    output = tmp_path / "book-vision-ocr"
    output.symlink_to(target)
    runner = FakeRunner()

    with pytest.raises(VisionHelperBuildError, match="symlink"):
        build_vision_helper(source=source, output=output, run_command=runner)

    assert runner.calls == []
    assert target.read_bytes() == b"old"


def test_builder_rejects_existing_non_regular_output(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "book-vision-ocr"
    output.mkdir()

    with pytest.raises(VisionHelperBuildError, match="regular file"):
        build_vision_helper(source=source, output=output, run_command=FakeRunner())


def test_builder_reports_failed_command_without_installing_partial_output(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "bin" / "book-vision-ocr"
    runner = FakeRunner(failures={"swiftc": (2, "synthetic compile failure")})

    with pytest.raises(
        VisionHelperBuildError,
        match="swiftc.*synthetic compile failure",
    ):
        build_vision_helper(source=source, output=output, run_command=runner)

    assert not output.exists()


def test_builder_requires_compiler_to_create_regular_macho_file(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "book-vision-ocr"

    with pytest.raises(VisionHelperBuildError, match="did not create"):
        build_vision_helper(
            source=source,
            output=output,
            run_command=FakeRunner(make_output=False),
        )

    with pytest.raises(VisionHelperBuildError, match="Mach-O"):
        build_vision_helper(
            source=source,
            output=output,
            run_command=FakeRunner(binary=b"not a macho"),
        )


@pytest.mark.parametrize("architectures", ["x86_64\n", "arm64 x86_64\n", "\n"])
def test_builder_accepts_only_thin_arm64_binary(
    tmp_path: Path,
    architectures: str,
) -> None:
    source = _source(tmp_path)

    with pytest.raises(VisionHelperBuildError, match="arm64"):
        build_vision_helper(
            source=source,
            output=tmp_path / "book-vision-ocr",
            run_command=FakeRunner(architectures=architectures),
        )


@pytest.mark.parametrize(
    ("capabilities", "message"),
    [
        ({"schema_version": True, "languages": ["zh-Hans", "en-US"]}, "schema"),
        ({"schema_version": 1.0, "languages": ["zh-Hans", "en-US"]}, "schema"),
        ({"schema_version": 1, "languages": ["zh-Hans"]}, "en-US"),
        ({"schema_version": 1, "languages": ["en-US"]}, "zh-Hans"),
        ({"schema_version": 1, "languages": "zh-Hans,en-US"}, "languages"),
        (
            {
                "schema_version": 1,
                "languages": ["zh-Hans", "en-US"],
                "extra": False,
            },
            "fields",
        ),
        ({"languages": ["zh-Hans", "en-US"]}, "fields"),
        ({"schema_version": 1, "languages": ["zh-Hans", 7]}, "languages"),
        (
            {"schema_version": 1, "languages": ["zh-Hans", "en-US", "en-US"]},
            "duplicate",
        ),
    ],
)
def test_builder_strictly_validates_capabilities(
    tmp_path: Path,
    capabilities: object,
    message: str,
) -> None:
    source = _source(tmp_path)

    with pytest.raises(VisionHelperBuildError, match=message):
        build_vision_helper(
            source=source,
            output=tmp_path / "book-vision-ocr",
            run_command=FakeRunner(capabilities=capabilities),
        )


def test_builder_rejects_non_json_capabilities(tmp_path: Path) -> None:
    class BadJsonRunner(FakeRunner):
        def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
            result = super().__call__(argv)
            if argv[-1:] == ["--capabilities"]:
                return subprocess.CompletedProcess(argv, 0, "not-json", "")
            return result

    with pytest.raises(VisionHelperBuildError, match="capabilities JSON"):
        build_vision_helper(
            source=_source(tmp_path),
            output=tmp_path / "book-vision-ocr",
            run_command=BadJsonRunner(),
        )


def test_builder_rejects_successful_capabilities_with_stderr(tmp_path: Path) -> None:
    class NoisyRunner(FakeRunner):
        def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
            result = super().__call__(argv)
            if argv[-1:] == ["--capabilities"]:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    result.stdout,
                    "unexpected warning",
                )
            return result

    with pytest.raises(VisionHelperBuildError, match="capabilities.*stderr"):
        build_vision_helper(
            source=_source(tmp_path),
            output=tmp_path / "book-vision-ocr",
            run_command=NoisyRunner(),
        )


def test_builder_rejects_non_finite_json_constant(tmp_path: Path) -> None:
    class NanRunner(FakeRunner):
        def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
            result = super().__call__(argv)
            if argv[-1:] == ["--capabilities"]:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    '{"schema_version":1,"languages":["zh-Hans","en-US"],"x":NaN}',
                    "",
                )
            return result

    with pytest.raises(VisionHelperBuildError, match="finite JSON"):
        build_vision_helper(
            source=_source(tmp_path),
            output=tmp_path / "book-vision-ocr",
            run_command=NanRunner(),
        )


def test_builder_validates_argument_types_and_source(tmp_path: Path) -> None:
    output = tmp_path / "book-vision-ocr"

    with pytest.raises(VisionHelperBuildError, match="source"):
        build_vision_helper(
            source=tmp_path / "missing.swift",
            output=output,
            run_command=FakeRunner(),
        )
    with pytest.raises(TypeError, match="source"):
        build_vision_helper(  # type: ignore[arg-type]
            source="main.swift",
            output=output,
            run_command=FakeRunner(),
        )


def test_builder_preserves_existing_output_when_validation_fails(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "book-vision-ocr"
    output.write_bytes(b"previous helper")
    output.chmod(0o755)

    with pytest.raises(VisionHelperBuildError, match="arm64"):
        build_vision_helper(
            source=source,
            output=output,
            run_command=FakeRunner(architectures="x86_64\n"),
        )

    assert output.read_bytes() == b"previous helper"
    assert stat.S_IMODE(output.stat().st_mode) == 0o755
