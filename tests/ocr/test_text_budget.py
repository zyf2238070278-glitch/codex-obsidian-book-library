from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_BUDGET_SOURCE = (
    PROJECT_ROOT / "native" / "book_vision_ocr" / "TextBudget.swift"
)


@pytest.mark.macos_vision
def test_swift_text_budget_enforces_scalar_and_utf8_boundaries(
    tmp_path: Path,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("Swift text-budget harness requires macOS and Xcode")
    harness = tmp_path / "main.swift"
    harness.write_text(
        """
import Foundation

func expectLimit(_ body: () throws -> Void) {
    do {
        try body()
        fatalError("expected recognized text budget limit")
    } catch RecognizedTextBudgetError.limitExceeded {
        return
    } catch {
        fatalError("unexpected error: \\(error)")
    }
}

var defaults = RecognizedTextBudget()
try defaults.add(String(repeating: "😀", count: 100_000))
precondition(defaults.unicodeScalarCount == 100_000)
precondition(defaults.utf8ByteCount == 400_000)
expectLimit { try defaults.add("a") }
precondition(defaults.unicodeScalarCount == 100_000)
precondition(defaults.utf8ByteCount == 400_000)

var scalarOnly = RecognizedTextBudget(
    maximumUnicodeScalars: 100_000,
    maximumUTF8Bytes: Int.max
)
try scalarOnly.add(String(repeating: "a", count: 100_000))
expectLimit { try scalarOnly.add("b") }

var bytesOnly = RecognizedTextBudget(
    maximumUnicodeScalars: Int.max,
    maximumUTF8Bytes: 400_000
)
try bytesOnly.add(String(repeating: "😀", count: 100_000))
expectLimit { try bytesOnly.add("😀") }

print("text-budget-ok")
""".lstrip(),
        encoding="utf-8",
    )
    module_cache = tmp_path / "module-cache"
    module_cache.mkdir()
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "en_US.UTF-8",
        "TMPDIR": str(tmp_path),
        "CLANG_MODULE_CACHE_PATH": str(module_cache),
        "SWIFT_MODULECACHE_PATH": str(module_cache),
    }
    executable = tmp_path / "text-budget-test"

    compiled = subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            "-O",
            "-target",
            "arm64-apple-macos13.0",
            str(TEXT_BUDGET_SOURCE),
            str(harness),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert compiled.returncode == 0, compiled.stderr[-4_000:]
    completed = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr[-4_000:]
    assert completed.stdout == "text-budget-ok\n"
