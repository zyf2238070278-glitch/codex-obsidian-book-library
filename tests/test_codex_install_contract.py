from __future__ import annotations

import hashlib
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV_SHA256 = "c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554"


def _tracked(path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    return result.returncode == 0


def _git_mode(path: str) -> str:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", path], cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    return result.stdout.split(maxsplit=1)[0]


def _macho_cpu_type(data: bytes) -> int:
    assert len(data) >= 32
    if data[:4] == b"\xcf\xfa\xed\xfe":
        endian = "<"
    elif data[:4] == b"\xfe\xed\xfa\xcf":
        endian = ">"
    else:
        raise AssertionError("expected a 64-bit Mach-O header")
    return struct.unpack_from(f"{endian}I", data, 4)[0]


def test_macho_cpu_type_parser_distinguishes_x86_64() -> None:
    x86_64_header = bytes.fromhex(
        "cffaedfe"  # MH_MAGIC_64, little-endian
        "07000001"  # CPU_TYPE_X86_64
        "03000000"  # CPU_SUBTYPE_X86_64_ALL
        "02000000"  # MH_EXECUTE
        "00000000"  # ncmds
        "00000000"  # sizeofcmds
        "00000000"  # flags
        "00000000"  # reserved
    )
    assert _macho_cpu_type(x86_64_header) == 0x01000007


def test_git_clone_contains_both_bootstrap_executables() -> None:
    uv = ROOT / "bin" / "uv"
    helper = ROOT / "bin" / "book-vision-ocr"
    assert _tracked("bin/uv")
    assert _tracked("bin/book-vision-ocr")
    assert _git_mode("bin/uv") == "100755"
    assert _git_mode("bin/book-vision-ocr") == "100755"
    assert uv.stat().st_mode & 0o111
    assert helper.stat().st_mode & 0o111
    assert hashlib.sha256(uv.read_bytes()).hexdigest() == UV_SHA256
    assert _macho_cpu_type(uv.read_bytes()) == 0x0100000C
    assert _macho_cpu_type(helper.read_bytes()) == 0x0100000C


def test_machine_specific_codex_config_is_generated_not_tracked() -> None:
    assert not _tracked(".codex/config.toml")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/.codex/config.toml" in ignore
    assert "/Obsidian书库/" in ignore


def test_agents_gives_codex_one_install_route() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    required = (
        "首次安装与修复", "./install-macos.command", "完整退出并重启 Codex",
        "library_status", "不要自行拼接另一套 Python、pip、uv 或模型下载命令",
    )
    assert all(item in text for item in required)


def test_readme_has_the_four_step_codex_flow() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    ordered = (
        "git clone https://github.com/zyf2238070278-glitch/codex-obsidian-book-library.git",
        "请安装并检查这个书库", "完整退出并重启 Codex", "检查书库状态",
    )
    positions = [text.index(item) for item in ordered]
    assert positions == sorted(positions)
