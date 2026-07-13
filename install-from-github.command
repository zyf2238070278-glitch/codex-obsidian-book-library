#!/bin/bash

set -eu

TAG="v0.1.0-beta.1"
ARCHIVE="codex-obsidian-book-library-v0.1.0-beta.1-macos-arm64-all-in-one.zip"
TOP_LEVEL="codex-obsidian-book-library-v0.1.0-beta.1-macos-arm64"
INSTALL_DIR="${BOOK_LIBRARY_INSTALL_DIR:-$HOME/CodexBookLibrary}"
TEMP_DIR=""

cleanup() {
  if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
  fi
}
trap cleanup EXIT HUP INT TERM

fail() {
  printf '错误：%s\n' "$1" >&2
  exit 1
}

run_installer() {
  cd "$INSTALL_DIR"
  ./install-macos.command
}

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  fail "当前版本只支持 Apple 芯片（M1/M2/M3/M4）的 Mac。"
fi

if [ -x "$INSTALL_DIR/install-macos.command" ] \
  && [ -x "$INSTALL_DIR/bin/uv" ] \
  && [ -f "$INSTALL_DIR/data/models/model.safetensors" ]; then
  printf '检测到已有完整安装，正在检查并更新 Codex 配置……\n'
  run_installer
  printf '\n安装完成。项目位置：%s\n' "$INSTALL_DIR"
  printf '请完全退出并重启 Codex，然后打开并信任这个项目。\n'
  exit 0
fi

if [ -e "$INSTALL_DIR" ]; then
  fail "安装位置已存在但内容不完整：$INSTALL_DIR。请先移动或重命名该目录后重试。"
fi

REPOSITORY="${BOOK_LIBRARY_REPOSITORY:-}"
if [ -z "$REPOSITORY" ]; then
  command -v git >/dev/null 2>&1 || fail "找不到 git，无法识别 GitHub 仓库。"
  REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
  case "$REMOTE_URL" in
    https://github.com/*)
      REPOSITORY="${REMOTE_URL#https://github.com/}"
      ;;
    git@github.com:*)
      REPOSITORY="${REMOTE_URL#git@github.com:}"
      ;;
    *)
      fail "无法从 origin 识别 GitHub 仓库地址。请在克隆后的项目目录运行本脚本。"
      ;;
  esac
  REPOSITORY="${REPOSITORY%.git}"
fi

if ! printf '%s\n' "$REPOSITORY" | grep -Eq '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$'; then
  fail "GitHub 仓库格式不正确：$REPOSITORY"
fi

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/codex-book-library.XXXXXX")"
CHECKSUMS="$TEMP_DIR/SHA256SUMS"
DOWNLOADED_ARCHIVE="$TEMP_DIR/$ARCHIVE"
EXTRACTED="$TEMP_DIR/extracted"
BASE_URL="https://github.com/$REPOSITORY/releases/download/$TAG"

printf '正在下载校验文件……\n'
curl -fL --retry 3 --connect-timeout 15 -o "$CHECKSUMS" "$BASE_URL/SHA256SUMS"
printf '正在下载书库安装包（约 292 MB）……\n'
curl -fL --retry 3 --connect-timeout 15 -o "$DOWNLOADED_ARCHIVE" "$BASE_URL/$ARCHIVE"

EXPECTED_SHA="$(awk -v name="$ARCHIVE" '$2 == name { print $1; exit }' "$CHECKSUMS")"
if [ "${#EXPECTED_SHA}" -ne 64 ]; then
  fail "SHA-256 校验文件无效。"
fi
case "$EXPECTED_SHA" in
  *[!0-9a-f]*) fail "SHA-256 校验文件无效。" ;;
esac

ACTUAL_SHA="$(shasum -a 256 "$DOWNLOADED_ARCHIVE" | awk '{ print $1 }')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  fail "SHA-256 校验失败，安装包可能下载不完整，已停止安装。"
fi

printf '校验通过，正在解压……\n'
mkdir -p "$EXTRACTED"
ditto -x -k "$DOWNLOADED_ARCHIVE" "$EXTRACTED"
BUNDLE_DIR="$EXTRACTED/$TOP_LEVEL"

if [ ! -d "$BUNDLE_DIR" ] || [ -L "$BUNDLE_DIR" ] \
  || [ ! -x "$BUNDLE_DIR/install-macos.command" ] \
  || [ ! -x "$BUNDLE_DIR/bin/uv" ] \
  || [ ! -f "$BUNDLE_DIR/data/models/model.safetensors" ]; then
  fail "安装包结构不完整，已停止安装。"
fi

mkdir -p "$(dirname "$INSTALL_DIR")"
mv "$BUNDLE_DIR" "$INSTALL_DIR"

printf '正在配置本地书库和 Codex……\n'
run_installer

printf '\n安装完成。项目位置：%s\n' "$INSTALL_DIR"
printf '请完全退出并重启 Codex，然后打开并信任这个项目。\n'
