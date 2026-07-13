#!/bin/bash

set -u

case "$0" in
    */*) script_directory=${0%/*} ;;
    *) script_directory=. ;;
esac

if ! PROJECT_ROOT="$(CDPATH= cd -- "$script_directory" && pwd -P)"; then
    printf '%s\n' "安装失败：无法确定安装包目录。" >&2
    exit 1
fi

INSTALLER_SCRIPT="$PROJECT_ROOT/installer/install_macos.py"
BUNDLED_UV="$PROJECT_ROOT/bin/uv"
status=0

if [ ! -f "$INSTALLER_SCRIPT" ]; then
    printf '%s\n' "安装失败：安装包缺少 installer/install_macos.py。" >&2
    status=1
elif command -v python3 >/dev/null 2>&1; then
    if python3 "$INSTALLER_SCRIPT" --project-root "$PROJECT_ROOT" "$@"; then
        status=0
    else
        status=$?
    fi
elif [ -x "$BUNDLED_UV" ]; then
    if "$BUNDLED_UV" run --no-project --python 3.12 "$INSTALLER_SCRIPT" \
        --project-root "$PROJECT_ROOT" "$@"; then
        status=0
    else
        status=$?
    fi
else
    printf '%s\n' \
        "安装失败：未找到 python3，且安装包内缺少可执行的 bin/uv。" >&2
    printf '%s\n' \
        "请重新下载完整的 macOS Apple Silicon 安装包后再试。" >&2
    status=1
fi

if [ -t 0 ] && [ -t 1 ]; then
    printf '\n%s' "按回车关闭…"
    IFS= read -r _unused || true
fi

exit "$status"
