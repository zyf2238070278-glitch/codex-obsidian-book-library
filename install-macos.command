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

installer_arguments=("--project-root" "$PROJECT_ROOT")
vault_seen=0
config_seen=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --vault)
            if [ "$vault_seen" -ne 0 ]; then
                printf '%s\n' "安装失败：参数 --vault 不得重复。" >&2
                exit 2
            fi
            if [ "$#" -lt 2 ]; then
                printf '%s\n' "安装失败：参数 --vault 缺少值。" >&2
                exit 2
            fi
            case "$2" in
                ''|-*)
                    printf '%s\n' "安装失败：参数 --vault 缺少有效值。" >&2
                    exit 2
                    ;;
            esac
            installer_arguments+=("--vault" "$2")
            vault_seen=1
            shift 2
            ;;
        --codex-config)
            if [ "$config_seen" -ne 0 ]; then
                printf '%s\n' "安装失败：参数 --codex-config 不得重复。" >&2
                exit 2
            fi
            if [ "$#" -lt 2 ]; then
                printf '%s\n' "安装失败：参数 --codex-config 缺少值。" >&2
                exit 2
            fi
            case "$2" in
                ''|-*)
                    printf '%s\n' "安装失败：参数 --codex-config 缺少有效值。" >&2
                    exit 2
                    ;;
            esac
            installer_arguments+=("--codex-config" "$2")
            config_seen=1
            shift 2
            ;;
        *)
            printf '%s\n' "安装失败：不支持的参数：$1" >&2
            exit 2
            ;;
    esac
done

INSTALLER_SCRIPT="$PROJECT_ROOT/installer/install_macos.py"
BUNDLED_UV="$PROJECT_ROOT/bin/uv"
PINNED_UV_SHA256="c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554"
status=0

if [ "$(/usr/bin/uname -s)" != "Darwin" ]; then
    printf '%s\n' "安装失败：当前安装包仅支持 macOS。" >&2
    exit 1
fi

if [ "$(/usr/bin/uname -m)" != "arm64" ]; then
    printf '%s\n' "安装失败：当前安装包仅支持 Apple 芯片 Mac。" >&2
    exit 1
fi

macos_version="$(/usr/bin/sw_vers -productVersion)"
macos_major=${macos_version%%.*}
case "$macos_major" in
    ''|*[!0-9]*)
        printf '%s\n' "安装失败：无法识别 macOS 版本：$macos_version" >&2
        exit 1
        ;;
esac
if [ "$macos_major" -lt 16 ]; then
    printf '%s\n' "安装失败：需要 macOS 16 或更高版本，当前为 ${macos_version}。" >&2
    exit 1
fi

if [ ! -f "$INSTALLER_SCRIPT" ]; then
    printf '%s\n' "安装失败：安装包缺少 installer/install_macos.py。" >&2
    exit 1
fi

if [ ! -f "$BUNDLED_UV" ] || [ -L "$BUNDLED_UV" ] || [ ! -x "$BUNDLED_UV" ]; then
    printf '%s\n' "安装失败：安装包内的 bin/uv 缺失、不是普通文件或不可执行。" >&2
    exit 1
fi

if ! uv_digest_output="$(/usr/bin/shasum -a 256 "$BUNDLED_UV")"; then
    printf '%s\n' "安装失败：无法校验安装包内的 bin/uv。" >&2
    exit 1
fi
uv_digest=${uv_digest_output%% *}
if [ "$uv_digest" != "$PINNED_UV_SHA256" ]; then
    printf '%s\n' "安装失败：bin/uv 完整性校验失败，安装包可能已损坏或被篡改。" >&2
    exit 1
fi

if "$BUNDLED_UV" run --no-project --python 3.12 "$INSTALLER_SCRIPT" \
    "${installer_arguments[@]}"; then
    status=0
else
    status=$?
fi

if [ -t 0 ] && [ -t 1 ]; then
    printf '\n%s' "按回车关闭…"
    IFS= read -r _unused || true
fi

exit "$status"
