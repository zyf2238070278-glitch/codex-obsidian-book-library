#!/bin/bash
set -u
case "$0" in */*) script_directory=${0%/*} ;; *) script_directory=. ;; esac
PROJECT_ROOT="$(CDPATH= cd -- "$script_directory" && pwd -P)" || exit 1
exec "$PROJECT_ROOT/install-macos.command" "$@"
