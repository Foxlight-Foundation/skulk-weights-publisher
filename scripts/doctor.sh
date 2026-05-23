#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

global_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config | --manifest)
      if [ "$#" -lt 2 ]; then
        echo "$1 requires a path" >&2
        exit 2
      fi
      global_args+=("$1" "$2")
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [ "${#global_args[@]}" -gt 0 ]; then
  exec python3 -m skulk_vindex_publisher.cli "${global_args[@]}" doctor "$@"
fi

exec python3 -m skulk_vindex_publisher.cli doctor "$@"
