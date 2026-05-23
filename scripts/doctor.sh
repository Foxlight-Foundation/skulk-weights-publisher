#!/usr/bin/env bash
set -euo pipefail

missing=0
publish_checks=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --publish)
      publish_checks=1
      shift
      ;;
    -h|--help)
      echo "Usage: scripts/doctor.sh [--publish]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "Usage: scripts/doctor.sh [--publish]" >&2
      exit 2
      ;;
  esac
done

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "missing command: $name" >&2
    missing=1
    return
  fi
  echo "found command: $name ($(command -v "$name"))"
}

if [ "$publish_checks" -eq 1 ]; then
  require_command larql
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import yaml
PY
then
  echo "missing Python package: PyYAML (install with: python3 -m pip install -r requirements.txt)" >&2
  missing=1
else
  echo "found Python package: PyYAML"
fi

if [ "$publish_checks" -eq 1 ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set; publication will fail until the GitHub secret is configured" >&2
    missing=1
  else
    echo "HF_TOKEN is set"
  fi
fi

scratch_root="${SKULK_VINDEX_SCRATCH:-$PWD/.scratch}"
mkdir -p "$scratch_root"

if [ ! -w "$scratch_root" ]; then
  echo "scratch root is not writable: $scratch_root" >&2
  missing=1
else
  echo "scratch root writable: $scratch_root"
fi

if [ ! -f models.yaml ]; then
  echo "models.yaml not found; run from the repository root" >&2
  missing=1
elif ! python3 scripts/manifest.py validate; then
  missing=1
fi

if [ "$missing" -ne 0 ]; then
  exit 1
fi

echo "doctor checks passed"
