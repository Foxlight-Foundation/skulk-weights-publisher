#!/usr/bin/env bash
set -euo pipefail

missing=0

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "missing command: $name" >&2
    missing=1
    return
  fi
  echo "found command: $name ($(command -v "$name"))"
}

require_command larql

if ! python3 - <<'PY' >/dev/null 2>&1
import yaml
PY
then
  echo "missing Python package: PyYAML (install with: python3 -m pip install -r requirements.txt)" >&2
  missing=1
else
  echo "found Python package: PyYAML"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set; publication will fail until the GitHub secret is configured" >&2
else
  echo "HF_TOKEN is set"
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
fi

if [ "$missing" -ne 0 ]; then
  exit 1
fi

echo "doctor checks passed"
