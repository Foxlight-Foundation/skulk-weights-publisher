#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/publish-vindex.sh --model <manifest-key> [--dry-run]

Environment:
  SKULK_VINDEX_SCRATCH  Scratch root for extraction output (default: ./.scratch)
  HF_TOKEN              HuggingFace token used by LARQL/HF publication tooling
USAGE
}

model_key=""
dry_run=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      model_key="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$model_key" ]; then
  echo "--model is required" >&2
  usage >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to read models.yaml" >&2
  exit 1
fi

manifest_json="$(
  python3 - "$model_key" <<'PY'
import json
import pathlib
import sys

try:
    import yaml
except ModuleNotFoundError:
    print("PyYAML is required: python3 -m pip install pyyaml", file=sys.stderr)
    raise SystemExit(1)

model_key = sys.argv[1]
manifest = yaml.safe_load(pathlib.Path("models.yaml").read_text())
for entry in manifest.get("models", []):
    if entry.get("key") == model_key:
        print(json.dumps(entry))
        break
else:
    print(f"model key not found in models.yaml: {model_key}", file=sys.stderr)
    raise SystemExit(1)
PY
)"

source_model="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["source_model"])' "$manifest_json")"
quant="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["quant"])' "$manifest_json")"
output_name="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["output_name"])' "$manifest_json")"
hf_repo="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["hf_repo"])' "$manifest_json")"
slices="$(python3 -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["slices"]))' "$manifest_json")"

scratch_root="${SKULK_VINDEX_SCRATCH:-$PWD/.scratch}"
output_path="$scratch_root/$output_name"

publish_slices="$slices"
if [ "$publish_slices" = "full" ]; then
  publish_slices="none"
fi

extract_cmd=(larql extract "$source_model" -o "$output_path" --quant "$quant")
publish_cmd=(larql publish "$output_path" --repo "$hf_repo" --slices "$publish_slices")

echo "model key: $model_key"
echo "source model: $source_model"
echo "output path: $output_path"
echo "target repo: hf://$hf_repo"
echo "publish slices: $publish_slices"
printf 'extract command:'
printf ' %q' "${extract_cmd[@]}"
printf '\n'
printf 'publish command:'
printf ' %q' "${publish_cmd[@]}"
printf '\n'

if [ "$dry_run" -eq 1 ]; then
  echo "dry run complete"
  exit 0
fi

if ! command -v larql >/dev/null 2>&1; then
  echo "larql is required for non-dry-run publishing" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required for non-dry-run publishing" >&2
  exit 1
fi

mkdir -p "$scratch_root"
"${extract_cmd[@]}"
"${publish_cmd[@]}"
