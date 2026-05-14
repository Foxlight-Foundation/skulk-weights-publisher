#!/usr/bin/env python3
"""Validate and query the Skulk vindex publisher manifest."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Literal

import yaml

MANIFEST_PATH = pathlib.Path("models.yaml")
ALLOWED_QUANTS = {"q4k"}
ALLOWED_SLICES = {"full", "expert-server"}
ALLOWED_TIERS = {"smoke", "moe"}
KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

ManifestEntry = dict[str, Any]


def _load_manifest() -> list[ManifestEntry]:
    if not MANIFEST_PATH.is_file():
        raise ValueError("models.yaml not found; run from the repository root")
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("models.yaml must contain a top-level mapping")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models.yaml must contain a non-empty models list")
    return models


def _require_string(entry: ManifestEntry, field: str, key: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key}: {field} must be a non-empty string")
    return value


def validate_manifest() -> list[ManifestEntry]:
    """Return validated manifest entries or raise a descriptive error."""

    models = _load_manifest()
    seen_keys: set[str] = set()
    seen_outputs: set[str] = set()
    seen_repos: set[str] = set()
    validated: list[ManifestEntry] = []

    for index, raw_entry in enumerate(models):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"models[{index}] must be a mapping")

        key = _require_string(raw_entry, "key", f"models[{index}]")
        if not KEY_PATTERN.fullmatch(key):
            raise ValueError(f"{key}: key must be lowercase kebab-case")
        if key in seen_keys:
            raise ValueError(f"{key}: duplicate key")
        seen_keys.add(key)

        source_model = _require_string(raw_entry, "source_model", key)
        if "/" not in source_model:
            raise ValueError(f"{key}: source_model must look like owner/name")

        quant = _require_string(raw_entry, "quant", key)
        if quant not in ALLOWED_QUANTS:
            raise ValueError(f"{key}: unsupported quant {quant!r}")

        tier = _require_string(raw_entry, "tier", key)
        if tier not in ALLOWED_TIERS:
            raise ValueError(f"{key}: unsupported tier {tier!r}")

        slices = raw_entry.get("slices")
        if not isinstance(slices, list) or not slices:
            raise ValueError(f"{key}: slices must be a non-empty list")
        if any(not isinstance(slice_name, str) for slice_name in slices):
            raise ValueError(f"{key}: slices must contain only strings")
        unknown_slices = set(slices) - ALLOWED_SLICES
        if unknown_slices:
            raise ValueError(f"{key}: unsupported slices {sorted(unknown_slices)}")
        if "full" in slices and len(slices) > 1:
            raise ValueError(f"{key}: full must not be combined with other slices")

        output_name = _require_string(raw_entry, "output_name", key)
        if "/" in output_name or not output_name.endswith(".vindex"):
            raise ValueError(f"{key}: output_name must be a .vindex basename")
        if output_name in seen_outputs:
            raise ValueError(f"{key}: duplicate output_name {output_name}")
        seen_outputs.add(output_name)

        hf_repo = _require_string(raw_entry, "hf_repo", key)
        if not HF_REPO_PATTERN.fullmatch(hf_repo):
            raise ValueError(f"{key}: hf_repo must look like owner/name")
        if hf_repo in seen_repos:
            raise ValueError(f"{key}: duplicate hf_repo {hf_repo}")
        seen_repos.add(hf_repo)

        validated.append(
            {
                "key": key,
                "source_model": source_model,
                "quant": quant,
                "tier": tier,
                "slices": slices,
                "output_name": output_name,
                "hf_repo": hf_repo,
            }
        )

    return validated


def _find_entry(key: str) -> ManifestEntry:
    for entry in validate_manifest():
        if entry["key"] == key:
            return entry
    raise ValueError(f"model key not found in models.yaml: {key}")


def _cmd_validate(_args: argparse.Namespace) -> int:
    entries = validate_manifest()
    print(f"models.yaml valid: {len(entries)} entries")
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    entry = _find_entry(args.key)
    print(json.dumps(entry, sort_keys=True))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    entries = validate_manifest()
    tier: Literal["all", "smoke", "moe"] = args.tier
    for entry in entries:
        if tier == "all" or entry["tier"] == tier:
            print(entry["key"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.set_defaults(func=_cmd_validate)

    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("--key", required=True)
    get_parser.set_defaults(func=_cmd_get)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--tier", choices=["all", "smoke", "moe"], default="all")
    list_parser.set_defaults(func=_cmd_list)

    args = parser.parse_args()
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
