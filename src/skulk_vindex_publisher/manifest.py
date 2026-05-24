"""Manifest schema and validation for Skulk vindex publishing."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

DEFAULT_MANIFEST_PATH = Path("models.yaml")
ALLOWED_QUANTS = {"q4k"}
ALLOWED_SLICES = {"full", "expert-server"}
ALLOWED_TIERS = {"smoke", "moe"}
KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
NAMESPACE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

VindexQuant = Literal["q4k"]
VindexSlice = Literal["full", "expert-server"]
VindexTier = Literal["smoke", "moe"]


class ManifestError(ValueError):
    """Raised when the vindex catalogue is malformed."""


@dataclass(frozen=True)
class ManifestEntry:
    """One publishable vindex entry from a catalogue source."""

    key: str
    source_model: str
    quant: VindexQuant
    tier: VindexTier
    slices: tuple[VindexSlice, ...]
    output_name: str
    hf_repo: str

    @property
    def publish_slices(self) -> str:
        """Return the LARQL publish ``--slices`` argument for this entry."""

        if self.slices == ("full",):
            return "none"
        return ",".join(self.slices)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this entry."""

        payload = asdict(self)
        payload["slices"] = list(self.slices)
        return payload

    def to_json(self) -> str:
        """Serialize this entry as stable compact JSON."""

        return json.dumps(self.to_dict(), sort_keys=True)


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"{path} not found; run from the repository root")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestError(f"{path} must contain a top-level mapping")
    return cast(dict[str, Any], payload)


def _require_string(entry: dict[str, Any], field: str, key: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{key}: {field} must be a non-empty string")
    return value


def validate_manifest_payload(
    payload: dict[str, Any],
    *,
    label: str,
    namespace: str | None = None,
    hf_owner: str | None = None,
) -> tuple[ManifestEntry, ...]:
    """Return validated manifest entries from an already loaded payload."""

    if namespace is not None and not NAMESPACE_PATTERN.fullmatch(namespace):
        raise ManifestError(f"{label}: namespace must be lowercase kebab/dot-case")
    if hf_owner is not None and not hf_owner.strip():
        raise ManifestError(f"{label}: hf_owner must be a non-empty string")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ManifestError(f"{label}: models must be a non-empty list")

    seen_keys: set[str] = set()
    seen_outputs: set[str] = set()
    seen_repos: set[str] = set()
    entries: list[ManifestEntry] = []

    for index, raw_entry in enumerate(raw_models):
        if not isinstance(raw_entry, dict):
            raise ManifestError(f"models[{index}] must be a mapping")
        entry = cast(dict[str, Any], raw_entry)

        key = _require_string(entry, "key", f"models[{index}]")
        if not KEY_PATTERN.fullmatch(key):
            raise ManifestError(f"{key}: key must be lowercase kebab-case")
        effective_key = f"{namespace}/{key}" if namespace else key
        if effective_key in seen_keys:
            raise ManifestError(f"{effective_key}: duplicate key")
        seen_keys.add(effective_key)

        source_model = _require_string(entry, "source_model", effective_key)
        if "/" not in source_model:
            raise ManifestError(
                f"{effective_key}: source_model must look like owner/name"
            )

        quant = _require_string(entry, "quant", effective_key)
        if quant not in ALLOWED_QUANTS:
            raise ManifestError(f"{effective_key}: unsupported quant {quant!r}")

        tier = _require_string(entry, "tier", effective_key)
        if tier not in ALLOWED_TIERS:
            raise ManifestError(f"{effective_key}: unsupported tier {tier!r}")

        raw_slices = entry.get("slices")
        if not isinstance(raw_slices, list) or not raw_slices:
            raise ManifestError(f"{effective_key}: slices must be a non-empty list")
        if any(not isinstance(slice_name, str) for slice_name in raw_slices):
            raise ManifestError(
                f"{effective_key}: slices must contain only strings"
            )
        slices = tuple(cast(list[str], raw_slices))
        unknown_slices = set(slices) - ALLOWED_SLICES
        if unknown_slices:
            raise ManifestError(
                f"{effective_key}: unsupported slices {sorted(unknown_slices)}"
            )
        if "full" in slices and len(slices) > 1:
            raise ManifestError(
                f"{effective_key}: full must not be combined with other slices"
            )

        output_name = _require_string(entry, "output_name", effective_key)
        if "/" in output_name or not output_name.endswith(".vindex"):
            raise ManifestError(
                f"{effective_key}: output_name must be a .vindex basename"
            )
        if output_name in seen_outputs:
            raise ManifestError(
                f"{effective_key}: duplicate output_name {output_name}"
            )
        seen_outputs.add(output_name)

        hf_repo = _require_string(entry, "hf_repo", effective_key)
        if not HF_REPO_PATTERN.fullmatch(hf_repo):
            raise ManifestError(f"{effective_key}: hf_repo must look like owner/name")
        if hf_owner is not None and hf_repo.split("/", maxsplit=1)[0] != hf_owner:
            raise ManifestError(
                f"{effective_key}: hf_repo owner must be {hf_owner!r}"
            )
        if hf_repo in seen_repos:
            raise ManifestError(f"{effective_key}: duplicate hf_repo {hf_repo}")
        seen_repos.add(hf_repo)

        entries.append(
            ManifestEntry(
                key=effective_key,
                source_model=source_model,
                quant=cast(VindexQuant, quant),
                tier=cast(VindexTier, tier),
                slices=cast(tuple[VindexSlice, ...], slices),
                output_name=output_name,
                hf_repo=hf_repo,
            )
        )

    return tuple(entries)


def validate_manifest(
    path: Path = DEFAULT_MANIFEST_PATH,
    *,
    namespace: str | None = None,
    hf_owner: str | None = None,
) -> tuple[ManifestEntry, ...]:
    """Return validated manifest entries or raise a descriptive error."""

    return validate_manifest_payload(
        _load_payload(path),
        label=str(path),
        namespace=namespace,
        hf_owner=hf_owner,
    )


def find_entry(
    key: str,
    path: Path = DEFAULT_MANIFEST_PATH,
    *,
    namespace: str | None = None,
    hf_owner: str | None = None,
) -> ManifestEntry:
    """Return the manifest entry matching ``key``."""

    for entry in validate_manifest(path, namespace=namespace, hf_owner=hf_owner):
        if entry.key == key:
            return entry
    raise ManifestError(f"model key not found in {path}: {key}")


def list_entries(
    tier: Literal["all", "smoke", "moe"] = "all",
    path: Path = DEFAULT_MANIFEST_PATH,
    *,
    namespace: str | None = None,
    hf_owner: str | None = None,
) -> tuple[ManifestEntry, ...]:
    """Return manifest entries filtered by publication tier."""

    entries = validate_manifest(path, namespace=namespace, hf_owner=hf_owner)
    if tier == "all":
        return entries
    return tuple(entry for entry in entries if entry.tier == tier)
