"""Catalog source loading and merged catalog validation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from skulk_weights_publisher.defaults import (
    DEFAULT_FOXLIGHT_HF_OWNER,
    DEFAULT_FOXLIGHT_VINDEX_COLLECTION,
)
from skulk_weights_publisher.manifest import (
    HF_COLLECTION_PATTERN,
    NAMESPACE_PATTERN,
    ManifestEntry,
    ManifestError,
    VindexTier,
    validate_manifest,
    validate_manifest_payload,
)

DEFAULT_CONFIG_PATH = Path("skulk-weights.yaml")
BUILTIN_FOXLIGHT = "foxlight"
DEFAULT_CONFIG_TEXT = """# Foxlight entries are included automatically.
# Replace the empty list with your own catalog sources when you are ready.
catalogs: []

# Example operator source:
# catalogs:
#   - path: ./operator-vindexes.yaml
#     namespace: my-org
#     hf_owner: my-org
#     hf_collection: my-org/my-vindexes-0123456789abcdef01234567
"""

CatalogueSourceKind = Literal["builtin", "path", "manifest"]


@dataclass(frozen=True)
class CatalogueSource:
    """One source that contributes entries to the merged catalog."""

    name: str
    kind: CatalogueSourceKind
    namespace: str | None
    hf_owner: str | None
    hf_collection: str | None
    origin: str
    entries: tuple[ManifestEntry, ...]


@dataclass(frozen=True)
class CatalogueView:
    """A merged view of catalog entries from one or more sources."""

    sources: tuple[CatalogueSource, ...]
    entries: tuple[ManifestEntry, ...]


def load_catalogue_view(
    *,
    config_path: Path | None = None,
    manifest_path: Path | None = None,
) -> CatalogueView:
    """Load the effective catalog view from config, manifest, or built-ins."""

    if config_path is not None and manifest_path is not None:
        raise ManifestError("--config and --manifest cannot be used together")
    if manifest_path is not None:
        return _load_legacy_manifest_view(manifest_path)
    if config_path is not None:
        return _load_config_view(config_path)
    if DEFAULT_CONFIG_PATH.is_file():
        return _load_config_view(DEFAULT_CONFIG_PATH)
    return _merge_sources((_load_builtin_source(BUILTIN_FOXLIGHT),))


def find_catalogue_entry(key: str, view: CatalogueView) -> ManifestEntry:
    """Return the entry matching an effective catalog key."""

    for entry in view.entries:
        if entry.key == key:
            return entry
    raise ManifestError(f"catalog key not found: {key}")


def filter_catalogue_entries(
    view: CatalogueView,
    tier: Literal["all", "smoke", "moe"] = "all",
) -> tuple[ManifestEntry, ...]:
    """Return entries in ``view`` filtered by tier."""

    if tier == "all":
        return view.entries
    selected_tier = cast(VindexTier, tier)
    return tuple(entry for entry in view.entries if entry.tier == selected_tier)


def write_default_config(path: Path, *, force: bool = False) -> None:
    """Write a starter catalog config file."""

    if path.exists() and not force:
        raise ManifestError(f"{path} already exists; rerun with --force to replace it")
    path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")


def _load_legacy_manifest_view(path: Path) -> CatalogueView:
    entries = validate_manifest(path)
    source = CatalogueSource(
        name=path.stem,
        kind="manifest",
        namespace=None,
        hf_owner=None,
        hf_collection=None,
        origin=str(path),
        entries=entries,
    )
    return _merge_sources((source,))


def _load_builtin_source(name: str) -> CatalogueSource:
    if name != BUILTIN_FOXLIGHT:
        raise ManifestError(f"unsupported builtin catalog: {name}")
    payload = _load_builtin_payload(name)
    entries = validate_manifest_payload(
        payload,
        label=f"builtin:{name}",
        namespace="foxlight",
        hf_owner=DEFAULT_FOXLIGHT_HF_OWNER,
        hf_collection=DEFAULT_FOXLIGHT_VINDEX_COLLECTION,
    )
    return CatalogueSource(
        name=name,
        kind="builtin",
        namespace="foxlight",
        hf_owner=DEFAULT_FOXLIGHT_HF_OWNER,
        hf_collection=DEFAULT_FOXLIGHT_VINDEX_COLLECTION,
        origin=f"package:{name}",
        entries=entries,
    )


def _load_builtin_payload(name: str) -> dict[str, Any]:
    resource = resources.files("skulk_weights_publisher").joinpath(
        "catalogues", f"{name}.yaml"
    )
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestError(f"builtin:{name}: must contain a top-level mapping")
    return cast(dict[str, Any], payload)


def _load_config_view(path: Path) -> CatalogueView:
    payload = _load_config_payload(path)
    has_catalogs = "catalogs" in payload
    has_legacy_catalogues = "catalogues" in payload
    if has_catalogs and has_legacy_catalogues:
        raise ManifestError(f"{path}: set only one of catalogs or the legacy key")
    raw_sources = payload.get("catalogs", payload.get("catalogues", []))
    if not isinstance(raw_sources, list):
        raise ManifestError(f"{path}: catalogs must be a list")
    sources: list[CatalogueSource] = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ManifestError(f"{path}: catalogs[{index}] must be a mapping")
        source = cast(dict[str, Any], raw_source)
        sources.append(_load_config_source(path, index, source))
    if not any(
        source.kind == "builtin" and source.name == BUILTIN_FOXLIGHT
        for source in sources
    ):
        sources.insert(0, _load_builtin_source(BUILTIN_FOXLIGHT))
    return _merge_sources(tuple(sources))


def _load_config_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"{path} not found")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestError(f"{path}: must contain a top-level mapping")
    return cast(dict[str, Any], payload)


def _load_config_source(
    config_path: Path,
    index: int,
    source: dict[str, Any],
) -> CatalogueSource:
    has_builtin = "builtin" in source
    has_path = "path" in source
    if has_builtin == has_path:
        raise ManifestError(
            f"{config_path}: catalogs[{index}] must set exactly one of builtin or path"
        )
    if has_builtin:
        builtin = _require_source_string(config_path, index, source, "builtin")
        if set(source) != {"builtin"}:
            raise ManifestError(
                f"{config_path}: catalogs[{index}] builtin sources do not "
                "accept overrides"
            )
        return _load_builtin_source(builtin)

    extra_fields = set(source) - {"path", "namespace", "hf_owner", "hf_collection"}
    if extra_fields:
        raise ManifestError(
            f"{config_path}: catalogs[{index}] unsupported fields "
            f"{sorted(extra_fields)}"
        )
    source_path = _resolve_source_path(
        config_path,
        _require_source_string(config_path, index, source, "path"),
    )
    namespace = _require_source_string(config_path, index, source, "namespace")
    hf_owner = _require_source_string(config_path, index, source, "hf_owner")
    hf_collection = _optional_source_string(
        config_path,
        index,
        source,
        "hf_collection",
    )
    if not NAMESPACE_PATTERN.fullmatch(namespace):
        raise ManifestError(
            f"{config_path}: catalogs[{index}] namespace must be lowercase "
            "kebab/dot-case"
        )
    if hf_collection is not None:
        if not HF_COLLECTION_PATTERN.fullmatch(hf_collection):
            raise ManifestError(
                f"{config_path}: catalogs[{index}].hf_collection must look "
                "like owner/slug"
            )
        if hf_collection.split("/", maxsplit=1)[0] != hf_owner:
            raise ManifestError(
                f"{config_path}: catalogs[{index}].hf_collection owner must "
                f"be {hf_owner!r}"
            )
    entries = validate_manifest(
        source_path,
        namespace=namespace,
        hf_owner=hf_owner,
        hf_collection=hf_collection,
    )
    return CatalogueSource(
        name=namespace,
        kind="path",
        namespace=namespace,
        hf_owner=hf_owner,
        hf_collection=hf_collection,
        origin=str(source_path),
        entries=entries,
    )


def _require_source_string(
    config_path: Path,
    index: int,
    source: dict[str, Any],
    field: str,
) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(
            f"{config_path}: catalogs[{index}].{field} must be a non-empty string"
        )
    return value


def _optional_source_string(
    config_path: Path,
    index: int,
    source: dict[str, Any],
    field: str,
) -> str | None:
    value = source.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(
            f"{config_path}: catalogs[{index}].{field} must be a non-empty string"
        )
    return value


def _resolve_source_path(config_path: Path, source_path: str) -> Path:
    path = Path(source_path).expanduser()
    if path.is_absolute():
        return path
    return config_path.parent / path


def _merge_sources(sources: tuple[CatalogueSource, ...]) -> CatalogueView:
    seen_keys: set[str] = set()
    seen_outputs: set[str] = set()
    seen_repos: set[str] = set()
    entries: list[ManifestEntry] = []
    for source in sources:
        for entry in source.entries:
            if entry.key in seen_keys:
                raise ManifestError(f"{entry.key}: duplicate catalog key")
            seen_keys.add(entry.key)
            if entry.output_name in seen_outputs:
                raise ManifestError(
                    f"{entry.key}: duplicate output_name {entry.output_name}"
                )
            seen_outputs.add(entry.output_name)
            if entry.hf_repo in seen_repos:
                raise ManifestError(f"{entry.key}: duplicate hf_repo {entry.hf_repo}")
            seen_repos.add(entry.hf_repo)
            entries.append(entry)
    return CatalogueView(sources=sources, entries=tuple(entries))
