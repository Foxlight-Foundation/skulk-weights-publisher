"""Auto-detect and add a HuggingFace model to the Foxlight catalog."""

from __future__ import annotations

import json
import re
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Any


class CatalogAddError(Exception):
    """Raised when catalog add detection or writing fails."""


def parse_hf_model_id(model_or_url: str) -> str:
    """Extract owner/repo from a HF URL or bare owner/repo string."""
    s = model_or_url.strip().rstrip("/")
    if "huggingface.co/" in s:
        path = s.split("huggingface.co/")[-1]
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            raise CatalogAddError(f"cannot parse model ID from URL: {model_or_url!r}")
        return f"{parts[0]}/{parts[1]}"
    if "/" not in s:
        raise CatalogAddError(
            f"expected owner/repo or a huggingface.co URL, got: {model_or_url!r}"
        )
    return s


def fetch_hf_model_info(model_id: str, token: str | None = None) -> dict[str, Any]:
    """Fetch model card metadata from the HF Hub API."""
    url = f"https://huggingface.co/api/models/{model_id}"
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise CatalogAddError(
            f"HF API returned {exc.code} for {model_id} — check the model ID"
        ) from exc
    except Exception as exc:
        raise CatalogAddError(
            f"failed to fetch HF metadata for {model_id}: {exc}"
        ) from exc
    return data  # type: ignore[return-value]


def detect_base_model(info: dict[str, Any]) -> str | None:
    """Return the upstream BF16 base model from model card tags."""
    tags: list[str] = info.get("tags", [])
    for tag in tags:
        if tag.startswith("base_model:quantized:"):
            return tag.split("base_model:quantized:", 1)[1]
    for tag in tags:
        if tag.startswith("base_model:"):
            val = tag.split("base_model:", 1)[1]
            if "/" in val:
                return val
    return None


def detect_quant(info: dict[str, Any]) -> str:
    """Infer quant scheme from tags or model name. Returns 'q4k' or 'q8k'."""
    model_id: str = info.get("id", "")
    tags: list[str] = info.get("tags", [])
    combined = " ".join(tags).lower() + " " + model_id.lower()
    if "8-bit" in combined or "-8bit" in combined or "q8" in combined:
        return "q8k"
    return "q4k"


_MOE_ARCH_TAGS = {
    "moe",
    "mixtral",
    "qwen_moe",
    "qwen3_5_moe",
    "deepseek_v2",
    "deepseek_v3",
}


def detect_tier(info: dict[str, Any]) -> str:
    """Return 'moe' for mixture-of-experts architectures, else 'smoke'."""
    tags: list[str] = info.get("tags", [])
    for tag in tags:
        if tag.lower() in _MOE_ARCH_TAGS:
            return "moe"
    return "smoke"


def detect_mtp_keys(base_model: str, token: str | None = None) -> list[str]:
    """Return MTP tensor key names present in base_model's weight map.

    Fetches the sharded index JSON; falls back to empty list on any error
    (single-file layout requires downloading the whole checkpoint).
    """
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = (
        f"https://huggingface.co/{base_model}/resolve/main"
        "/model.safetensors.index.json"
    )
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            index = json.load(resp)
        weight_map: dict[str, str] = index.get("weight_map", {})
        return sorted(k for k in weight_map if k.startswith("mtp.") or ".mtp." in k)
    except Exception:
        return []


_STRIP_SUFFIXES = [
    "-4bit", "-8bit",
    "-mlx", "-optiq",
    "-instruct", "-it",
]


def derive_key_slug(model_id: str, quant: str) -> str:
    """Derive a kebab-case catalog key slug from a HF model ID."""
    repo = model_id.split("/")[-1]
    # Strip known MLX/quant qualifiers (case-insensitive)
    for suffix in _STRIP_SUFFIXES:
        repo = re.sub(re.escape(suffix), "", repo, flags=re.IGNORECASE)
    slug = repo.lower().replace(".", "-")
    slug = re.sub(r"-+", "-", slug).strip("-")
    quant_suffix = "q4-k" if quant == "q4k" else "q8-k"
    return f"{slug}-full-{quant_suffix}"


_FOXLIGHT_COLLECTION = "FoxlightAI/vindexes-6a124406dd5fb439c431b051"
_FOXLIGHT_HF_OWNER = "FoxlightAI"


def _base_model_slug(base_model: str) -> str:
    repo = base_model.split("/")[-1]
    slug = repo.lower().replace(".", "-")
    return re.sub(r"-+", "-", slug).strip("-")


def build_entry_block(
    *,
    key_slug: str,
    source_model: str,
    quant: str,
    tier: str,
    base_model: str | None,
    mtp_keys: list[str],
    namespace: str = "foxlight",
) -> str:
    """Return an indented YAML block (with leading blank line) for one entry."""
    hf_owner = _FOXLIGHT_HF_OWNER if namespace == "foxlight" else namespace
    lines = [
        f"  - key: {key_slug}",
        f"    source_model: {source_model}",
        f"    quant: {quant}",
        f"    tier: {tier}",
        "    slices:",
        "      - full",
        f"    output_name: {key_slug}.vindex",
        f"    hf_repo: {hf_owner}/{key_slug}-vindex",
        f"    hf_collection: {_FOXLIGHT_COLLECTION}",
    ]
    if mtp_keys and base_model:
        sidecar = f"{hf_owner}/{_base_model_slug(base_model)}-mtp"
        lines += [
            f"    mtp_source_repo: {base_model}",
            f"    mtp_sidecar_repo: {sidecar}",
            f"    mtp_quant: {quant}",
        ]
    return "\n" + "\n".join(lines) + "\n"


def find_builtin_catalog_path() -> Path:
    """Return the filesystem path of the packaged foxlight.yaml.

    Works in editable installs; the path resolves to the actual source file.
    """
    resource = resources.files("skulk_weights_publisher").joinpath(
        "catalogues", "foxlight.yaml"
    )
    return Path(str(resource))
