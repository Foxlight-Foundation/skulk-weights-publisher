"""Auto-detect and add a HuggingFace model to the Foxlight catalog."""

from __future__ import annotations

import json
import re
import urllib.error
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


def resolve_base_model(
    info: dict[str, Any],
    token: str | None = None,
    max_depth: int = 5,
) -> str | None:
    """Return the true BF16 root model by traversing the base_model:quantized: chain.

    Many community MLX quants point their ``base_model:quantized:`` tag at a
    fine-tuned intermediate rather than the original base checkpoint.  This
    function follows the chain — fetching each intermediate's metadata — until
    it reaches a model that is not itself a quantization, then returns that
    model's ``base_model:`` target.

    Returns ``None`` when the leaf has no ``base_model:`` tag and is the same as
    the input model (i.e. the input is already a plain base with nothing to
    resolve). Only after at least one hop does a leaf's own ID become the
    answer (an intermediate with no further base tag).
    """
    current = info
    for _ in range(max_depth):
        tags: list[str] = current.get("tags", [])
        quantized_from: str | None = None
        for tag in tags:
            if tag.startswith("base_model:quantized:"):
                quantized_from = tag.split("base_model:quantized:", 1)[1]
                break

        if quantized_from is None:
            # Leaf node — not a quantization.  Return its own base_model or
            # (if it has none) its model ID, provided it differs from the
            # original model we started with.
            base = detect_base_model(current)
            if base is not None:
                return base
            current_id = current.get("id")
            if isinstance(current_id, str) and current_id != info.get("id"):
                return current_id
            return None

        try:
            current = fetch_hf_model_info(quantized_from, token=token)
        except CatalogAddError:
            # Can't fetch the intermediate — return it as the best available answer.
            return quantized_from

    return detect_base_model(current)


def detect_quant(info: dict[str, Any]) -> str:
    """Infer the **vindex** quant scheme from tags or model name ('q4k' or 'q8k').

    Drives the full-model vindex tier only — it does **not** feed MTP sidecar
    naming. MTP sidecars are quant-independent: one bf16 sidecar per base model
    serves every quantization of it (see :func:`mtp_extractor.extract_mtp`).
    """
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


def detect_assistant_model(model_id: str, token: str | None = None) -> str | None:
    """Return the companion assistant model repo if it exists on HuggingFace.

    Appends ``-assistant`` to ``model_id`` and performs a lightweight HEAD
    request against the HF API.  Returns the candidate ID on HTTP 200, or
    ``None`` on any error (404, network failure, etc.).

    Used to detect Gemma 4-style companion-assistant models where the
    assistant is already published separately by the model owner rather than
    embedded as ``mtp.*`` tensors in the base checkpoint.
    """
    candidate = f"{model_id}-assistant"
    url = f"https://huggingface.co/api/models/{candidate}"
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                return candidate
    except Exception:  # noqa: BLE001
        pass
    return None


def find_assistant_model(
    candidates: list[str | None], token: str | None = None
) -> str | None:
    """Return the first published ``-assistant`` repo among candidate model IDs.

    The Gemma 4 assistant is named after the *instruct* model
    (``google/gemma-4-26B-A4B-it`` → ``…-it-assistant``), which is usually the
    model the user pastes — not its base. So we check the pasted model itself
    first, then its immediate base, then the resolved BF16 root. Deduped; falls
    back to ``None``.
    """
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        assistant = detect_assistant_model(candidate, token=token)
        if assistant is not None:
            return assistant
    return None


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


_STRIP_SUFFIXES_KEY = ["-4bit", "-8bit", "-mlx", "-optiq", "-instruct", "-it"]
_STRIP_SUFFIXES_ARTIFACT = ["-4bit", "-8bit", "-mlx", "-optiq"]


def _make_slug(repo: str, *, strip_instruct: bool) -> str:
    suffixes = _STRIP_SUFFIXES_KEY if strip_instruct else _STRIP_SUFFIXES_ARTIFACT
    for suffix in suffixes:
        repo = re.sub(re.escape(suffix), "", repo, flags=re.IGNORECASE)
    slug = repo.lower().replace(".", "-").replace("_", "-")
    if not strip_instruct:
        slug = re.sub(r"-instruct\b", "-it", slug, flags=re.IGNORECASE)
    return re.sub(r"-+", "-", slug).strip("-")


def quant_suffix(quant: str) -> str:
    """Return the kebab-case quant suffix used in repo and artifact names."""
    return "q4-k" if quant == "q4k" else "q8-k"


def derive_key_slug(model_id: str, quant: str) -> str:
    """Derive a kebab-case catalog key slug (instruct/it stripped)."""
    slug = _make_slug(model_id.split("/")[-1], strip_instruct=True)
    return f"{slug}-full-{quant_suffix(quant)}"


def derive_artifact_slug(model_id: str, quant: str) -> str:
    """Derive a kebab-case artifact slug (instruct normalized to -it-)."""
    slug = _make_slug(model_id.split("/")[-1], strip_instruct=False)
    return f"{slug}-full-{quant_suffix(quant)}"


_FOXLIGHT_COLLECTION = "FoxlightAI/vindexes-6a124406dd5fb439c431b051"
_FOXLIGHT_HF_OWNER = "FoxlightAI"


def base_model_slug(base_model: str) -> str:
    """Derive a lowercase kebab-case slug from a base model ID (repo part only)."""
    repo = base_model.split("/")[-1]
    slug = repo.lower().replace(".", "-")
    return re.sub(r"-+", "-", slug).strip("-")


def build_entry_block(
    *,
    key_slug: str,
    artifact_slug: str | None = None,
    source_model: str,
    quant: str,
    tier: str,
    base_model: str | None,
    mtp_keys: list[str],
    assistant_model_repo: str | None = None,
    namespace: str = "foxlight",
) -> str:
    """Return an indented YAML block (with leading blank line) for one entry.

    ``artifact_slug`` is used for ``output_name`` and ``hf_repo``. It differs
    from ``key_slug`` when the source model has an instruct/it qualifier (which
    the catalog key omits but artifact names retain as ``-it-``). Defaults to
    ``key_slug`` when not supplied.

    When ``assistant_model_repo`` is provided the block includes
    ``assistant_model_repo`` and omits the ``mtp_*`` fields (mutually exclusive
    per the catalog schema).
    """
    hf_owner = _FOXLIGHT_HF_OWNER if namespace == "foxlight" else namespace
    aslug = artifact_slug if artifact_slug is not None else key_slug
    lines = [
        f"  - key: {key_slug}",
        f"    source_model: {source_model}",
        f"    quant: {quant}",
        f"    tier: {tier}",
        "    slices:",
        "      - full",
        f"    output_name: {aslug}.vindex",
        f"    hf_repo: {hf_owner}/{aslug}-vindex",
        f"    hf_collection: {_FOXLIGHT_COLLECTION}",
    ]
    if assistant_model_repo:
        lines.append(f"    assistant_model_repo: {assistant_model_repo}")
    elif mtp_keys and base_model:
        # One bf16 sidecar per base model — quant-independent, no quant suffix.
        sidecar = f"{hf_owner}/{base_model_slug(base_model)}-mtp"
        lines += [
            f"    mtp_source_repo: {base_model}",
            f"    mtp_sidecar_repo: {sidecar}",
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
