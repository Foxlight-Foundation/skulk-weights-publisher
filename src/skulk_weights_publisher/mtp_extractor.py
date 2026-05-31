"""MTP weight extraction and sidecar publishing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class MtpExtractionError(RuntimeError):
    """Raised when MTP extraction or publishing fails."""


def extract_mtp(
    source_repo: str,
    sidecar_repo: str,
    mtp_quant: str,
    scratch_root: Path,
    *,
    token: str | None,
    dry_run: bool = False,
) -> None:
    """Extract MTP weights from source_repo, quantize, and publish to sidecar_repo.

    Downloads only the safetensors shards that contain ``mtp.*`` keys, quantizes
    them to the specified quant scheme, saves as ``mtp.safetensors``, and uploads
    to ``sidecar_repo`` on Hugging Face.

    In dry-run mode prints the plan without downloading or uploading anything.
    """

    output_path = scratch_root / _sidecar_filename(sidecar_repo)

    if dry_run:
        _print_dry_run_plan(source_repo, sidecar_repo, mtp_quant, output_path)
        return

    try:
        from huggingface_hub import create_repo, hf_hub_download, upload_file
    except ImportError as exc:
        raise MtpExtractionError(
            "huggingface_hub is required for MTP extraction"
        ) from exc

    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MtpExtractionError(
            "mlx is required for MTP weight quantization"
        ) from exc

    # Verify/create the sidecar repo before the expensive download+quantize work.
    create_repo(
        sidecar_repo, repo_type="model", private=False, exist_ok=True, token=token
    )

    # Identify which shards contain mtp.* keys.
    cache_dir = str(scratch_root / "_hf_cache")
    shard_files = _find_mtp_shards(source_repo, token=token, cache_dir=cache_dir)
    if not shard_files:
        raise MtpExtractionError(
            f"no mtp.* keys found in {source_repo}; "
            "confirm this model has native MTP heads"
        )

    print(f"mtp: found mtp.* keys in {len(shard_files)} shard(s)", file=sys.stderr)

    # Download the relevant shards into scratch.
    scratch_root.mkdir(parents=True, exist_ok=True)
    local_shards: list[Path] = []
    for shard in shard_files:
        local = Path(
            hf_hub_download(
                repo_id=source_repo,
                filename=shard,
                token=token,
                cache_dir=cache_dir,
            )
        )
        local_shards.append(local)
        print(f"mtp: downloaded {shard}", file=sys.stderr)

    # Extract mtp.* tensors as mlx arrays.
    # mx.load handles bfloat16 natively; safe_open+numpy does not.
    mtp_tensors: dict[str, Any] = {}
    for shard_path in local_shards:
        shard_weights: dict[str, Any] = mx.load(str(shard_path))  # type: ignore[assignment]
        for key, arr in shard_weights.items():
            if key.startswith("mtp.") or ".mtp." in key:
                mtp_tensors[key] = arr

    if not mtp_tensors:
        raise MtpExtractionError(
            f"shards were downloaded but no mtp.* tensors could be read"
            f" from {source_repo}"
        )

    print(f"mtp: extracted {len(mtp_tensors)} tensor(s)", file=sys.stderr)

    # Quantize.
    quant_bits = _quant_bits(mtp_quant)
    quantized = _quantize(mtp_tensors, bits=quant_bits, mx=mx)
    print(f"mtp: quantized to {quant_bits}-bit", file=sys.stderr)

    # Save.
    mx.save_safetensors(str(output_path), quantized)
    print(f"mtp: saved to {output_path}", file=sys.stderr)

    # Upload.
    upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo="mtp.safetensors",
        repo_id=sidecar_repo,
        repo_type="model",
        token=token,
        commit_message=f"Add MTP sidecar from {source_repo} ({mtp_quant})",
    )
    print(f"mtp: published to hf://{sidecar_repo}/mtp.safetensors", file=sys.stderr)


def _find_mtp_shards(
    source_repo: str,
    *,
    token: str | None,
    cache_dir: str | None = None,
) -> list[str]:
    """Return the shard filenames in source_repo that contain mtp.* keys.

    Checks for a sharded index first; falls back to the single-file layout.
    Pass cache_dir so any single-file download lands in the same location that
    the caller will use for the full extraction, avoiding a double download.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    # Try sharded index first; only swallow 404 (no index file = single-file layout).
    try:
        index_path = hf_hub_download(
            repo_id=source_repo,
            filename="model.safetensors.index.json",
            token=token,
            cache_dir=cache_dir,
        )
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
        weight_map: dict[str, str] = index.get("weight_map", {})
        shards: set[str] = set()
        for tensor_key, shard_file in weight_map.items():
            if tensor_key.startswith("mtp.") or ".mtp." in tensor_key:
                shards.add(shard_file)
        return sorted(shards)
    except EntryNotFoundError:
        pass

    # Fall back to single-file layout — check if model.safetensors has mtp.* keys.
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MtpExtractionError(
            "safetensors is required for single-file model inspection"
        ) from exc

    try:
        single_path = hf_hub_download(
            repo_id=source_repo,
            filename="model.safetensors",
            token=token,
            cache_dir=cache_dir,
        )
    except EntryNotFoundError:
        return []

    with safe_open(single_path, framework="numpy") as f:
        if any(k.startswith("mtp.") or ".mtp." in k for k in f.keys()):  # noqa: SIM118
            return ["model.safetensors"]
    return []


def _quantize(
    tensors: dict[str, Any],
    *,
    bits: int,
    mx: Any,
) -> dict[str, Any]:
    """Quantize a dict of mlx arrays to the given bit width.

    Linear (weight-only) tensors are group-quantized; small tensors (biases,
    norms, embeddings) are kept in float16 to preserve accuracy.
    """
    import mlx.core as _mx  # type: ignore[import-not-found]

    result: dict[str, Any] = {}
    group_size = 64

    for name, arr in tensors.items():
        a = _mx.array(arr) if not isinstance(arr, _mx.array) else arr
        # Only quantize 2-D weight matrices whose last dim is divisible by group_size;
        # keep everything else in fp16 to avoid MLX quantize shape errors.
        if (
            a.ndim == 2
            and a.shape[0] >= group_size
            and a.shape[1] >= group_size
            and a.shape[1] % group_size == 0
        ):
            q, scales, biases = _mx.quantize(a, bits=bits, group_size=group_size)
            result[name] = q
            result[f"{name}_scales"] = scales
            result[f"{name}_biases"] = biases
        else:
            result[name] = a.astype(_mx.float16)

    return result


def _quant_bits(mtp_quant: str) -> int:
    """Map a quant scheme string to an integer bit width."""
    mapping = {"q4k": 4, "q8k": 8}
    if mtp_quant not in mapping:
        raise MtpExtractionError(f"unsupported mtp_quant for extraction: {mtp_quant!r}")
    return mapping[mtp_quant]


def _sidecar_filename(sidecar_repo: str) -> str:
    return sidecar_repo.replace("/", "--") + "-mtp.safetensors"


def _print_dry_run_plan(
    source_repo: str,
    sidecar_repo: str,
    mtp_quant: str,
    output_path: Path,
) -> None:
    lines = [
        f"mtp source repo:  hf://{source_repo}",
        f"mtp sidecar repo: hf://{sidecar_repo}/mtp.safetensors",
        f"mtp quant:        {mtp_quant}",
        f"mtp output path:  {output_path}",
        "mtp step:         extract mtp.* tensors → quantize → upload",
    ]
    print("\n".join(lines))
