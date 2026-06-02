"""MTP weight extraction and sidecar publishing."""

from __future__ import annotations

import json
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


class MtpExtractionError(RuntimeError):
    """Raised when MTP extraction or publishing fails."""


def _stderr_log(message: str) -> None:
    """Default log sink — writes a progress line to stderr (CLI behavior)."""
    print(message, file=sys.stderr)


def _sidecar_already_published(sidecar_repo: str, *, token: str | None) -> bool:
    """Return True if ``sidecar_repo`` already has ``mtp.safetensors`` on the Hub.

    Best-effort: any lookup failure (repo absent, no network, hub unavailable)
    is treated as "not published" so extraction proceeds normally.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False
    try:
        return bool(HfApi().file_exists(sidecar_repo, "mtp.safetensors", token=token))
    except Exception:  # noqa: BLE001 - existence check is best-effort
        return False


def _is_mtp_key(key: str) -> bool:
    return key.startswith("mtp.") or ".mtp." in key


def _read_mtp_tensors(shard_path: Path, *, mx: Any) -> dict[str, Any]:
    """Return the mtp.* tensors from a safetensors shard, reading only their bytes.

    Parses the safetensors header and reads each mtp.* tensor's byte range
    directly, so the (potentially multi-GB) non-MTP weights are never read.
    bf16/f16 are bitcast through uint16 because numpy — which mlx/safetensors
    would otherwise route through — has no bfloat16 dtype.
    """
    dtype_map = {
        "BF16": mx.bfloat16,
        "F16": mx.float16,
        "F32": mx.float32,
    }
    with open(shard_path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))
        data_base = 8 + header_len

        tensors: dict[str, Any] = {}
        for key, meta in header.items():
            if key == "__metadata__" or not _is_mtp_key(key):
                continue
            dtype_name = meta["dtype"]
            if dtype_name not in dtype_map:
                raise MtpExtractionError(
                    f"unsupported MTP tensor dtype {dtype_name!r} for {key} "
                    f"in {shard_path.name}"
                )
            start, end = meta["data_offsets"]
            fh.seek(data_base + start)
            raw = fh.read(end - start)
            shape = meta["shape"]
            if dtype_name in ("BF16", "F16"):
                # 2 bytes/elt: load as uint16, then bitcast to the float dtype.
                arr = mx.array(memoryview(raw).cast("H"), dtype=mx.uint16)
                arr = arr.view(dtype_map[dtype_name])
            else:  # F32
                arr = mx.array(memoryview(raw).cast("f"), dtype=mx.float32)
            tensors[key] = arr.reshape(shape) if shape else arr
        return tensors


def extract_mtp(
    source_repo: str,
    sidecar_repo: str,
    scratch_root: Path,
    *,
    token: str | None,
    dry_run: bool = False,
    force: bool = False,
    catalog_key: str | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """Extract MTP weights from source_repo and publish them to sidecar_repo.

    Downloads only the safetensors shards that contain ``mtp.*`` keys, reads only
    the ``mtp.*`` tensors out of each shard (never the full shard), saves them at
    full precision (bf16, **unquantized**) as ``mtp.safetensors``, and uploads to
    ``sidecar_repo`` on Hugging Face.

    The heads are deliberately **not quantized**. They are the speculative
    drafter, whose sole job is draft acceptance, so quantizing them would degrade
    the very thing the sidecar exists to provide — to save only tens of MB on a
    multi-GB model. They are also independent of the target's quantization (the
    Skulk runtime loads them through its own path, not the model's), so **one
    bf16 sidecar per base model serves every quantization of that model**.

    Progress lines go through the ``log`` callback. It defaults to writing to
    stderr (the CLI behavior); callers that need per-job routing (e.g. the
    skulk-ui server) pass their own sink instead of relying on a global stream.

    In dry-run mode prints the plan without downloading or uploading anything.
    """

    emit = log if log is not None else _stderr_log

    output_path = scratch_root / _sidecar_filename(sidecar_repo)

    if dry_run:
        _print_dry_run_plan(source_repo, sidecar_repo, output_path)
        return

    # The heads belong to the base model, so one sidecar serves every
    # quantization of it. If it's already published, don't silently re-extract —
    # tell the operator it already covers this model and skip.
    if not force and _sidecar_already_published(sidecar_repo, token=token):
        emit(
            f"mtp: sidecar already exists at hf://{sidecar_repo}/mtp.safetensors "
            f"— it already covers {source_repo} and every quantization of it. "
            "Skipping (pass --force to re-extract)."
        )
        return

    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MtpExtractionError(
            "mlx is required for reading MTP weights"
        ) from exc

    try:
        from huggingface_hub import create_repo, hf_hub_download, upload_file
        from huggingface_hub.utils.tqdm import (
            disable_progress_bars,  # type: ignore[import-untyped]
        )
        disable_progress_bars()
    except ImportError as exc:
        raise MtpExtractionError(
            "huggingface_hub is required for MTP extraction"
        ) from exc

    # Verify/create the sidecar repo before the expensive download work.
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

    emit(f"mtp: found mtp.* keys in {len(shard_files)} shard(s)")

    # Download the relevant shards into scratch.
    scratch_root.mkdir(parents=True, exist_ok=True)
    local_shards: list[Path] = []
    for index, shard in enumerate(shard_files, 1):
        emit(f"mtp: downloading shard {index}/{len(shard_files)}: {shard}")
        local = Path(
            hf_hub_download(
                repo_id=source_repo,
                filename=shard,
                token=token,
                cache_dir=cache_dir,
            )
        )
        local_shards.append(local)
        emit(f"mtp: shard {index}/{len(shard_files)} ready")

    # Read ONLY the mtp.* tensors out of each shard by parsing the safetensors
    # header and reading just those byte ranges. Avoids materializing the whole
    # shard (gigabytes) to recover a handful of small heads, and handles bf16
    # natively (safe_open(framework="mlx") cannot — it raises on bfloat16).
    mtp_tensors: dict[str, Any] = {}
    for shard_path in local_shards:
        mtp_tensors.update(_read_mtp_tensors(shard_path, mx=mx))

    if not mtp_tensors:
        raise MtpExtractionError(
            f"shards were downloaded but no mtp.* tensors could be read"
            f" from {source_repo}"
        )

    emit(f"mtp: extracted {len(mtp_tensors)} tensor(s)")

    # Ship the heads UNQUANTIZED at full precision (bf16). They are the
    # speculative drafter — quantizing them would cost draft acceptance (the
    # whole point) to save only tens of MB, and the Skulk runtime consumes bf16
    # heads via its plain-matmul path.
    sidecar_tensors = {
        name: arr.astype(mx.bfloat16) for name, arr in mtp_tensors.items()
    }

    # Save.
    mx.save_safetensors(str(output_path), sidecar_tensors)
    emit(
        f"mtp: saved {len(sidecar_tensors)} tensor(s) at bf16 (unquantized) "
        f"to {output_path}"
    )

    # Upload.
    emit(f"mtp: uploading to hf://{sidecar_repo}/mtp.safetensors")
    upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo="mtp.safetensors",
        repo_id=sidecar_repo,
        repo_type="model",
        token=token,
        commit_message=f"Add MTP sidecar from {source_repo} (bf16, unquantized)",
    )
    emit(f"mtp: published to hf://{sidecar_repo}/mtp.safetensors")

    # Publish a self-describing model card so the sidecar carries its provenance
    # (source repo + revision), target, and inherited license.
    from skulk_weights_publisher.card_publish import publish_model_card

    publish_model_card(
        repo_id=sidecar_repo,
        artifact_type="mtp-sidecar",
        source_repo=source_repo,
        token=token,
        target_model=source_repo,
        catalog_key=catalog_key,
        weight_filename="mtp.safetensors",
        log=emit,
    )


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


def _sidecar_filename(sidecar_repo: str) -> str:
    return sidecar_repo.replace("/", "--") + "-mtp.safetensors"


def _print_dry_run_plan(
    source_repo: str,
    sidecar_repo: str,
    output_path: Path,
) -> None:
    lines = [
        f"mtp source repo:  hf://{source_repo}",
        f"mtp sidecar repo: hf://{sidecar_repo}/mtp.safetensors",
        "mtp precision:    bf16 (unquantized)",
        f"mtp output path:  {output_path}",
        "mtp step:         extract mtp.* tensors → upload (bf16, unquantized)",
    ]
    print("\n".join(lines))
