"""MTP weight extraction and sidecar publishing."""

from __future__ import annotations

import io
import json
import shutil
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


class _ProgressFile(io.BufferedIOBase):
    """BufferedIOBase that emits byte-level progress during the LFS upload pass.

    Inheriting from io.BufferedIOBase satisfies the isinstance check that
    huggingface_hub performs before computing UploadInfo. HF Hub's upload path
    calls seek(0) twice: once before hashing the file (UploadInfo.from_fileobj)
    and once before the actual LFS HTTP PUT. We count seek(0) calls and only
    arm progress emission on the second rewind, so the UI tracks the real network
    transfer rather than the fast CPU hash pass that completes in seconds.
    """

    def __init__(
        self, path: Path, emit: Callable[[str], None], pct_step: int = 2
    ) -> None:
        super().__init__()
        self._f = open(path, "rb")  # noqa: SIM115
        self._size = path.stat().st_size
        self._emit = emit
        self._pct_step = pct_step
        self._seek_zero_count = 0
        self._in_upload_pass = False
        self._last_pct = -1

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        data = self._f.read(size)
        if self._in_upload_pass and self._size > 0 and data:
            pos = self._f.tell()
            pct = min(99, int(pos * 100 // self._size))
            if pct >= self._last_pct + self._pct_step:
                self._last_pct = pct
                gb_done = pos / 1_073_741_824
                gb_total = self._size / 1_073_741_824
                self._emit(
                    f"mtp: uploading {pct}% ({gb_done:.1f} GB / {gb_total:.1f} GB)"
                )
        return data

    def read1(self, size: int = -1) -> bytes:  # type: ignore[override]
        return self.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        result = self._f.seek(offset, whence)
        if offset == 0 and whence == 0:
            self._seek_zero_count += 1
            # Second seek(0) = start of LFS upload; first = start of hash pass.
            # Always reset _last_pct so any later pass starts progress from 0.
            self._in_upload_pass = self._seek_zero_count >= 2
            self._last_pct = -1
        return result

    def tell(self) -> int:
        return self._f.tell()

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        if not self.closed:
            self._f.close()
            super().close()

    @property
    def name(self) -> str:
        return self._f.name


# ── FP8 / quantised-dtype decoding ───────────────────────────────────────────
# MLX 0.31.x has no float8 dtype, so we decode FP8 via a precomputed lookup
# table and numpy, then hand the result to MLX as float32/bfloat16.


def _build_fp8_e4m3fn_lut() -> list[float]:
    """Build a 256-entry F8_E4M3FN → float32 lookup table.

    E4M3FN spec (OCP MX): bias=7, no infinity, NaN only at exp=0b1111 man=0b111.
    All other exp=15 values are valid finite numbers (max ≈ 448).
    """
    lut: list[float] = []
    for byte in range(256):
        sign = (byte >> 7) & 1
        exp = (byte >> 3) & 0xF
        man = byte & 0x7
        if exp == 0xF and man == 0x7:
            val: float = float("nan")
        elif exp == 0:
            val = (man / 8.0) * (2.0 ** -6)
        else:
            val = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
        lut.append(-val if sign else val)
    return lut


_FP8_E4M3FN_LUT: list[float] = _build_fp8_e4m3fn_lut()


def _decode_fp8_e4m3fn(raw: bytes) -> Any:
    """Decode raw F8_E4M3FN bytes to a flat float32 numpy array.

    numpy is imported lazily so that importing this module does not require
    numpy in base (non-mtp) installs.
    """
    import numpy as np

    table = np.array(_FP8_E4M3FN_LUT, dtype=np.float32)
    return table[np.frombuffer(raw, dtype=np.uint8)]


def _decode_e8m0(raw: bytes) -> Any:
    """Decode F8_E8M0 block-exponent bytes to float32 scale values (2^(b−127))."""
    import numpy as np

    return np.power(
        2.0, np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.0
    )


def _decode_bf16_as_f32(raw: bytes) -> Any:
    """Reinterpret raw BF16 bytes as float32 (zero-extend upper 16 bits)."""
    import numpy as np

    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


def _f32_to_bf16_bytes(arr: Any) -> bytes:
    """Encode a float32 array as raw BF16 bytes using round-to-nearest-even.

    Shifting bits off with >> 16 truncates the mantissa. The correct conversion
    adds a rounding bias of 0x7FFF plus the LSB of the BF16 result, which
    implements round-to-nearest-even per IEEE 754.
    """
    import numpy as np

    u32 = np.asarray(arr, dtype=np.float32).view(np.uint32).copy()
    u32 += np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    return (u32 >> np.uint32(16)).astype(np.uint16).tobytes()


def _apply_block_scale(
    weights_f32: Any,
    scales_f32: Any,
    shape: list[int],
) -> Any:
    """Apply per-block MX scales to a weight array, inferring block size.

    Handles two layouts:

    * **2D tiled** — weight stored as [R, C] with one scale per B×B tile
      (ceil(R/B) × ceil(C/B) == n_scale). Checked first for 2D weights so
      that axis-aligned tile boundaries are respected even when n_weight is
      divisible by n_scale (e.g. [256, 256] / 4 scales with B=128 — the flat
      path would silently scale rows instead of quadrants).

    * **1D flat** — n_weight divisible by n_scale; block_size = n_weight / n_scale.
      Used by DeepSeek V3 (_scale_inv) and V4-Flash/Pro (.scale), or as fallback
      when no 2D tiling matches.
    """
    import numpy as np

    n_weight = int(np.prod(shape)) if shape else 1
    n_scale = scales_f32.size
    if n_scale == 0 or n_weight == 0:
        return weights_f32.reshape(shape).astype(np.float32)

    # ── 2D tiled block quantisation — checked before flat ────────────────────
    # For weight [R, C] with block size B: scale count == ceil(R/B) * ceil(C/B).
    # Partial right/bottom-edge tiles are handled by zero-padding to full block
    # boundaries, scaling, then stripping the padding.
    if len(shape) == 2:
        R, C = shape
        for B in (128, 64, 256, 32):
            Rb = (R + B - 1) // B
            Cb = (C + B - 1) // B
            if Rb * Cb != n_scale:
                continue
            R_pad, C_pad = Rb * B, Cb * B
            w_pad = np.zeros((R_pad, C_pad), dtype=np.float32)
            w_pad[:R, :C] = weights_f32.reshape(R, C)
            # [Rb, B, Cb, B]: each tile is contiguous in the last two axes.
            tiles = w_pad.reshape(Rb, B, Cb, B)
            s = scales_f32.reshape(Rb, 1, Cb, 1)
            return (tiles * s).reshape(R_pad, C_pad)[:R, :C].reshape(shape).astype(
                np.float32
            )

    # ── Flat 1D block quantisation ────────────────────────────────────────────
    if n_weight % n_scale == 0:
        block_size = n_weight // n_scale
        w = weights_f32.reshape(n_scale, block_size)
        s = scales_f32.reshape(n_scale, 1)
        return (w * s).reshape(shape).astype(np.float32)

    raise MtpExtractionError(
        f"weight elements ({n_weight}) not divisible by scale count "
        f"({n_scale}) — unexpected quantisation block layout"
    )


# ── Sidecar existence check ───────────────────────────────────────────────────


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


# ── Shard detection ───────────────────────────────────────────────────────────


def _is_mtp_key(key: str) -> bool:
    return key.startswith("mtp.") or ".mtp." in key


def _find_mtp_shards(
    source_repo: str,
    *,
    token: str | None,
    cache_dir: str | None = None,
) -> tuple[list[str], str]:
    """Return ``(shard_filenames, key_prefix)`` for the MTP tensors in source_repo.

    Two layouts are supported:

    * **New style** (e.g. DeepSeek V4-Flash): ``mtp.*`` tensor keys.
      ``key_prefix`` is ``"mtp."``.
    * **Old style** (e.g. DeepSeek V3 / V3-0324): MTP heads are stored as the
      extra transformer layer beyond ``num_hidden_layers``.  Detected by reading
      ``config.json`` and checking ``num_nextn_predict_layers > 0``, then looking
      for ``model.layers.{num_hidden_layers}.*`` keys in the weight map.
      ``key_prefix`` is ``"model.layers.{N}."``.

    Falls back to the single-file ``model.safetensors`` layout for new-style keys
    only.  Returns ``([], "mtp.")`` when no MTP tensors are found.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    # ── sharded index path ────────────────────────────────────────────────────
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

        # New style: mtp.* keys.
        shards: set[str] = set()
        for tensor_key, shard_file in weight_map.items():
            if tensor_key.startswith("mtp.") or ".mtp." in tensor_key:
                shards.add(shard_file)
        if shards:
            return sorted(shards), "mtp."

        # Old style: model.layers.{num_hidden_layers}.* extra MTP layer.
        try:
            config_path = hf_hub_download(
                repo_id=source_repo,
                filename="config.json",
                token=token,
                cache_dir=cache_dir,
            )
            with open(config_path, encoding="utf-8") as fh:
                config = json.load(fh)
        except (EntryNotFoundError, Exception):  # noqa: BLE001
            return [], "mtp."

        num_hidden: int = config.get("num_hidden_layers", 0)
        num_nextn: int = config.get("num_nextn_predict_layers", 0)
        if num_nextn <= 0 or num_hidden <= 0:
            return [], "mtp."

        prefix = f"model.layers.{num_hidden}."
        for tensor_key, shard_file in weight_map.items():
            if tensor_key.startswith(prefix):
                shards.add(shard_file)
        return (sorted(shards), prefix) if shards else ([], "mtp.")

    except EntryNotFoundError:
        pass

    # ── single-file fallback (new-style keys only) ────────────────────────────
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
        return [], "mtp."

    with safe_open(single_path, framework="numpy") as f:
        if any(k.startswith("mtp.") or ".mtp." in k for k in f.keys()):  # noqa: SIM118
            return ["model.safetensors"], "mtp."
    return [], "mtp."


# ── Tensor reading ────────────────────────────────────────────────────────────


def _find_scale_key(
    weight_key: str, header: dict[str, Any]
) -> tuple[str, bool] | None:
    """Return ``(scale_key, is_inverse)`` for a quantised weight tensor, or None.

    Two pairing conventions:

    * **V3 style** (``weight_scale_inv`` suffix): the stored value is ``1/scale``,
      so ``is_inverse=True`` — the caller must take the reciprocal before multiplying.
    * **V4-Flash style** (``.scale`` sibling key, dtype F8_E8M0): the stored value
      IS the scale, so ``is_inverse=False``.
    """
    # V3: weight_key ends with ".weight"; scale is weight_key + "_scale_inv"
    candidate_v3 = weight_key + "_scale_inv"
    if candidate_v3 in header:
        return candidate_v3, True
    # V4-Flash: weight_key ends with ".weight"; scale is base + ".scale"
    if weight_key.endswith(".weight"):
        candidate_v4 = weight_key[: -len(".weight")] + ".scale"
        if candidate_v4 in header:
            return candidate_v4, False
    return None


# ── Streaming safetensors writer ─────────────────────────────────────────────


def _write_mtp_streaming(
    local_shards: list[Path],
    output_path: Path,
    key_prefix: str,
    emit: Callable[[str], None],
) -> int:
    """Stream MTP tensors from shards into a single BF16 safetensors output file.

    Processes one tensor at a time so memory usage is bounded to a single
    tensor regardless of how many tensors or how large the source shards are.
    This is the correct path for large models (DeepSeek V4-Pro, V3, etc.) where
    accumulating all dequantised tensors in a dict before saving would require
    tens of GB of RAM.

    Returns the number of tensors written.
    """
    import numpy as np

    def _in_ns(key: str) -> bool:
        if key_prefix == "mtp.":
            return _is_mtp_key(key)
        return key.startswith(key_prefix)

    # ── Pass 1: collect tensor metadata without loading data ──────────────
    shard_meta: list[tuple[Path, dict[str, Any], int]] = []
    for shard_path in local_shards:
        with open(shard_path, "rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            hdr: dict[str, Any] = json.loads(fh.read(header_len).decode("utf-8"))
        shard_meta.append((shard_path, hdr, 8 + header_len))

    # Identify primary vs. scale-companion keys.
    # Build scale_keys by resolving each quantised weight's companion via
    # _find_scale_key — the same pairing logic used when writing tensor data.
    # This catches companions of any dtype (BF16, F32, F8_E8M0), not just F8_E8M0.
    _quantised_set = {"F8_E4M3", "I8"}
    output_entries: list[tuple[str, int, list[int], str]] = []
    for shard_idx, (_, hdr, _) in enumerate(shard_meta):
        scale_keys: set[str] = set()
        for k, m in hdr.items():
            if k != "__metadata__" and _in_ns(k) and m["dtype"] in _quantised_set:
                pair = _find_scale_key(k, hdr)
                if pair is not None:
                    scale_keys.add(pair[0])
        for key, meta in hdr.items():
            if key == "__metadata__" or not _in_ns(key) or key in scale_keys:
                continue
            output_entries.append((key, shard_idx, meta["shape"], meta["dtype"]))

    if not output_entries:
        raise MtpExtractionError(
            f"shards downloaded but no MTP tensors found with prefix {key_prefix!r}"
        )

    # ── Build output safetensors header (data_offsets relative to data start) ─
    offset = 0
    out_hdr: dict[str, Any] = {}
    for key, _, shape, _ in output_entries:
        n_elems = int(np.prod(shape)) if shape else 1
        byte_size = n_elems * 2  # BF16 = 2 bytes/element
        out_hdr[key] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + byte_size],
        }
        offset += byte_size

    # Pad header JSON to a multiple of 8 bytes (safetensors spec).
    hdr_json = json.dumps(out_hdr, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(hdr_json) % 8) % 8
    hdr_json += b" " * pad

    # ── Pass 2: stream one tensor at a time into the output file ─────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_fhs = [  # noqa: SIM115 — closed in the finally block below
        open(path, "rb") for path, _, _ in shard_meta  # noqa: SIM115
    ]
    try:
        with open(output_path, "wb") as out_fh:
            out_fh.write(struct.pack("<Q", len(hdr_json)))
            out_fh.write(hdr_json)

            for key, shard_idx, shape, src_dtype in output_entries:
                _, shard_hdr, data_base = shard_meta[shard_idx]
                fh = shard_fhs[shard_idx]
                meta = shard_hdr[key]
                start, end = meta["data_offsets"]
                fh.seek(data_base + start)
                raw = fh.read(end - start)

                if src_dtype == "BF16":
                    out_fh.write(raw)

                elif src_dtype == "F16":
                    f32 = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                    out_fh.write(_f32_to_bf16_bytes(f32))

                elif src_dtype == "F32":
                    f32 = np.frombuffer(raw, dtype=np.float32)
                    out_fh.write(_f32_to_bf16_bytes(f32))

                elif src_dtype in ("F8_E4M3", "I8"):
                    result = _find_scale_key(key, shard_hdr)
                    if result is None:
                        raise MtpExtractionError(
                            f"no scale tensor found for {src_dtype} tensor {key!r}"
                        )
                    scale_key, is_inverse = result
                    scale_meta = shard_hdr[scale_key]
                    s_start, s_end = scale_meta["data_offsets"]
                    fh.seek(data_base + s_start)
                    scale_raw = fh.read(s_end - s_start)

                    if src_dtype == "F8_E4M3":
                        weights_f32 = _decode_fp8_e4m3fn(raw)
                    else:  # I8
                        weights_f32 = np.frombuffer(
                            raw, dtype=np.int8
                        ).astype(np.float32)

                    scale_dtype = scale_meta["dtype"]
                    if scale_dtype == "F8_E8M0":
                        scales_f32 = _decode_e8m0(scale_raw)
                    elif scale_dtype == "F32":
                        scales_f32 = np.frombuffer(
                            scale_raw, dtype=np.float32
                        ).copy()
                    elif scale_dtype == "BF16":
                        scales_f32 = _decode_bf16_as_f32(scale_raw)
                    else:
                        raise MtpExtractionError(
                            f"unsupported scale dtype {scale_dtype!r} for "
                            f"{scale_key!r}"
                        )

                    if is_inverse:
                        scales_f32 = 1.0 / scales_f32

                    dequant = _apply_block_scale(weights_f32, scales_f32, shape)
                    out_fh.write(_f32_to_bf16_bytes(dequant))

                else:
                    raise MtpExtractionError(
                        f"unsupported MTP tensor dtype {src_dtype!r} for {key!r}"
                    )
    finally:
        for fh in shard_fhs:
            fh.close()

    return len(output_entries)


# ── Public extraction entry point ─────────────────────────────────────────────


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

    Downloads only the safetensors shards that contain MTP head tensors, reads
    only those tensors out of each shard (never the full shard), dequantises any
    FP8 or INT8 tensors to BF16, saves the result as ``mtp.safetensors``, and
    uploads to ``sidecar_repo`` on Hugging Face.

    Two MTP storage layouts are supported automatically:

    * **New style** (e.g. DeepSeek V4-Flash): ``mtp.*`` tensor keys.
    * **Old style** (e.g. DeepSeek V3/V3-0324): MTP heads stored as the extra
      transformer layer ``model.layers.{num_hidden_layers}.*``, detected via
      ``num_nextn_predict_layers`` in the model's ``config.json``.

    The heads are deliberately **not quantized**. They are the speculative
    drafter, whose sole job is draft acceptance, so quantizing them would degrade
    the very thing the sidecar exists to provide — to save only tens of MB on a
    multi-GB model. They are also independent of the target's quantization (the
    Skulk runtime loads them through its own path, not the model's), so **one
    bf16 sidecar per base model serves every quantization of that model**.

    Progress lines go through the ``log`` callback. It defaults to writing to
    stderr (the CLI behavior); callers that need per-job routing (e.g. the
    skulk-ui server) pass their own sink instead of relying on a global stream.

    Scratch artifacts (the ``.safetensors`` output file and the ``_hf_cache``
    shard cache) are deleted automatically after a successful upload. Skulk
    owns the artifact lifecycle; SWP's job ends when the push completes.

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

    cache_dir = str(scratch_root / "_hf_cache")
    try:
        # Identify which shards contain MTP tensors and what key prefix they use.
        shard_files, key_prefix = _find_mtp_shards(
            source_repo, token=token, cache_dir=cache_dir
        )
        if not shard_files:
            raise MtpExtractionError(
                f"no MTP head tensors found in {source_repo}; "
                "confirm this model has native MTP heads "
                "(checked for mtp.* keys and model.layers.{N}.* via config.json)"
            )

        emit(
            f"mtp: found MTP tensors in {len(shard_files)} shard(s)"
            f" (prefix: {key_prefix!r})"
        )

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

        # Stream MTP tensors from shards directly into the output file one
        # tensor at a time. This keeps peak memory bounded to a single tensor
        # regardless of total dataset size — essential for large models
        # (DeepSeek V4-Pro etc.) where accumulating everything in RAM before
        # saving would OOM.
        emit("mtp: streaming tensors to disk (bf16, unquantized)...")
        n_tensors = _write_mtp_streaming(local_shards, output_path, key_prefix, emit)
        emit(
            f"mtp: saved {n_tensors} tensor(s) at bf16 (unquantized) "
            f"to {output_path}"
        )

        # Upload.
        emit(f"mtp: uploading to hf://{sidecar_repo}/mtp.safetensors")
        with _ProgressFile(output_path, emit) as pf:
            upload_file(
                path_or_fileobj=pf,  # type: ignore[arg-type]
                path_in_repo="mtp.safetensors",
                repo_id=sidecar_repo,
                repo_type="model",
                token=token,
                commit_message=(
                    f"Add MTP sidecar from {source_repo} (bf16, unquantized)"
                ),
            )
        emit(f"mtp: published to hf://{sidecar_repo}/mtp.safetensors")
    finally:
        # Skulk owns artifact lifecycle — discard everything we staged locally,
        # success or failure. A failed run would otherwise leak the partial
        # output plus tens of GB of cached shards (the server gives each job a
        # fresh scratch dir, so a leaked cache is never reused or reclaimed).
        output_path.unlink(missing_ok=True)
        hf_cache = scratch_root / "_hf_cache"
        if hf_cache.exists():
            shutil.rmtree(hf_cache)

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
        "mtp step:         extract MTP tensors → dequantise (FP8/I8→BF16) → upload",
    ]
    print("\n".join(lines))
