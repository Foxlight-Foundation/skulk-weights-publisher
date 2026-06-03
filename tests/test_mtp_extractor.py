"""Unit tests for mtp_extractor internal helpers."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import skulk_weights_publisher.mtp_extractor as mtp_mod
from skulk_weights_publisher.mtp_extractor import (
    MtpExtractionError,
    _apply_block_scale,
    _build_fp8_e4m3fn_lut,
    _decode_e8m0,
    _decode_fp8_e4m3fn,
    _find_scale_key,
    _print_dry_run_plan,
    _sidecar_filename,
    extract_mtp,
)

# ---------------------------------------------------------------------------
# extract_mtp — sidecar-exists skip (one sidecar per base, quant-independent)
# ---------------------------------------------------------------------------


def test_extract_mtp_skips_when_sidecar_already_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When the sidecar exists, extract_mtp informs the operator that it already
    # covers the model (and all its quantizations) and skips — no re-extraction.
    monkeypatch.setattr(mtp_mod, "_sidecar_already_published", lambda *a, **k: True)
    logs: list[str] = []

    extract_mtp(
        "Qwen/Qwen3.6-35B-A3B",
        "FoxlightAI/qwen3-6-35b-a3b-mtp",
        tmp_path,
        token="hf_tok",
        log=logs.append,
    )

    joined = "\n".join(logs)
    assert "already exists" in joined
    assert "every quantization" in joined
    # It returned before doing any extraction work — no shards downloaded.
    assert not (tmp_path / "_hf_cache").exists()


# ---------------------------------------------------------------------------
# _sidecar_filename
# ---------------------------------------------------------------------------


def test_sidecar_filename_replaces_slash() -> None:
    expected = "acme--model-mtp-mtp.safetensors"
    assert _sidecar_filename("acme/model-mtp") == expected


def test_sidecar_filename_no_slash() -> None:
    assert _sidecar_filename("modelonly") == "modelonly-mtp.safetensors"


# ---------------------------------------------------------------------------
# _print_dry_run_plan
# ---------------------------------------------------------------------------


def test_print_dry_run_plan_output(capsys: pytest.CaptureFixture[str]) -> None:
    _print_dry_run_plan(
        "owner/source-bf16",
        "owner/sidecar",
        Path("/scratch/sidecar.safetensors"),
    )
    out = capsys.readouterr().out
    assert "owner/source-bf16" in out
    assert "owner/sidecar" in out
    # Heads ship unquantized at full precision — the plan must say so.
    assert "bf16" in out
    assert "/scratch/sidecar.safetensors" in out


# ---------------------------------------------------------------------------
# _find_mtp_shards
# ---------------------------------------------------------------------------


def _fake_safetensors_module(keys: list[str]) -> MagicMock:
    """Return a fake safetensors module whose safe_open yields ``keys``."""
    mock_f = MagicMock()
    mock_f.__enter__ = lambda s: s
    mock_f.__exit__ = MagicMock(return_value=False)
    mock_f.keys.return_value = keys

    mod = MagicMock()
    mod.safe_open.return_value = mock_f
    return mod


def test_find_mtp_shards_sharded_index(tmp_path: Path) -> None:
    from skulk_weights_publisher.mtp_extractor import _find_mtp_shards

    index = {
        "weight_map": {
            "mtp.layer.weight": "model-00003-of-00010.safetensors",
            "model.layer.weight": "model-00001-of-00010.safetensors",
            ".mtp.head.weight": "model-00004-of-00010.safetensors",
        }
    }
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with patch("huggingface_hub.hf_hub_download", return_value=str(index_path)):
        shards, prefix = _find_mtp_shards("owner/repo", token=None)

    assert shards == sorted(
        {"model-00003-of-00010.safetensors", "model-00004-of-00010.safetensors"}
    )
    assert prefix == "mtp."


def test_find_mtp_shards_no_mtp_keys_in_index(tmp_path: Path) -> None:
    from skulk_weights_publisher.mtp_extractor import _find_mtp_shards

    # No mtp.* keys and no num_nextn_predict_layers in config → empty.
    index = {"weight_map": {"model.layer.weight": "model-00001-of-00002.safetensors"}}
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with patch("huggingface_hub.hf_hub_download", return_value=str(index_path)):
        shards, prefix = _find_mtp_shards("owner/repo", token=None)

    assert shards == []
    assert prefix == "mtp."


def test_find_mtp_shards_old_style_via_config(tmp_path: Path) -> None:
    """DeepSeek V3-style: MTP layer at model.layers.{num_hidden_layers}.*"""
    from skulk_weights_publisher.mtp_extractor import _find_mtp_shards

    index = {
        "weight_map": {
            "model.layers.60.mlp.weight": "shard-00001.safetensors",
            "model.layers.61.attn.weight": "shard-00002.safetensors",
            "model.layers.61.mlp.weight": "shard-00002.safetensors",
            "model.layers.61.attn.weight_scale_inv": "shard-00002.safetensors",
        }
    }
    config = {"num_hidden_layers": 61, "num_nextn_predict_layers": 1}
    index_path = tmp_path / "model.safetensors.index.json"
    config_path = tmp_path / "config.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fake_download(repo_id: str, filename: str, **kw: Any) -> str:
        if filename == "model.safetensors.index.json":
            return str(index_path)
        if filename == "config.json":
            return str(config_path)
        raise FileNotFoundError(filename)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        shards, prefix = _find_mtp_shards("owner/repo", token=None)

    assert shards == ["shard-00002.safetensors"]
    assert prefix == "model.layers.61."


def test_find_mtp_shards_falls_back_when_no_index(tmp_path: Path) -> None:
    from huggingface_hub.errors import EntryNotFoundError

    from skulk_weights_publisher.mtp_extractor import _find_mtp_shards

    single_path = tmp_path / "model.safetensors"
    single_path.write_bytes(b"")

    def fake_download(repo_id: str, filename: str, **kw: Any) -> str:
        if filename == "model.safetensors.index.json":
            raise EntryNotFoundError("no index")
        return str(single_path)

    fake_st = _fake_safetensors_module(["mtp.layer.weight", "model.embed.weight"])

    with (
        patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
        patch.dict(sys.modules, {"safetensors": fake_st}),
    ):
        shards, prefix = _find_mtp_shards("owner/repo", token=None)

    assert shards == ["model.safetensors"]
    assert prefix == "mtp."


def test_find_mtp_shards_no_mtp_keys_in_single_file(tmp_path: Path) -> None:
    from huggingface_hub.errors import EntryNotFoundError

    from skulk_weights_publisher.mtp_extractor import _find_mtp_shards

    single_path = tmp_path / "model.safetensors"
    single_path.write_bytes(b"")

    def fake_download(repo_id: str, filename: str, **kw: Any) -> str:
        if filename == "model.safetensors.index.json":
            raise EntryNotFoundError("no index")
        return str(single_path)

    fake_st = _fake_safetensors_module(["model.embed.weight", "model.layer.weight"])

    with (
        patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
        patch.dict(sys.modules, {"safetensors": fake_st}),
    ):
        shards, prefix = _find_mtp_shards("owner/repo", token=None)

    assert shards == []
    assert prefix == "mtp."


def test_find_mtp_shards_no_files_at_all(tmp_path: Path) -> None:
    from huggingface_hub.errors import EntryNotFoundError

    from skulk_weights_publisher.mtp_extractor import _find_mtp_shards

    def fake_download(repo_id: str, filename: str, **kw: Any) -> str:
        raise EntryNotFoundError("not found")

    fake_st = _fake_safetensors_module([])

    with (
        patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
        patch.dict(sys.modules, {"safetensors": fake_st}),
    ):
        shards, prefix = _find_mtp_shards("owner/repo", token=None)

    assert shards == []
    assert prefix == "mtp."


def test_read_mtp_tensors_reads_only_mtp_keys(tmp_path: Path) -> None:
    """_read_mtp_tensors returns only mtp.* tensors and round-trips bf16."""
    import struct

    mx = pytest.importorskip("mlx.core")
    from skulk_weights_publisher.mtp_extractor import _read_mtp_tensors

    # bf16 encodings: 1,2,3,4 and 5,6,7,8
    mtp_bytes = struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080)
    other_bytes = struct.pack("<4H", 0x40A0, 0x40C0, 0x40E0, 0x4100)
    header = {
        "mtp.fc.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
        "model.other.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [8, 16],
        },
    }
    hb = json.dumps(header).encode()
    shard = tmp_path / "shard.safetensors"
    with open(shard, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(mtp_bytes + other_bytes)

    tensors = _read_mtp_tensors(shard, mx=mx)

    assert list(tensors) == ["mtp.fc.weight"]  # non-mtp skipped
    assert tensors["mtp.fc.weight"].dtype == mx.bfloat16
    as_floats = tensors["mtp.fc.weight"].astype(mx.float32).tolist()
    assert as_floats == [[1.0, 2.0], [3.0, 4.0]]


def test_read_mtp_tensors_rejects_unknown_dtype(tmp_path: Path) -> None:
    import struct

    mx = pytest.importorskip("mlx.core")
    from skulk_weights_publisher.mtp_extractor import (
        MtpExtractionError,
        _read_mtp_tensors,
    )

    header = {"mtp.x": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}}
    hb = json.dumps(header).encode()
    shard = tmp_path / "shard.safetensors"
    with open(shard, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(struct.pack("<q", 1))

    with pytest.raises(MtpExtractionError, match="unsupported MTP tensor dtype"):
        _read_mtp_tensors(shard, mx=mx)


# ---------------------------------------------------------------------------
# FP8 / quantisation helpers
# ---------------------------------------------------------------------------


def test_fp8_e4m3fn_lut_known_values() -> None:
    lut = _build_fp8_e4m3fn_lut()
    assert len(lut) == 256
    # 0x38 = sign=0, exp=7, man=0 → 2^(7-7)*1.0 = 1.0
    assert lut[0x38] == pytest.approx(1.0)
    # 0x40 = sign=0, exp=8, man=0 → 2^(8-7)*1.0 = 2.0
    assert lut[0x40] == pytest.approx(2.0)
    # 0x00 = +0.0
    assert lut[0x00] == 0.0
    # 0x7F = NaN (exp=15, man=7)
    assert math.isnan(lut[0x7F])
    # 0xFF = NaN (sign=1, exp=15, man=7)
    assert math.isnan(lut[0xFF])
    # 0x7E = sign=0, exp=15, man=6 → 2^8 * (1+6/8) = 256 * 1.75 = 448.0
    assert lut[0x7E] == pytest.approx(448.0)


def test_decode_fp8_e4m3fn_array() -> None:
    raw = bytes([0x38, 0x40])  # FP8 values 1.0, 2.0
    result = _decode_fp8_e4m3fn(raw)
    assert result.tolist() == pytest.approx([1.0, 2.0])


def test_decode_e8m0_values() -> None:
    raw = bytes([127, 128, 126])  # → 1.0, 2.0, 0.5
    result = _decode_e8m0(raw)
    assert result.tolist() == pytest.approx([1.0, 2.0, 0.5])


def test_apply_block_scale_basic() -> None:
    # 4 weights, 2 block-scales → block_size=2
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scales = np.array([2.0, 0.5], dtype=np.float32)
    result = _apply_block_scale(weights, scales, [4])
    assert result.tolist() == pytest.approx([2.0, 4.0, 1.5, 2.0])


def test_apply_block_scale_2d() -> None:
    # shape [2, 4], 4 block-scales (one per row * 2 blocks of 2)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    scales = np.array([2.0, 1.0, 0.5, 1.0], dtype=np.float32)
    result = _apply_block_scale(weights, scales, [2, 4])
    expected = np.array([
        [1.0 * 2.0, 2.0 * 2.0, 3.0 * 1.0, 4.0 * 1.0],
        [5.0 * 0.5, 6.0 * 0.5, 7.0 * 1.0, 8.0 * 1.0],
    ], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_find_scale_key_v4_flash_style() -> None:
    header = {
        "mtp.0.attn.wkv.weight": {"dtype": "F8_E4M3"},
        "mtp.0.attn.wkv.scale": {"dtype": "F8_E8M0"},
    }
    result = _find_scale_key("mtp.0.attn.wkv.weight", header)
    assert result == ("mtp.0.attn.wkv.scale", False)


def test_find_scale_key_v3_style() -> None:
    header = {
        "model.layers.61.mlp.down.weight": {"dtype": "F8_E4M3"},
        "model.layers.61.mlp.down.weight_scale_inv": {"dtype": "F32"},
    }
    result = _find_scale_key("model.layers.61.mlp.down.weight", header)
    assert result == ("model.layers.61.mlp.down.weight_scale_inv", True)


def test_find_scale_key_none_when_missing() -> None:
    header = {"mtp.0.norm.weight": {"dtype": "BF16"}}
    assert _find_scale_key("mtp.0.norm.weight", header) is None


def test_apply_block_scale_2d_partial_col_block() -> None:
    """2D tiling where the last column block is partial (V3.2 MTP style).

    Mirrors the real failure: weight [7168, 576] with ceil(7168/128)*ceil(576/128)
    = 56*5 = 280 scales but 576 % 128 == 64 (non-zero). We use [96, 65] with
    block_size=32 (tried last in the B list) to keep data small:
      Rb=3, Cb=3, 9 scales, 6240 elements, 6240%9 != 0 → forces 2D path.
    """
    weights = np.ones(96 * 65, dtype=np.float32)
    scales = np.arange(1, 10, dtype=np.float32)  # 1..9
    result = _apply_block_scale(weights, scales, [96, 65])

    assert result.shape == (96, 65)
    # tile (0,0): row 0, col 0 → scale[0]=1
    assert result[0, 0] == pytest.approx(1.0)
    # tile (0,1): row 0, col 32 → scale[1]=2
    assert result[0, 32] == pytest.approx(2.0)
    # tile (0,2): row 0, col 64 (partial, only 1 element wide) → scale[2]=3
    assert result[0, 64] == pytest.approx(3.0)
    # tile (2,2): row 64, col 64 → scale[8]=9
    assert result[64, 64] == pytest.approx(9.0)


def test_apply_block_scale_2d_v32_shape() -> None:
    """Smoke test with the exact element counts from the V3.2 MTP failure."""
    # [7168, 576] with 280 scales must not raise.
    weights = np.ones(7168 * 576, dtype=np.float32)
    scales = np.full(280, 2.0, dtype=np.float32)
    result = _apply_block_scale(weights, scales, [7168, 576])
    assert result.shape == (7168, 576)
    np.testing.assert_allclose(result, 2.0)


def _make_shard(
    tmp_path: Path, tensors: list[tuple[str, str, list[int], bytes]]
) -> Path:
    """Write a minimal safetensors shard with the given tensors."""
    offset = 0
    header: dict[str, Any] = {}
    parts: list[bytes] = []
    for key, dtype, shape, data in tensors:
        end = offset + len(data)
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        parts.append(data)
        offset += len(data)
    hb = json.dumps(header).encode()
    shard = tmp_path / "shard.safetensors"
    with open(shard, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for p in parts:
            fh.write(p)
    return shard


def _decode_bf16_output(path: Path) -> dict[str, np.ndarray]:
    """Read a BF16 safetensors file from _write_mtp_streaming as float32 arrays."""
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        hdr: dict[str, Any] = json.loads(f.read(hdr_len))
        data = f.read()
    result: dict[str, np.ndarray] = {}
    for key, meta in hdr.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw_u16 = np.frombuffer(data[start:end], dtype=np.uint16).copy()
        f32 = (raw_u16.astype(np.uint32) << 16).view(np.float32)
        result[key] = f32.reshape(meta["shape"])
    return result


def test_read_mtp_tensors_fp8_e4m3_with_e8m0_scale(tmp_path: Path) -> None:
    """F8_E4M3 weight + F8_E8M0 scale → dequantised BF16 (V4-Flash style)."""
    mx = pytest.importorskip("mlx.core")
    from skulk_weights_publisher.mtp_extractor import _read_mtp_tensors

    # weights: FP8 1.0, 2.0 (bytes 0x38, 0x40); scale: E8M0 byte 128 → 2.0
    weight_bytes = bytes([0x38, 0x40])
    scale_bytes = bytes([128])  # 2^(128-127) = 2.0, block covers both elements

    shard = _make_shard(tmp_path, [
        ("mtp.0.attn.wkv.weight", "F8_E4M3", [2], weight_bytes),
        ("mtp.0.attn.wkv.scale",  "F8_E8M0", [1], scale_bytes),
    ])
    tensors = _read_mtp_tensors(shard, mx=mx)

    assert set(tensors) == {"mtp.0.attn.wkv.weight"}  # scale key excluded
    vals = tensors["mtp.0.attn.wkv.weight"].astype(mx.float32).tolist()
    assert vals == pytest.approx([2.0, 4.0], abs=0.05)  # 1.0*2, 2.0*2


def test_read_mtp_tensors_i8_with_e8m0_scale(tmp_path: Path) -> None:
    """I8 weight + F8_E8M0 scale → dequantised BF16 (V4-Flash expert style)."""
    mx = pytest.importorskip("mlx.core")
    from skulk_weights_publisher.mtp_extractor import _read_mtp_tensors

    # int8 values 3, 4; scale byte 127 → 2^0 = 1.0
    weight_bytes = struct.pack("<2b", 3, 4)
    scale_bytes = bytes([127])

    shard = _make_shard(tmp_path, [
        ("mtp.0.ffn.experts.0.w1.weight", "I8",      [2], weight_bytes),
        ("mtp.0.ffn.experts.0.w1.scale",  "F8_E8M0", [1], scale_bytes),
    ])
    tensors = _read_mtp_tensors(shard, mx=mx)

    assert set(tensors) == {"mtp.0.ffn.experts.0.w1.weight"}
    vals = tensors["mtp.0.ffn.experts.0.w1.weight"].astype(mx.float32).tolist()
    assert vals == pytest.approx([3.0, 4.0], abs=0.05)


def test_read_mtp_tensors_fp8_with_f32_scale_inv(tmp_path: Path) -> None:
    """F8_E4M3 weight + F32 weight_scale_inv → dequantised BF16 (V3 style)."""
    mx = pytest.importorskip("mlx.core")
    from skulk_weights_publisher.mtp_extractor import _read_mtp_tensors

    # weights: FP8 1.0, 2.0; scale_inv = 0.5 → actual scale = 1/0.5 = 2.0
    weight_bytes = bytes([0x38, 0x40])
    scale_inv_bytes = struct.pack("<f", 0.5)  # 1/scale → scale = 2.0

    shard = _make_shard(tmp_path, [
        ("model.layers.61.attn.w.weight",           "F8_E4M3", [2], weight_bytes),
        ("model.layers.61.attn.w.weight_scale_inv", "F32",     [1], scale_inv_bytes),
    ])
    tensors = _read_mtp_tensors(shard, mx=mx, key_prefix="model.layers.61.")

    assert set(tensors) == {"model.layers.61.attn.w.weight"}
    vals = tensors["model.layers.61.attn.w.weight"].astype(mx.float32).tolist()
    assert vals == pytest.approx([2.0, 4.0], abs=0.05)


def test_read_mtp_tensors_raises_when_scale_missing(tmp_path: Path) -> None:
    """F8_E4M3 tensor without a paired scale raises MtpExtractionError."""
    mx = pytest.importorskip("mlx.core")
    from skulk_weights_publisher.mtp_extractor import _read_mtp_tensors

    shard = _make_shard(tmp_path, [
        ("mtp.0.attn.w.weight", "F8_E4M3", [2], bytes([0x38, 0x40])),
    ])
    with pytest.raises(MtpExtractionError, match="no scale tensor found"):
        _read_mtp_tensors(shard, mx=mx)


def test_read_mtp_tensors_old_style_key_prefix(tmp_path: Path) -> None:
    """key_prefix='model.layers.61.' filters to V3-style MTP tensors."""
    mx = pytest.importorskip("mlx.core")
    from skulk_weights_publisher.mtp_extractor import _read_mtp_tensors

    bf16_bytes = struct.pack("<2H", 0x3F80, 0x4000)  # 1.0, 2.0

    shard = _make_shard(tmp_path, [
        ("model.layers.61.norm.weight", "BF16", [2], bf16_bytes),
        ("model.layers.60.norm.weight", "BF16", [2], bf16_bytes),  # excluded
    ])
    tensors = _read_mtp_tensors(shard, mx=mx, key_prefix="model.layers.61.")

    assert set(tensors) == {"model.layers.61.norm.weight"}
    vals = tensors["model.layers.61.norm.weight"].astype(mx.float32).tolist()
    assert vals == pytest.approx([1.0, 2.0])


# ---------------------------------------------------------------------------
# _f32_to_bf16_bytes
# ---------------------------------------------------------------------------


def test_f32_to_bf16_bytes_rounds_not_truncates() -> None:
    """Conversion must round to nearest even, not truncate.

    1.0 in float32 is 0x3F800000. The next BF16 value up is 0x3F81 (≈1.0078).
    A value halfway between them is 0x3F8080xx. Truncation gives 0x3F80 (1.0);
    round-to-nearest gives 0x3F81 (the closer value when the low bits push past
    the midpoint).
    """
    from skulk_weights_publisher.mtp_extractor import _f32_to_bf16_bytes

    def _bf16(b: bytes) -> int:
        raw = np.frombuffer(b, dtype=np.float32)
        return int(np.frombuffer(_f32_to_bf16_bytes(raw), dtype=np.uint16)[0])

    # 0x3F808000: halfway between BF16 0x3F80 and 0x3F81.
    # BF16 LSB is 0 (even) → round down → 0x3F80.
    assert _bf16(b'\x00\x80\x80\x3f') == 0x3F80

    # 0x3F818000: halfway between BF16 0x3F81 and 0x3F82.
    # BF16 LSB is 1 (odd) → round up to even → 0x3F82.
    assert _bf16(b'\x00\x80\x81\x3f') == 0x3F82

    # Just past midpoint: always round up regardless of LSB.
    assert _bf16(b'\x01\x80\x80\x3f') == 0x3F81


# ---------------------------------------------------------------------------
# _ProgressFile
# ---------------------------------------------------------------------------


def test_progress_file_is_buffered_io_base(tmp_path: Path) -> None:
    """_ProgressFile must pass the isinstance(obj, io.BufferedIOBase) check."""
    import io

    from skulk_weights_publisher.mtp_extractor import _ProgressFile

    p = tmp_path / "f.bin"
    p.write_bytes(b"\x00" * 8)
    with _ProgressFile(p, [].append) as pf:
        assert isinstance(pf, io.BufferedIOBase)


def test_progress_file_only_emits_on_second_read_pass(tmp_path: Path) -> None:
    """Progress must be silent during the hash pass and active during upload.

    HF Hub calls seek(0) before hashing (pass 1) and again before the LFS PUT
    (pass 2). Emitting during the fast CPU hash would show near-complete progress
    before the real upload even starts; _last_pct = 99 then prevents the upload
    pass from emitting any progress at all.
    """
    from skulk_weights_publisher.mtp_extractor import _ProgressFile

    p = tmp_path / "f.bin"
    p.write_bytes(b"\xff" * 1024)
    logs: list[str] = []

    with _ProgressFile(p, logs.append, pct_step=1) as pf:
        # Simulate HF Hub: seek(0) → hash read → seek(0) → upload read
        pf.seek(0)
        pf.read()   # hash pass — must emit nothing
        assert logs == [], "no progress should fire during the hash pass"

        pf.seek(0)
        pf.read()   # upload pass — must emit progress
        assert any("mtp: uploading" in line for line in logs)


# ---------------------------------------------------------------------------
# _write_mtp_streaming
# ---------------------------------------------------------------------------


def test_write_mtp_streaming_bf16_passthrough(tmp_path: Path) -> None:
    from skulk_weights_publisher.mtp_extractor import _write_mtp_streaming

    bf16 = struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080)  # 1,2,3,4
    shard = _make_shard(tmp_path, [("mtp.w.weight", "BF16", [2, 2], bf16)])
    out = tmp_path / "out.safetensors"

    n = _write_mtp_streaming([shard], out, "mtp.", [].append)

    assert n == 1
    tensors = _decode_bf16_output(out)
    assert list(tensors) == ["mtp.w.weight"]
    np.testing.assert_allclose(tensors["mtp.w.weight"].ravel(), [1, 2, 3, 4], rtol=0.01)


def test_write_mtp_streaming_f16_to_bf16(tmp_path: Path) -> None:
    from skulk_weights_publisher.mtp_extractor import _write_mtp_streaming

    f16 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes()
    shard = _make_shard(tmp_path, [("mtp.w.weight", "F16", [4], f16)])
    out = tmp_path / "out.safetensors"

    _write_mtp_streaming([shard], out, "mtp.", [].append)

    tensors = _decode_bf16_output(out)
    np.testing.assert_allclose(tensors["mtp.w.weight"], [1, 2, 3, 4], rtol=0.01)


def test_write_mtp_streaming_f32_to_bf16(tmp_path: Path) -> None:
    from skulk_weights_publisher.mtp_extractor import _write_mtp_streaming

    f32 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
    shard = _make_shard(tmp_path, [("mtp.w.weight", "F32", [4], f32)])
    out = tmp_path / "out.safetensors"

    _write_mtp_streaming([shard], out, "mtp.", [].append)

    tensors = _decode_bf16_output(out)
    np.testing.assert_allclose(tensors["mtp.w.weight"], [1, 2, 3, 4], rtol=0.01)


def test_write_mtp_streaming_fp8_with_e8m0_scale(tmp_path: Path) -> None:
    """F8_E4M3 + F8_E8M0 scale → dequantised BF16 (V4-Flash style)."""
    from skulk_weights_publisher.mtp_extractor import _write_mtp_streaming

    weight_bytes = bytes([0x38, 0x40])  # FP8: 1.0, 2.0
    scale_bytes = bytes([128])          # E8M0 128 → 2^1 = 2.0
    shard = _make_shard(tmp_path, [
        ("mtp.0.w.weight", "F8_E4M3", [2], weight_bytes),
        ("mtp.0.w.scale",  "F8_E8M0", [1], scale_bytes),
    ])
    out = tmp_path / "out.safetensors"

    n = _write_mtp_streaming([shard], out, "mtp.", [].append)

    assert n == 1  # scale key must not be counted
    tensors = _decode_bf16_output(out)
    assert set(tensors) == {"mtp.0.w.weight"}
    np.testing.assert_allclose(tensors["mtp.0.w.weight"], [2.0, 4.0], rtol=0.05)


def test_write_mtp_streaming_i8_with_e8m0_scale(tmp_path: Path) -> None:
    """I8 + F8_E8M0 scale → dequantised BF16 (V4-Pro expert style)."""
    from skulk_weights_publisher.mtp_extractor import _write_mtp_streaming

    weight_bytes = struct.pack("<2b", 3, 4)
    scale_bytes = bytes([127])  # E8M0 127 → 2^0 = 1.0
    shard = _make_shard(tmp_path, [
        ("mtp.0.ffn.w1.weight", "I8",      [2], weight_bytes),
        ("mtp.0.ffn.w1.scale",  "F8_E8M0", [1], scale_bytes),
    ])
    out = tmp_path / "out.safetensors"

    _write_mtp_streaming([shard], out, "mtp.", [].append)

    tensors = _decode_bf16_output(out)
    assert set(tensors) == {"mtp.0.ffn.w1.weight"}
    np.testing.assert_allclose(tensors["mtp.0.ffn.w1.weight"], [3.0, 4.0], rtol=0.05)


def test_write_mtp_streaming_v3_scale_inv(tmp_path: Path) -> None:
    """F8_E4M3 + F32 weight_scale_inv → dequantised BF16 (V3 style)."""
    from skulk_weights_publisher.mtp_extractor import _write_mtp_streaming

    weight_bytes = bytes([0x38, 0x40])        # FP8: 1.0, 2.0
    scale_inv_bytes = struct.pack("<f", 0.5)  # 1/scale → scale = 2.0
    shard = _make_shard(tmp_path, [
        ("model.layers.61.attn.w.weight",           "F8_E4M3", [2], weight_bytes),
        ("model.layers.61.attn.w.weight_scale_inv", "F32",     [1], scale_inv_bytes),
    ])
    out = tmp_path / "out.safetensors"

    n = _write_mtp_streaming([shard], out, "model.layers.61.", [].append)

    assert n == 1
    tensors = _decode_bf16_output(out)
    assert set(tensors) == {"model.layers.61.attn.w.weight"}
    np.testing.assert_allclose(
        tensors["model.layers.61.attn.w.weight"], [2.0, 4.0], rtol=0.05
    )


def test_write_mtp_streaming_bf16_scale_excluded(tmp_path: Path) -> None:
    """A .scale companion with non-F8_E8M0 dtype must not appear in the output.

    Regression: the original scale_keys set only excluded F8_E8M0 and _scale_inv
    keys, so a BF16 .scale companion would pass through as a spurious tensor.
    """
    from skulk_weights_publisher.mtp_extractor import _write_mtp_streaming

    weight_bytes = bytes([0x38, 0x40])         # FP8: 1.0, 2.0
    scale_bytes = struct.pack("<H", 0x4000)    # BF16 2.0 (not F8_E8M0)
    shard = _make_shard(tmp_path, [
        ("mtp.0.attn.w.weight", "F8_E4M3", [2], weight_bytes),
        ("mtp.0.attn.w.scale",  "BF16",    [1], scale_bytes),
    ])
    out = tmp_path / "out.safetensors"

    n = _write_mtp_streaming([shard], out, "mtp.", [].append)

    assert n == 1
    tensors = _decode_bf16_output(out)
    assert "mtp.0.attn.w.scale" not in tensors


def test_write_mtp_streaming_raises_on_no_mtp_tensors(tmp_path: Path) -> None:
    from skulk_weights_publisher.mtp_extractor import (
        MtpExtractionError,
        _write_mtp_streaming,
    )

    non_mtp = struct.pack("<2H", 0x3F80, 0x4000)
    shard = _make_shard(tmp_path, [("model.embed.weight", "BF16", [2], non_mtp)])
    out = tmp_path / "out.safetensors"

    with pytest.raises(MtpExtractionError, match="no MTP tensors found"):
        _write_mtp_streaming([shard], out, "mtp.", [].append)


def test_write_mtp_streaming_multi_shard(tmp_path: Path) -> None:
    """Tensors from two separate shards are merged into one output file."""
    from skulk_weights_publisher.mtp_extractor import _write_mtp_streaming

    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    shard_a = _make_shard(
        dir_a, [("mtp.a.weight", "BF16", [2], struct.pack("<2H", 0x3F80, 0x4000))]
    )
    shard_b = _make_shard(
        dir_b, [("mtp.b.weight", "BF16", [2], struct.pack("<2H", 0x4040, 0x4080))]
    )
    out = tmp_path / "out.safetensors"

    n = _write_mtp_streaming([shard_a, shard_b], out, "mtp.", [].append)

    assert n == 2
    tensors = _decode_bf16_output(out)
    assert set(tensors) == {"mtp.a.weight", "mtp.b.weight"}
    np.testing.assert_allclose(tensors["mtp.a.weight"], [1.0, 2.0], rtol=0.01)
    np.testing.assert_allclose(tensors["mtp.b.weight"], [3.0, 4.0], rtol=0.01)


def test_write_mtp_streaming_2d_block_scale(tmp_path: Path) -> None:
    """I8 weight with 2D block scale (non-divisible dims, V3.2 MTP style).

    Uses [96, 65] with block_size=32 (Rb=3, Cb=3, 9 E8M0 scales) — the same
    shape class as DeepSeek-V3.2-Exp-Base [7168, 576] / 280 scales.
    """
    from skulk_weights_publisher.mtp_extractor import _write_mtp_streaming

    R, C = 96, 65
    # All I8 weights = 1; all E8M0 scales = byte 127 → 2^0 = 1.0
    weight_bytes = struct.pack(f"<{R * C}b", *([1] * (R * C)))
    scale_bytes = bytes([127] * 9)  # 9 scales (Rb*Cb=3*3), each = 1.0
    shard = _make_shard(tmp_path, [
        ("model.layers.61.w.weight", "I8",      [R, C], weight_bytes),
        ("model.layers.61.w.weight_scale_inv", "F8_E8M0", [9], scale_bytes),
    ])
    out = tmp_path / "out.safetensors"

    n = _write_mtp_streaming([shard], out, "model.layers.61.", [].append)

    assert n == 1
    tensors = _decode_bf16_output(out)
    assert tensors["model.layers.61.w.weight"].shape == (R, C)
    np.testing.assert_allclose(tensors["model.layers.61.w.weight"], 1.0, rtol=0.05)
