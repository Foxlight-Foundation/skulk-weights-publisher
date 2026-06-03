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


def _make_shard(tmp_path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> Path:
    """Write a minimal safetensors shard with the given tensors."""
    offset = 0
    header: dict[str, Any] = {}
    parts: list[bytes] = []
    for key, dtype, shape, data in tensors:
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
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
