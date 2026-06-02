"""Unit tests for mtp_extractor internal helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import skulk_weights_publisher.mtp_extractor as mtp_mod
from skulk_weights_publisher.mtp_extractor import (
    MtpExtractionError,
    _print_dry_run_plan,
    _quant_bits,
    _sidecar_filename,
    extract_mtp,
)


def test_extract_mtp_skips_when_sidecar_already_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When the sidecar exists, extract_mtp informs the operator that it already
    # covers the model (and all its quantizations) and skips — no re-extraction.
    monkeypatch.setattr(mtp_mod, "_sidecar_already_published", lambda *a, **k: True)
    logs: list[str] = []

    extract_mtp(
        "Qwen/Qwen3.6-35B-A3B",
        "FoxlightAI/qwen3-6-35b-a3b-mtp-q4-k",
        "q4k",
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
# _quant_bits
# ---------------------------------------------------------------------------


def test_quant_bits_q4k() -> None:
    assert _quant_bits("q4k") == 4


def test_quant_bits_q8k() -> None:
    assert _quant_bits("q8k") == 8


def test_quant_bits_unknown_raises() -> None:
    with pytest.raises(MtpExtractionError, match="unsupported mtp_quant"):
        _quant_bits("q99z")


# ---------------------------------------------------------------------------
# _sidecar_filename
# ---------------------------------------------------------------------------


def test_sidecar_filename_replaces_slash() -> None:
    expected = "acme--model-mtp-int4-mtp.safetensors"
    assert _sidecar_filename("acme/model-mtp-int4") == expected


def test_sidecar_filename_no_slash() -> None:
    assert _sidecar_filename("modelonly") == "modelonly-mtp.safetensors"


# ---------------------------------------------------------------------------
# _print_dry_run_plan
# ---------------------------------------------------------------------------


def test_print_dry_run_plan_output(capsys: pytest.CaptureFixture[str]) -> None:
    _print_dry_run_plan(
        "owner/source-bf16",
        "owner/sidecar-int4",
        "q4k",
        Path("/scratch/sidecar.safetensors"),
    )
    out = capsys.readouterr().out
    assert "owner/source-bf16" in out
    assert "owner/sidecar-int4" in out
    assert "q4k" in out
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
        shards = _find_mtp_shards("owner/repo", token=None)

    assert shards == sorted(
        {"model-00003-of-00010.safetensors", "model-00004-of-00010.safetensors"}
    )


def test_find_mtp_shards_no_mtp_keys_in_index(tmp_path: Path) -> None:
    from skulk_weights_publisher.mtp_extractor import _find_mtp_shards

    index = {"weight_map": {"model.layer.weight": "model-00001-of-00002.safetensors"}}
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with patch("huggingface_hub.hf_hub_download", return_value=str(index_path)):
        shards = _find_mtp_shards("owner/repo", token=None)

    assert shards == []


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
        shards = _find_mtp_shards("owner/repo", token=None)

    assert shards == ["model.safetensors"]


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
        shards = _find_mtp_shards("owner/repo", token=None)

    assert shards == []


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
        shards = _find_mtp_shards("owner/repo", token=None)

    assert shards == []


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
