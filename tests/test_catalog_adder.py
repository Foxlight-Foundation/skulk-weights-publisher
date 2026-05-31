from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from skulk_weights_publisher.catalog_adder import (
    CatalogAddError,
    build_entry_block,
    derive_artifact_slug,
    derive_key_slug,
    detect_base_model,
    detect_mtp_keys,
    detect_quant,
    detect_tier,
    parse_hf_model_id,
)

# ── parse_hf_model_id ────────────────────────────────────────────────────────


def test_parse_hf_model_id_bare() -> None:
    assert parse_hf_model_id("owner/repo") == "owner/repo"


def test_parse_hf_model_id_url() -> None:
    assert (
        parse_hf_model_id("https://huggingface.co/mlx-community/Foo-4bit")
        == "mlx-community/Foo-4bit"
    )


def test_parse_hf_model_id_url_trailing_slash() -> None:
    assert (
        parse_hf_model_id("https://huggingface.co/mlx-community/Foo-4bit/")
        == "mlx-community/Foo-4bit"
    )


def test_parse_hf_model_id_no_slash_raises() -> None:
    with pytest.raises(CatalogAddError, match="expected owner/repo"):
        parse_hf_model_id("justarepo")


def test_parse_hf_model_id_url_no_repo_raises() -> None:
    with pytest.raises(CatalogAddError, match="cannot parse model ID"):
        parse_hf_model_id("https://huggingface.co/mlx-community")


# ── detect_base_model ────────────────────────────────────────────────────────


def test_detect_base_model_quantized_tag() -> None:
    info = {"tags": ["base_model:quantized:Qwen/Qwen3.5-9B"]}
    assert detect_base_model(info) == "Qwen/Qwen3.5-9B"


def test_detect_base_model_plain_tag() -> None:
    info = {"tags": ["base_model:Qwen/Qwen3.5-9B"]}
    assert detect_base_model(info) == "Qwen/Qwen3.5-9B"


def test_detect_base_model_quantized_wins_over_plain() -> None:
    info = {"tags": ["base_model:Foo/Bar", "base_model:quantized:Real/Base"]}
    assert detect_base_model(info) == "Real/Base"


def test_detect_base_model_no_slash_skipped() -> None:
    info = {"tags": ["base_model:notamodel"]}
    assert detect_base_model(info) is None


def test_detect_base_model_none_when_no_tags() -> None:
    assert detect_base_model({}) is None


# ── detect_quant ─────────────────────────────────────────────────────────────


def test_detect_quant_defaults_to_q4k() -> None:
    assert detect_quant({"id": "mlx-community/Foo-4bit", "tags": []}) == "q4k"


def test_detect_quant_q8_from_id() -> None:
    assert detect_quant({"id": "mlx-community/Foo-8bit", "tags": []}) == "q8k"


def test_detect_quant_q8_from_tag() -> None:
    assert detect_quant({"id": "mlx-community/Foo", "tags": ["8-bit"]}) == "q8k"


def test_detect_quant_q8k_tag_keyword() -> None:
    assert detect_quant({"id": "mlx-community/Foo", "tags": ["q8"]}) == "q8k"


# ── detect_tier ──────────────────────────────────────────────────────────────


def test_detect_tier_moe_from_tag() -> None:
    assert detect_tier({"tags": ["moe"]}) == "moe"


def test_detect_tier_mixtral() -> None:
    assert detect_tier({"tags": ["mixtral"]}) == "moe"


def test_detect_tier_qwen_moe() -> None:
    assert detect_tier({"tags": ["qwen3_5_moe"]}) == "moe"


def test_detect_tier_smoke_default() -> None:
    assert detect_tier({"tags": ["text-generation"]}) == "smoke"


# ── derive_key_slug ──────────────────────────────────────────────────────────


def test_derive_key_slug_strips_4bit() -> None:
    assert derive_key_slug("mlx-community/Qwen3.6-35B-A3B-4bit", "q4k") == (
        "qwen3-6-35b-a3b-full-q4-k"
    )


def test_derive_key_slug_strips_instruct() -> None:
    assert derive_key_slug("mlx-community/Gemma-3-4B-Instruct-4bit", "q4k") == (
        "gemma-3-4b-full-q4-k"
    )


def test_derive_key_slug_q8k() -> None:
    assert derive_key_slug("mlx-community/Qwen-7B-8bit", "q8k") == (
        "qwen-7b-full-q8-k"
    )


def test_derive_key_slug_normalises_dots() -> None:
    assert derive_key_slug("mlx-community/Llama-3.2-3B", "q4k") == (
        "llama-3-2-3b-full-q4-k"
    )


def test_derive_key_slug_normalises_underscores() -> None:
    assert derive_key_slug("owner/Foo_Bar-4bit", "q4k") == "foo-bar-full-q4-k"


# ── derive_artifact_slug ─────────────────────────────────────────────────────


def test_derive_artifact_slug_retains_it_from_instruct() -> None:
    assert derive_artifact_slug("mlx-community/Gemma-3-4B-Instruct-4bit", "q4k") == (
        "gemma-3-4b-it-full-q4-k"
    )


def test_derive_artifact_slug_retains_it_suffix() -> None:
    assert derive_artifact_slug("mlx-community/Llama-3.2-3B-Instruct-4bit", "q4k") == (
        "llama-3-2-3b-it-full-q4-k"
    )


def test_derive_artifact_slug_no_instruct_unchanged() -> None:
    assert derive_artifact_slug("mlx-community/Qwen3.6-35B-A3B-4bit", "q4k") == (
        "qwen3-6-35b-a3b-full-q4-k"
    )


# ── build_entry_block ────────────────────────────────────────────────────────


def test_build_entry_block_no_mtp() -> None:
    block = build_entry_block(
        key_slug="gemma-3-4b-full-q4-k",
        source_model="mlx-community/gemma-3-4b-it",
        quant="q4k",
        tier="smoke",
        base_model=None,
        mtp_keys=[],
    )
    assert "key: gemma-3-4b-full-q4-k" in block
    assert "mtp_source_repo" not in block
    assert "hf_collection: FoxlightAI/" in block


def test_build_entry_block_artifact_slug_used_for_output_and_repo() -> None:
    block = build_entry_block(
        key_slug="gemma-3-4b-full-q4-k",
        artifact_slug="gemma-3-4b-it-full-q4-k",
        source_model="mlx-community/gemma-3-4b-it-4bit",
        quant="q4k",
        tier="smoke",
        base_model=None,
        mtp_keys=[],
    )
    assert "key: gemma-3-4b-full-q4-k" in block
    assert "output_name: gemma-3-4b-it-full-q4-k.vindex" in block
    assert "hf_repo: FoxlightAI/gemma-3-4b-it-full-q4-k-vindex" in block


def test_build_entry_block_with_mtp() -> None:
    block = build_entry_block(
        key_slug="qwen3-5-9b-full-q4-k",
        source_model="mlx-community/Qwen3.5-9B-4bit",
        quant="q4k",
        tier="smoke",
        base_model="Qwen/Qwen3.5-9B",
        mtp_keys=["mtp.fc.weight"],
    )
    assert "mtp_source_repo: Qwen/Qwen3.5-9B" in block
    assert "mtp_sidecar_repo: FoxlightAI/qwen3-5-9b-mtp" in block
    assert "mtp_quant: q4k" in block


def test_build_entry_block_starts_with_blank_line() -> None:
    block = build_entry_block(
        key_slug="x-full-q4-k",
        source_model="a/b",
        quant="q4k",
        tier="smoke",
        base_model=None,
        mtp_keys=[],
    )
    assert block.startswith("\n")


# ── detect_mtp_keys ──────────────────────────────────────────────────────────


def test_detect_mtp_keys_returns_matching_keys() -> None:
    index = {
        "weight_map": {
            "mtp.fc.weight": "shard-0.safetensors",
            "model.embed_tokens.weight": "shard-0.safetensors",
        }
    }
    bio = io.BytesIO(json.dumps(index).encode())
    bio.__enter__ = lambda s: s  # type: ignore[method-assign]
    bio.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=bio):
        keys = detect_mtp_keys("Qwen/Qwen3.5-9B")

    assert keys == ["mtp.fc.weight"]


def test_detect_mtp_keys_returns_empty_on_http_error() -> None:
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(None, 404, "Not Found", {}, None),  # type: ignore[arg-type]
    ):
        assert detect_mtp_keys("some/model") == []


def test_detect_mtp_keys_returns_empty_on_generic_error() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("network")):
        assert detect_mtp_keys("some/model") == []
