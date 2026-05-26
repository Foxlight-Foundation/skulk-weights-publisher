from __future__ import annotations

from pathlib import Path

import pytest

from skulk_weights_publisher.manifest import (
    ManifestError,
    find_entry,
    list_entries,
    validate_manifest,
)


def test_validate_manifest_loads_catalogue() -> None:
    entries = validate_manifest(Path("models.yaml"))

    assert len(entries) == 9
    assert entries[0].key == "gemma-3-4b-full-q4-k"
    assert entries[0].publish_slices == "none"
    assert entries[0].hf_collection == "FoxlightAI/vindexes-6a124406dd5fb439c431b051"


def test_list_entries_filters_by_tier() -> None:
    smoke_entries = list_entries("smoke", Path("models.yaml"))
    moe_entries = list_entries("moe", Path("models.yaml"))

    assert [entry.key for entry in smoke_entries] == [
        "gemma-3-4b-full-q4-k",
        "llama-3-2-3b-full-q4-k",
        "qwen-2-5-7b-full-q4-k",
    ]
    assert len(moe_entries) == 6


def test_find_entry_returns_json_serializable_payload() -> None:
    entry = find_entry("gemma-4-26b-a4b-expert-server-q4-k", Path("models.yaml"))

    assert entry.to_dict()["slices"] == ["expert-server"]
    assert entry.publish_slices == "expert-server"
    assert "gemma-4-26b-a4b-it" in entry.to_json()


def test_duplicate_key_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - key: duplicate
    source_model: owner/model-a
    quant: q4k
    tier: smoke
    slices: [full]
    output_name: model-a.vindex
    hf_repo: FoxlightAI/model-a
  - key: duplicate
    source_model: owner/model-b
    quant: q4k
    tier: smoke
    slices: [full]
    output_name: model-b.vindex
    hf_repo: FoxlightAI/model-b
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="duplicate key"):
        validate_manifest(manifest)


def test_default_hugging_face_collection_is_applied(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - key: model-a
    source_model: owner/model-a
    quant: q4k
    tier: smoke
    slices: [full]
    output_name: model-a.vindex
    hf_repo: acme/model-a
""",
        encoding="utf-8",
    )

    entries = validate_manifest(
        manifest,
        hf_owner="acme",
        hf_collection="acme/vindexes-0123456789abcdef01234567",
    )

    assert entries[0].hf_collection == "acme/vindexes-0123456789abcdef01234567"


def test_hugging_face_collection_owner_must_match_config(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - key: model-a
    source_model: owner/model-a
    quant: q4k
    tier: smoke
    slices: [full]
    output_name: model-a.vindex
    hf_repo: acme/model-a
    hf_collection: other/vindexes-0123456789abcdef01234567
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="hf_collection owner must be 'acme'"):
        validate_manifest(manifest, hf_owner="acme")
