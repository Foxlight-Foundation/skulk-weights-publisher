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


def _mtp_manifest(tmp_path: Path, *, extra: str = "") -> Path:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        f"""
models:
  - key: model-a
    source_model: owner/model-a
    quant: q4k
    tier: smoke
    slices: [full]
    output_name: model-a.vindex
    hf_repo: acme/model-a
    mtp_source_repo: owner/model-a-bf16
    mtp_sidecar_repo: acme/model-a-mtp-int4
    mtp_quant: q4k
{extra}
""",
        encoding="utf-8",
    )
    return manifest


def test_mtp_fields_accepted(tmp_path: Path) -> None:
    entries = validate_manifest(_mtp_manifest(tmp_path))

    assert entries[0].mtp_source_repo == "owner/model-a-bf16"
    assert entries[0].mtp_sidecar_repo == "acme/model-a-mtp-int4"
    assert entries[0].mtp_quant == "q4k"


def test_mtp_fields_absent_leaves_none(tmp_path: Path) -> None:
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

    entries = validate_manifest(manifest)

    assert entries[0].mtp_source_repo is None
    assert entries[0].mtp_sidecar_repo is None
    assert entries[0].mtp_quant is None


def test_mtp_partial_fields_rejected(tmp_path: Path) -> None:
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
    mtp_source_repo: owner/model-a-bf16
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ManifestError, match="mtp_source_repo, mtp_sidecar_repo, and mtp_quant"
    ):
        validate_manifest(manifest)


def test_mtp_q8k_accepted(tmp_path: Path) -> None:
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
    mtp_source_repo: owner/model-a-bf16
    mtp_sidecar_repo: acme/model-a-mtp-q8k
    mtp_quant: q8k
""",
        encoding="utf-8",
    )

    entries = validate_manifest(manifest)

    assert entries[0].mtp_quant == "q8k"


def test_mtp_bad_quant_rejected(tmp_path: Path) -> None:
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
    mtp_source_repo: owner/model-a-bf16
    mtp_sidecar_repo: acme/model-a-mtp-int4
    mtp_quant: q99z
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="unsupported mtp_quant"):
        validate_manifest(manifest)


def test_mtp_bad_repo_pattern_rejected(tmp_path: Path) -> None:
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
    mtp_source_repo: not-a-valid-repo
    mtp_sidecar_repo: acme/model-a-mtp-int4
    mtp_quant: q4k
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ManifestError, match="mtp_source_repo must look like owner/name"
    ):
        validate_manifest(manifest)


def test_mtp_sidecar_owner_must_match_hf_repo_owner(tmp_path: Path) -> None:
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
    mtp_source_repo: owner/model-a-bf16
    mtp_sidecar_repo: other/model-a-mtp-int4
    mtp_quant: q4k
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="mtp_sidecar_repo owner must be 'acme'"):
        validate_manifest(manifest)


def test_mtp_partial_fields_two_set_rejected(tmp_path: Path) -> None:
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
    mtp_source_repo: acme/model-a-bf16
    mtp_sidecar_repo: acme/model-a-mtp-int4
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ManifestError, match="mtp_source_repo, mtp_sidecar_repo, and mtp_quant"
    ):
        validate_manifest(manifest)


def test_mtp_same_source_and_sidecar_still_requires_all_three(tmp_path: Path) -> None:
    """Set deduplication must not mask partial config when source == sidecar."""
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
    mtp_source_repo: acme/model-a
    mtp_sidecar_repo: acme/model-a
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ManifestError, match="mtp_source_repo, mtp_sidecar_repo, and mtp_quant"
    ):
        validate_manifest(manifest)


def _assistant_manifest(
    tmp_path: Path, *, assistant_repo: str = "google/gemma-4-27b-it-assistant"
) -> Path:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        f"""
models:
  - key: model-a
    source_model: owner/model-a
    quant: q4k
    tier: moe
    slices: [full]
    output_name: model-a.vindex
    hf_repo: acme/model-a
    assistant_model_repo: {assistant_repo}
""",
        encoding="utf-8",
    )
    return manifest


def test_assistant_model_repo_accepted(tmp_path: Path) -> None:
    entries = validate_manifest(_assistant_manifest(tmp_path))

    assert entries[0].assistant_model_repo == "google/gemma-4-27b-it-assistant"
    assert entries[0].mtp_source_repo is None


def test_assistant_model_repo_absent_leaves_none(tmp_path: Path) -> None:
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
    entries = validate_manifest(manifest)
    assert entries[0].assistant_model_repo is None


def test_assistant_model_repo_bad_pattern_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        ManifestError, match="assistant_model_repo must look like owner/name"
    ):
        validate_manifest(
            _assistant_manifest(tmp_path, assistant_repo="not-a-valid-repo")
        )


def test_assistant_model_repo_and_mtp_mutually_exclusive(tmp_path: Path) -> None:
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
    mtp_source_repo: owner/model-a-bf16
    mtp_sidecar_repo: acme/model-a-mtp-q4k
    mtp_quant: q4k
    assistant_model_repo: google/gemma-4-27b-it-assistant
""",
        encoding="utf-8",
    )
    with pytest.raises(
        ManifestError,
        match="assistant_model_repo and mtp_source_repo are mutually exclusive",
    ):
        validate_manifest(manifest)


def test_assistant_model_repo_in_to_dict(tmp_path: Path) -> None:
    entries = validate_manifest(_assistant_manifest(tmp_path))
    d = entries[0].to_dict()
    assert d["assistant_model_repo"] == "google/gemma-4-27b-it-assistant"


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
