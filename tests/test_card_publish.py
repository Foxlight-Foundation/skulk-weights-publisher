from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from skulk_weights_publisher.card_publish import (
    SourceProvenance,
    publish_model_card,
    resolve_source_provenance,
)


def test_resolve_source_provenance_reads_sha_and_license() -> None:
    info = SimpleNamespace(
        sha="abc123def456",
        cardData={"license": "apache-2.0"},
    )
    with patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.return_value = info
        prov = resolve_source_provenance("Qwen/Qwen3.6-35B-A3B", token="t")

    assert prov.revision == "abc123def456"
    assert prov.license == "apache-2.0"


def test_resolve_source_provenance_custom_license_fields() -> None:
    info = SimpleNamespace(
        sha="sha",
        cardData={
            "license": "other",
            "license_name": "gemma",
            "license_link": "https://ai.google.dev/gemma/terms",
        },
    )
    with patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.return_value = info
        prov = resolve_source_provenance("google/gemma-3-4b-it", token=None)

    assert prov.license == "other"
    assert prov.license_name == "gemma"
    assert prov.license_link == "https://ai.google.dev/gemma/terms"


class _ModelCardData:
    """Mimics huggingface_hub's ModelCardData: a to_dict()-able object, not a dict."""

    def __init__(self, **fields: object) -> None:
        self._fields = fields

    def to_dict(self) -> dict[str, object]:
        return dict(self._fields)


def test_resolve_source_provenance_reads_modelcarddata_not_just_dict() -> None:
    # huggingface_hub>=1.0 returns card_data as a ModelCardData object; the
    # inherited license must still be extracted (regression guard for PR #27).
    info = SimpleNamespace(
        sha="sha",
        card_data=_ModelCardData(license="gemma", license_link="https://x/terms"),
    )
    with patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.return_value = info
        prov = resolve_source_provenance("google/gemma-3-4b-it", token=None)

    assert prov.revision == "sha"
    assert prov.license == "gemma"
    assert prov.license_link == "https://x/terms"


def test_resolve_source_provenance_is_best_effort_on_error() -> None:
    with patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.side_effect = RuntimeError("offline")
        prov = resolve_source_provenance("org/x", token=None)

    assert prov == SourceProvenance()


def test_resolve_source_provenance_handles_missing_carddata() -> None:
    info = SimpleNamespace(sha="sha", cardData=None)
    with patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.return_value = info
        prov = resolve_source_provenance("org/x", token=None)

    assert prov.revision == "sha"
    assert prov.license is None


def test_publish_model_card_uploads_readme_with_provenance() -> None:
    uploaded: dict[str, Any] = {}

    def fake_upload_file(**kw: Any) -> None:
        uploaded.update(kw)

    prov = SourceProvenance(revision="abc123", license="apache-2.0")
    with (
        patch("huggingface_hub.upload_file", side_effect=fake_upload_file),
        patch(
            "skulk_weights_publisher.card_publish.resolve_source_provenance",
            return_value=prov,
        ),
    ):
        publish_model_card(
            repo_id="FoxlightAI/qwen3-6-35b-a3b-mtp-q4k",
            artifact_type="mtp-sidecar",
            source_repo="Qwen/Qwen3.6-35B-A3B",
            token="t",
            quant="q4k",
            catalog_key="foxlight/qwen3-6-35b-a3b-full-q4-k",
            weight_filename="mtp.safetensors",
        )

    assert uploaded["path_in_repo"] == "README.md"
    assert uploaded["repo_id"] == "FoxlightAI/qwen3-6-35b-a3b-mtp-q4k"
    assert uploaded["repo_type"] == "model"
    content = uploaded["path_or_fileobj"].decode("utf-8")
    assert "base_model: Qwen/Qwen3.6-35B-A3B" in content
    assert "abc123" in content  # pinned source revision
    assert "apache-2.0" in content  # inherited license
    assert "not a standalone model" in content


def test_publish_model_card_target_defaults_to_source() -> None:
    uploaded: dict[str, Any] = {}

    def _capture(**kw: Any) -> None:
        uploaded.update(kw)

    with (
        patch("huggingface_hub.upload_file", side_effect=_capture),
        patch(
            "skulk_weights_publisher.card_publish.resolve_source_provenance",
            return_value=SourceProvenance(),
        ),
    ):
        publish_model_card(
            repo_id="FoxlightAI/x-vindex",
            artifact_type="vindex",
            source_repo="org/x",
            token=None,
            quant="q4k",
        )

    content = uploaded["path_or_fileobj"].decode("utf-8")
    assert "target_model: org/x" in content  # defaulted from source_repo
