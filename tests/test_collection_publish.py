from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from skulk_weights_publisher.collection_publish import (
    COLLECTION_TITLES,
    CollectionError,
    ensure_collection,
    file_artifact_in_collection,
)


def test_collection_titles_cover_all_artifact_types() -> None:
    assert COLLECTION_TITLES == {
        "vindex": "Vindexes",
        "mtp-sidecar": "MTP Sidecars",
        "vision-sidecar": "Vision Sidecars",
    }


def test_ensure_collection_creates_if_missing_and_returns_slug() -> None:
    created: dict[str, Any] = {}

    def fake_create(**kw: Any) -> SimpleNamespace:
        created.update(kw)
        return SimpleNamespace(slug="FoxlightAI/mtp-sidecars-abc123")

    with patch("huggingface_hub.create_collection", side_effect=fake_create):
        slug = ensure_collection("MTP Sidecars", "FoxlightAI", token="t")

    assert slug == "FoxlightAI/mtp-sidecars-abc123"
    assert created["title"] == "MTP Sidecars"
    assert created["namespace"] == "FoxlightAI"
    assert created["exists_ok"] is True


def test_ensure_collection_wraps_failures() -> None:
    with patch(
        "huggingface_hub.create_collection", side_effect=RuntimeError("boom")
    ), pytest.raises(CollectionError, match="failed to ensure collection"):
        ensure_collection("Vindexes", "FoxlightAI", token=None)


def test_file_artifact_ensures_collection_and_adds_item() -> None:
    added: dict[str, Any] = {}

    def fake_create(**kw: Any) -> SimpleNamespace:
        return SimpleNamespace(slug="FoxlightAI/vindexes-xyz")

    def fake_add(slug: str, **kw: Any) -> None:
        added["slug"] = slug
        added.update(kw)

    with (
        patch("huggingface_hub.create_collection", side_effect=fake_create),
        patch("huggingface_hub.add_collection_item", side_effect=fake_add),
    ):
        file_artifact_in_collection(
            "FoxlightAI/gemma-3-4b-it-full-q4-k-vindex", "vindex", token="t"
        )

    assert added["slug"] == "FoxlightAI/vindexes-xyz"
    assert added["item_id"] == "FoxlightAI/gemma-3-4b-it-full-q4-k-vindex"
    assert added["item_type"] == "model"
    assert added["exists_ok"] is True


def test_file_artifact_honors_explicit_slug_without_ensuring() -> None:
    # When a slug is configured, add to it directly — don't create a new
    # title-based collection (preserves operator config + dry-run consistency).
    added: dict[str, Any] = {}
    create_called = {"n": 0}

    def fake_create(**kw: Any) -> SimpleNamespace:
        create_called["n"] += 1
        return SimpleNamespace(slug="should-not-be-used")

    with (
        patch("huggingface_hub.create_collection", side_effect=fake_create),
        patch(
            "huggingface_hub.add_collection_item",
            side_effect=lambda slug, **k: added.update({"slug": slug, **k}),
        ),
    ):
        file_artifact_in_collection(
            "acme/local-7b-full-q4-k-vindex",
            "vindex",
            token="t",
            collection_slug="acme/custom-vindexes-0123456789abcdef01234567",
        )

    assert create_called["n"] == 0  # ensure_collection NOT called
    assert added["slug"] == "acme/custom-vindexes-0123456789abcdef01234567"
    assert added["item_id"] == "acme/local-7b-full-q4-k-vindex"


def test_file_artifact_derives_owner_from_repo() -> None:
    seen: dict[str, Any] = {}

    def fake_create(**kw: Any) -> SimpleNamespace:
        seen.update(kw)
        return SimpleNamespace(slug="acme/mtp-sidecars-1")

    with (
        patch("huggingface_hub.create_collection", side_effect=fake_create),
        patch("huggingface_hub.add_collection_item", lambda *a, **k: None),
    ):
        file_artifact_in_collection(
            "acme/some-model-mtp-q4k", "mtp-sidecar", token=None
        )

    # Collection is created under the repo's own owner, titled per artifact type.
    assert seen["namespace"] == "acme"
    assert seen["title"] == "MTP Sidecars"


def test_file_artifact_rejects_unknown_type() -> None:
    with pytest.raises(CollectionError, match="no collection configured"):
        file_artifact_in_collection("FoxlightAI/x", "bogus", token=None)  # type: ignore[arg-type]


def test_file_artifact_rejects_repo_without_owner() -> None:
    with pytest.raises(CollectionError, match="cannot derive collection owner"):
        file_artifact_in_collection("no-slash-repo", "vindex", token=None)
