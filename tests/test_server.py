"""Tests for the skulk-ui FastAPI server (the [ui] extra)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# The server requires the [ui] extras (fastapi). Skip cleanly if absent.
if importlib.util.find_spec("fastapi") is None:  # pragma: no cover
    pytest.skip("fastapi not installed ([ui] extra)", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

import skulk_weights_publisher.server.app as app_module  # noqa: E402
from skulk_weights_publisher.server.app import app  # noqa: E402

client = TestClient(app)


def test_register_appends_assistant_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/register actually writes a catalog entry (not a no-op)."""
    catalog = tmp_path / "foxlight.yaml"
    catalog.write_text("models: []\n", encoding="utf-8")

    info = {
        "id": "mlx-community/gemma-4-27b-it-4bit",
        "tags": ["base_model:quantized:google/gemma-4-27b-it"],
    }
    monkeypatch.setattr(app_module, "fetch_hf_model_info", lambda *a, **k: info)
    monkeypatch.setattr(
        app_module, "resolve_base_model", lambda *a, **k: "google/gemma-4-27b-it"
    )
    monkeypatch.setattr(
        app_module, "detect_base_model", lambda *a, **k: "google/gemma-4-27b-it"
    )
    monkeypatch.setattr(app_module, "detect_mtp_keys", lambda *a, **k: [])
    monkeypatch.setattr(
        app_module,
        "detect_assistant_model",
        lambda *a, **k: "google/gemma-4-27b-it-assistant",
    )
    monkeypatch.setattr(app_module, "find_builtin_catalog_path", lambda: catalog)

    # Empty merged catalog so no collisions.
    class _View:
        entries: tuple[object, ...] = ()

    monkeypatch.setattr(app_module, "load_catalogue_view", lambda *a, **k: _View())

    resp = client.post(
        "/api/register",
        json={"url": "mlx-community/gemma-4-27b-it-4bit"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["assistant_model_repo"] == "google/gemma-4-27b-it-assistant"

    # The file was actually appended to, with the assistant field.
    written = catalog.read_text(encoding="utf-8")
    assert "assistant_model_repo: google/gemma-4-27b-it-assistant" in written
    assert "mtp_source_repo" not in written


def test_register_rejects_duplicate_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration is blocked when the catalog key already exists."""
    catalog = tmp_path / "foxlight.yaml"
    catalog.write_text("models: []\n", encoding="utf-8")
    before = catalog.read_text(encoding="utf-8")

    info = {"id": "mlx-community/gemma-4-27b-it-4bit", "tags": []}
    monkeypatch.setattr(app_module, "fetch_hf_model_info", lambda *a, **k: info)
    monkeypatch.setattr(
        app_module, "resolve_base_model", lambda *a, **k: "google/gemma-4-27b-it"
    )
    monkeypatch.setattr(
        app_module, "detect_base_model", lambda *a, **k: "google/gemma-4-27b-it"
    )
    monkeypatch.setattr(app_module, "detect_mtp_keys", lambda *a, **k: [])
    monkeypatch.setattr(
        app_module,
        "detect_assistant_model",
        lambda *a, **k: "google/gemma-4-27b-it-assistant",
    )
    monkeypatch.setattr(app_module, "find_builtin_catalog_path", lambda: catalog)

    class _Entry:
        key = "foxlight/gemma-4-27b-full-q4-k"
        hf_repo = "x/y"
        output_name = "z.vindex"

    class _View:
        entries = (_Entry(),)

    monkeypatch.setattr(app_module, "load_catalogue_view", lambda *a, **k: _View())

    resp = client.post(
        "/api/register",
        json={"url": "mlx-community/gemma-4-27b-it-4bit"},
    )

    assert resp.status_code == 409
    assert "already exists" in resp.json()["error"]
    # Nothing was written.
    assert catalog.read_text(encoding="utf-8") == before
