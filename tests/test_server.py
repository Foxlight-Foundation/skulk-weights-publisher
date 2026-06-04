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
        "find_assistant_model",
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
        "find_assistant_model",
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


def test_register_rejects_unsupported_quant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A q8 model is rejected before writing (manifest only allows q4k vindexes)."""
    catalog = tmp_path / "foxlight.yaml"
    catalog.write_text("models: []\n", encoding="utf-8")
    before = catalog.read_text(encoding="utf-8")

    info = {"id": "mlx-community/some-model-8bit", "tags": ["8-bit"]}
    monkeypatch.setattr(app_module, "fetch_hf_model_info", lambda *a, **k: info)
    monkeypatch.setattr(app_module, "find_builtin_catalog_path", lambda: catalog)

    resp = client.post(
        "/api/register",
        json={"url": "mlx-community/some-model-8bit"},
    )

    assert resp.status_code == 400
    assert "not supported" in resp.json()["error"]
    assert catalog.read_text(encoding="utf-8") == before


def test_ensure_built_honors_dist_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_ensure_built returns early when SKULK_UI_DIST has a built index.html."""
    from skulk_weights_publisher.server import _ensure_built

    dist = tmp_path / "prebuilt"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setenv("SKULK_UI_DIST", str(dist))

    # Should return without raising SystemExit or attempting a build, even with
    # no source ui/ tree present.
    _ensure_built()


def test_catalog_find_resolves_builtin_source() -> None:
    """POST /api/catalog/find returns the entries for a known source model."""
    resp = client.post("/api/catalog/find", json={"url": "google/gemma-3-4b-it"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["source_model"] == "google/gemma-3-4b-it"
    assert len(body["entries"]) == 1
    assert body["entries"][0]["key"] == "foxlight/gemma-3-4b-full-q4-k"
    assert body["entries"][0]["source_model"] == "google/gemma-3-4b-it"


def test_catalog_find_returns_all_matches() -> None:
    """A source model with multiple slices returns every matching entry."""
    resp = client.post(
        "/api/catalog/find", json={"url": "google/gemma-4-26b-a4b-it"}
    )

    assert resp.status_code == 200
    keys = {e["key"] for e in resp.json()["entries"]}
    assert keys == {
        "foxlight/gemma-4-26b-a4b-full-q4-k",
        "foxlight/gemma-4-26b-a4b-expert-server-q4-k",
    }


def test_catalog_find_accepts_full_url() -> None:
    """POST /api/catalog/find normalizes a full huggingface.co URL."""
    resp = client.post(
        "/api/catalog/find",
        json={"url": "https://huggingface.co/google/gemma-3-4b-it"},
    )

    assert resp.status_code == 200
    assert resp.json()["entries"][0]["key"] == "foxlight/gemma-3-4b-full-q4-k"


def test_catalog_find_load_failure_is_500_not_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed catalog is a 500 config error, not a 404 'no entry' miss."""
    from skulk_weights_publisher.manifest import ManifestError

    def boom() -> None:
        raise ManifestError("catalog is broken")

    monkeypatch.setattr(app_module, "load_catalogue_view", boom)
    resp = client.post("/api/catalog/find", json={"url": "google/gemma-3-4b-it"})

    assert resp.status_code == 500
    assert "catalog failed to load" in resp.json()["error"]


def test_catalog_find_non_manifest_load_failure_is_json_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-ManifestError (malformed YAML / OSError) still returns the JSON
    error envelope, not a bare 500 that breaks the UI's res.json()."""

    def boom() -> None:
        raise OSError("skulk-weights.yaml is unreadable")

    monkeypatch.setattr(app_module, "load_catalogue_view", boom)
    resp = client.post("/api/catalog/find", json={"url": "google/gemma-3-4b-it"})

    assert resp.status_code == 500
    assert "catalog failed to load" in resp.json()["error"]


def test_catalog_find_unknown_source_returns_404() -> None:
    """An unmatched source model yields 404 with a clear message."""
    resp = client.post("/api/catalog/find", json={"url": "does-not/exist"})

    assert resp.status_code == 404
    body = resp.json()
    assert "no catalog entry found for source_model" in body["error"]
    assert body["source_model"] == "does-not/exist"


def test_catalog_find_unparseable_input_returns_400() -> None:
    """An input that is neither owner/repo nor a URL yields 400."""
    resp = client.post("/api/catalog/find", json={"url": "not-a-valid-id"})

    assert resp.status_code == 400
    assert "expected owner/repo or a huggingface.co URL" in resp.json()["error"]


def test_catalog_find_empty_input_returns_400() -> None:
    """A blank query yields 400."""
    resp = client.post("/api/catalog/find", json={"url": "  "})

    assert resp.status_code == 400
    assert resp.json()["error"] == "url is required"


def test_save_token_is_owner_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The saved .env holding HF_TOKEN must be 0600, not world-readable."""
    import stat

    env_file = tmp_path / "cfg" / ".env"
    monkeypatch.setattr(app_module, "_CONFIG_DIR", env_file.parent)
    monkeypatch.setattr(app_module, "_ENV_FILE", env_file)

    resp = client.post("/api/config", json={"hf_token": "hf_secret"})
    assert resp.status_code == 200

    mode = stat.S_IMODE(env_file.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


# ---------------------------------------------------------------------------
# /api/publish + /api/stream job lifecycle
# ---------------------------------------------------------------------------

def _clear_jobs() -> None:
    with app_module._jobs_lock:
        app_module._jobs.clear()
        app_module._jobs_finished_at.clear()


def test_publish_without_token_returns_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No HF token → fail fast with 400, like detect/register, not a deep job error."""
    monkeypatch.setattr(app_module, "_get_token", lambda: None)

    resp = client.post(
        "/api/publish",
        json={"base_model": "owner/model", "sidecar_repo": "FoxlightAI/model-mtp"},
    )

    assert resp.status_code == 400
    assert "token" in resp.json()["error"].lower()


def test_publish_streams_lines_and_done_then_evicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: publish spawns a job whose log lines arrive over SSE,
    terminated by [done], after which the job is dropped from the registry."""
    _clear_jobs()
    monkeypatch.setattr(app_module, "_get_token", lambda: "hf_test")
    monkeypatch.setattr(app_module, "default_scratch_root", lambda: tmp_path)

    def fake_extract_mtp(*, source_repo, sidecar_repo, scratch_root, token, log):
        log("mtp: found MTP tensors in 1 shard(s) (prefix: 'mtp.')")
        log("mtp: published to hf://FoxlightAI/model-mtp/mtp.safetensors")

    monkeypatch.setattr(app_module, "extract_mtp", fake_extract_mtp)

    resp = client.post(
        "/api/publish",
        json={"base_model": "owner/model", "sidecar_repo": "FoxlightAI/model-mtp"},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    lines: list[str] = []
    with client.stream("GET", f"/api/stream/{job_id}") as stream_resp:
        assert stream_resp.status_code == 200
        for raw in stream_resp.iter_lines():
            if raw.startswith("data: "):
                lines.append(raw[len("data: "):])
            if raw == "data: [done]":
                break

    assert "publish complete" in lines
    assert lines[-1] == "[done]"
    # The completed stream drops the job — a second connect is a 404.
    resp = client.get(f"/api/stream/{job_id}")
    assert resp.status_code == 404


def test_publish_worker_failure_streams_error_and_cleans_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing job surfaces an [error]: line and removes its scratch dir."""
    _clear_jobs()
    monkeypatch.setattr(app_module, "_get_token", lambda: "hf_test")
    monkeypatch.setattr(app_module, "default_scratch_root", lambda: tmp_path)

    def fake_extract_mtp(*, source_repo, sidecar_repo, scratch_root, token, log):
        scratch_root.mkdir(parents=True, exist_ok=True)
        (scratch_root / "leftover.bin").write_bytes(b"x")
        raise app_module.MtpExtractionError("no MTP head tensors found")

    monkeypatch.setattr(app_module, "extract_mtp", fake_extract_mtp)

    resp = client.post(
        "/api/publish",
        json={"base_model": "owner/model", "sidecar_repo": "FoxlightAI/model-mtp"},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    lines: list[str] = []
    with client.stream("GET", f"/api/stream/{job_id}") as stream_resp:
        for raw in stream_resp.iter_lines():
            if raw.startswith("data: "):
                lines.append(raw[len("data: "):])
            if raw == "data: [done]":
                break

    assert any(line.startswith("[error]: no MTP head tensors") for line in lines)
    # The per-job scratch dir was removed despite the failure.
    assert not (tmp_path / "ui-jobs" / job_id).exists()


def test_publish_rejects_when_at_concurrency_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More than _MAX_CONCURRENT_JOBS running jobs → 429 backpressure."""
    import threading as _threading

    _clear_jobs()
    monkeypatch.setattr(app_module, "_get_token", lambda: "hf_test")
    monkeypatch.setattr(app_module, "default_scratch_root", lambda: tmp_path)

    release = _threading.Event()

    def blocking_extract_mtp(*, source_repo, sidecar_repo, scratch_root, token, log):
        release.wait(timeout=10)

    monkeypatch.setattr(app_module, "extract_mtp", blocking_extract_mtp)

    body = {"base_model": "owner/model", "sidecar_repo": "FoxlightAI/model-mtp"}
    try:
        for _ in range(app_module._MAX_CONCURRENT_JOBS):
            assert client.post("/api/publish", json=body).status_code == 200

        resp = client.post("/api/publish", json=body)
        assert resp.status_code == 429
        assert "limit" in resp.json()["error"]
    finally:
        release.set()
        _clear_jobs()


def test_evict_stale_jobs_reaps_old_finished_jobs() -> None:
    """Finished jobs past the retention window are dropped from both dicts."""
    import queue as _queue
    import time as _time

    _clear_jobs()
    with app_module._jobs_lock:
        app_module._jobs["stale"] = _queue.Queue()
        app_module._jobs_finished_at["stale"] = (
            _time.monotonic() - app_module._JOB_RETENTION_SECONDS - 1
        )
        app_module._jobs["fresh"] = _queue.Queue()
        app_module._jobs_finished_at["fresh"] = _time.monotonic()

    app_module._evict_stale_jobs()

    with app_module._jobs_lock:
        assert "stale" not in app_module._jobs
        assert "stale" not in app_module._jobs_finished_at
        assert "fresh" in app_module._jobs
    _clear_jobs()
