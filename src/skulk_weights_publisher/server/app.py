"""FastAPI application for the skulk-ui local GUI."""

from __future__ import annotations

import asyncio
import os
import queue
import shutil
import threading
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from skulk_weights_publisher.catalog_adder import (
    CatalogAddError,
    base_model_slug,
    build_entry_block,
    derive_artifact_slug,
    derive_key_slug,
    detect_base_model,
    detect_mtp_keys,
    detect_quant,
    detect_tier,
    fetch_hf_model_info,
    find_assistant_model,
    find_builtin_catalog_path,
    parse_hf_model_id,
    resolve_base_model,
)
from skulk_weights_publisher.catalogue import (
    find_catalogue_entries_by_source,
    load_catalogue_view,
)
from skulk_weights_publisher.manifest import ALLOWED_QUANTS, ManifestError
from skulk_weights_publisher.mtp_extractor import MtpExtractionError, extract_mtp
from skulk_weights_publisher.publisher import default_scratch_root

_FOXLIGHT_HF_OWNER = "FoxlightAI"
_CONFIG_DIR = Path.home() / ".config" / "skulk-weights"
_ENV_FILE = _CONFIG_DIR / ".env"

# Active publish jobs: job_id → line queue (None = sentinel/done)
_jobs: dict[str, queue.Queue[str | None]] = {}
# Completion timestamps (monotonic) for finished jobs, used to evict ones whose
# client never reconnected to /api/stream so _jobs can't grow unbounded.
_jobs_finished_at: dict[str, float] = {}
# Guards _jobs and _jobs_finished_at: both are mutated from request handlers
# (event loop) and publish worker threads. The queues themselves are
# thread-safe; the dicts around them are not.
_jobs_lock = threading.Lock()
# Serializes the catalog read-check-append in /api/register. Registration runs
# in the default thread pool, so two concurrent requests could otherwise both
# pass the collision checks and double-append the same entry. (In-process only;
# a register racing a CLI `catalog add` in another process is a known follow-up.)
_catalog_write_lock = threading.Lock()
# Keep a finished job around this long so a disconnected client can reconnect.
_JOB_RETENTION_SECONDS = 600.0
# Local single-operator GUI: allow a little parallelism but refuse runaway
# fan-out — each job can download tens of GB.
_MAX_CONCURRENT_JOBS = 2
_EVICT_INTERVAL_SECONDS = 60.0


def _evict_stale_jobs() -> None:
    """Drop finished jobs whose client never reconnected within the grace window."""
    now = time.monotonic()
    with _jobs_lock:
        stale = [
            jid
            for jid, done_at in _jobs_finished_at.items()
            if now - done_at > _JOB_RETENTION_SECONDS
        ]
        for jid in stale:
            _jobs.pop(jid, None)
            _jobs_finished_at.pop(jid, None)


async def _evict_periodically() -> None:
    """Reap stale jobs on a timer.

    Eviction used to piggyback on /api/publish and /api/stream traffic, so a
    fire-and-forget publish whose client never returned was retained until
    unrelated traffic happened to arrive. A background timer reaps it within
    one interval regardless.
    """
    while True:
        await asyncio.sleep(_EVICT_INTERVAL_SECONDS)
        _evict_stale_jobs()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    evictor = asyncio.create_task(_evict_periodically())
    try:
        yield
    finally:
        evictor.cancel()


app = FastAPI(
    title="Skulk Weights Publisher",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConfigBody(BaseModel):
    """Request body for POST /api/config."""

    hf_token: str


class DetectBody(BaseModel):
    """Request body for POST /api/detect."""

    url: str


class PublishBody(BaseModel):
    """Request body for POST /api/publish (MTP sidecar extraction)."""

    base_model: str
    sidecar_repo: str


class RegisterBody(BaseModel):
    """Request body for POST /api/register (catalog entry, no upload)."""

    url: str


class CatalogFindBody(BaseModel):
    """Request body for POST /api/catalog/find (reverse-lookup by source model)."""

    url: str


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def status() -> dict[str, Any]:
    """Return whether an HF token is configured."""
    # MTP extraction is pure numpy (streaming writer) — there is no platform
    # gate to report. The old `mlx_available` flag is gone.
    return {"hf_token_set": bool(_get_token())}


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    """Return a masked view of the stored HF token."""
    token = _get_token()
    if not token:
        return {"hf_token_masked": None}
    masked = token[:7] + "..." + token[-4:] if len(token) > 11 else "***"
    return {"hf_token_masked": masked}


@app.post("/api/config")
async def save_config(body: ConfigBody) -> Any:
    """Persist the HF token to ~/.config/skulk-weights/.env."""
    token = body.hf_token.strip()
    if not token:
        return JSONResponse({"error": "hf_token must not be empty"}, status_code=400)
    _save_token(token)
    return {"ok": True}


@app.post("/api/catalog/find")
async def catalog_find(body: CatalogFindBody) -> Any:
    """Reverse-lookup catalog entries by their HF source model.

    The GUI counterpart of ``skulk-weights catalog find``: given the upstream
    source model (URL or ``owner/repo``), return all matching catalog entries
    (the mapping is one-to-many — e.g. full + expert-server slices), or 404 when
    none match. Read-only — never mutates the catalog.
    """
    query = body.url.strip()
    if not query:
        return JSONResponse({"error": "url is required"}, status_code=400)
    try:
        source_model = parse_hf_model_id(query)
    except CatalogAddError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    # Load the catalog OUTSIDE the lookup try: a malformed/unreadable catalog is a
    # 500 configuration error, not a 404 "no entry" miss. Only the lookup's own
    # ManifestError (genuinely no match) maps to 404. Catch broadly here —
    # load_catalogue_view can raise yaml.YAMLError, OSError, etc. (not just
    # ManifestError) — so every load failure still returns the JSON error
    # envelope the UI expects instead of a bare 500 that breaks res.json().
    try:
        view = load_catalogue_view()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"error": f"catalog failed to load: {exc}"}, status_code=500
        )
    try:
        entries = find_catalogue_entries_by_source(source_model, view)
    except ManifestError as exc:
        return JSONResponse(
            {"error": str(exc), "source_model": source_model}, status_code=404
        )
    return {
        "source_model": source_model,
        "entries": [entry.to_dict() for entry in entries],
    }


def _detect_impl(url: str, token: str | None) -> dict[str, Any]:
    """Blocking detection work for /api/detect — runs in a worker thread."""
    model_id = parse_hf_model_id(url)
    info = fetch_hf_model_info(model_id, token=token)
    immediate_base = detect_base_model(info)
    base_model = resolve_base_model(info, token=token)
    quant = detect_quant(info)
    tier = detect_tier(info)
    mtp_keys: list[str] = []
    sidecar_repo: str | None = None
    if base_model:
        mtp_keys = detect_mtp_keys(base_model, token=token)
        # Only advertise a sidecar repo when there are actually MTP tensors
        # to publish — otherwise the response is self-contradictory
        # (can_publish False but sidecar_repo populated).
        if mtp_keys:
            # One bf16 sidecar per base model — quant-independent.
            sidecar_repo = f"{_FOXLIGHT_HF_OWNER}/{base_model_slug(base_model)}-mtp"
    # If no MTP tensors, check for a Gemma 4-style companion assistant.
    # The assistant is named after the instruct model the user pasted, so
    # check model_id first, then its base(s).
    assistant_model_repo: str | None = None
    if not mtp_keys:
        assistant_model_repo = find_assistant_model(
            [model_id, immediate_base, base_model], token=token
        )
    return {
        "model_id": model_id,
        "base_model": base_model,
        "quant": quant,
        "tier": tier,
        "mtp_key_count": len(mtp_keys),
        "mtp_keys": mtp_keys,
        "sidecar_repo": sidecar_repo,
        "can_publish": len(mtp_keys) > 0 and base_model is not None,
        "assistant_model_repo": assistant_model_repo,
        "can_publish_assistant": assistant_model_repo is not None,
    }


@app.post("/api/detect")
async def detect(body: DetectBody) -> Any:
    """Parse a HF model URL and return full detection metadata."""
    url = body.url.strip()
    if not url:
        return JSONResponse({"error": "url is required"}, status_code=400)

    token = _get_token()
    loop = asyncio.get_running_loop()
    try:
        # Detection makes several sequential HF API calls (sync urllib) — run
        # them off the event loop so a slow detect can't stall a concurrent
        # publish's SSE stream.
        return await loop.run_in_executor(None, _detect_impl, url, token)
    except CatalogAddError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/publish")
async def publish(body: PublishBody) -> Any:
    """Start an async MTP extraction job and return its job ID."""
    _evict_stale_jobs()

    # Fail fast like /api/detect and /api/register do: without a token the job
    # would only die deep inside the HF upload with an opaque [error] line.
    token = _get_token()
    if not token:
        return JSONResponse(
            {"error": "HF token not configured — open Settings to add one"},
            status_code=400,
        )

    job_id = str(uuid.uuid4())
    q: queue.Queue[str | None] = queue.Queue()
    with _jobs_lock:
        running = len(_jobs) - len(_jobs_finished_at)
        if running >= _MAX_CONCURRENT_JOBS:
            return JSONResponse(
                {
                    "error": (
                        f"{running} publish job(s) already running "
                        f"(limit {_MAX_CONCURRENT_JOBS}) — retry when one finishes"
                    )
                },
                status_code=429,
            )
        _jobs[job_id] = q

    scratch = default_scratch_root() / "ui-jobs" / job_id

    def _run() -> None:
        # Route progress through a per-job callback rather than monkeypatching
        # the process-wide sys.stderr (which would interleave across concurrent
        # jobs and the server's own stderr).
        try:
            extract_mtp(
                source_repo=body.base_model,
                sidecar_repo=body.sidecar_repo,
                scratch_root=scratch,
                token=token,
                log=q.put,
            )
            q.put("publish complete")
        except MtpExtractionError as exc:
            q.put(f"[error]: {exc}")
        except Exception as exc:  # noqa: BLE001
            q.put(f"[error]: unexpected error: {exc}")
        finally:
            # extract_mtp cleans its own staging; remove the per-job dir
            # skeleton too so failed jobs leave nothing behind.
            shutil.rmtree(scratch, ignore_errors=True)
            with _jobs_lock:
                _jobs_finished_at[job_id] = time.monotonic()
            q.put(None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


class _ApiError(Exception):
    """An error with an HTTP status, raised from sync impl helpers."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _register_impl(url: str, token: str | None) -> dict[str, Any]:
    """Blocking registration work for /api/register — runs in a worker thread."""
    model_id = parse_hf_model_id(url)
    info = fetch_hf_model_info(model_id, token=token)
    immediate_base = detect_base_model(info)
    base_model = resolve_base_model(info, token=token)
    quant = detect_quant(info)
    if quant not in ALLOWED_QUANTS:
        raise _ApiError(
            400,
            f"detected quant '{quant}' is not supported "
            f"(allowed: {', '.join(sorted(ALLOWED_QUANTS))})",
        )
    tier = detect_tier(info)

    mtp_keys: list[str] = []
    if base_model:
        mtp_keys = detect_mtp_keys(base_model, token=token)
    assistant_model_repo: str | None = None
    if not mtp_keys:
        assistant_model_repo = find_assistant_model(
            [model_id, immediate_base, base_model], token=token
        )

    key_slug = derive_key_slug(model_id, quant)
    artifact_slug = derive_artifact_slug(model_id, quant)
    effective_key = f"foxlight/{key_slug}"
    hf_repo_new = f"FoxlightAI/{artifact_slug}-vindex"
    output_name_new = f"{artifact_slug}.vindex"

    # Check-then-append must be atomic: hold the lock across both so two
    # concurrent registrations of the same model can't both pass the collision
    # checks and double-append.
    with _catalog_write_lock:
        entries = load_catalogue_view().entries
        if effective_key in {e.key for e in entries}:
            raise _ApiError(
                409, f"'{effective_key}' already exists in the catalog"
            )
        if hf_repo_new in {e.hf_repo for e in entries}:
            raise _ApiError(
                409, f"hf_repo '{hf_repo_new}' already exists in the catalog"
            )
        if output_name_new in {e.output_name for e in entries}:
            raise _ApiError(
                409,
                f"output_name '{output_name_new}' already exists in the catalog",
            )

        entry_block = build_entry_block(
            key_slug=key_slug,
            artifact_slug=artifact_slug,
            source_model=model_id,
            quant=quant,
            tier=tier,
            base_model=base_model,
            mtp_keys=mtp_keys,
            assistant_model_repo=assistant_model_repo,
        )
        catalog_path = find_builtin_catalog_path()
        with open(catalog_path, "a", encoding="utf-8") as fh:
            fh.write(entry_block)

    return {
        "ok": True,
        "key": effective_key,
        "assistant_model_repo": assistant_model_repo,
        "catalog_path": str(catalog_path),
        "entry_block": entry_block,
    }


@app.post("/api/register")
async def register(body: RegisterBody) -> Any:
    """Append a catalog entry for a model (no upload).

    Used by the GUI's "Register in Catalog" action — chiefly for Gemma 4
    assistant-type models, where there is nothing to extract and the entry
    simply records the source-model → assistant pairing. Mirrors the
    `skulk-weights catalog add` flow: detect, build the entry, collision-check,
    append to the built-in foxlight.yaml.
    """
    url = body.url.strip()
    if not url:
        return JSONResponse({"error": "url is required"}, status_code=400)

    token = _get_token()
    loop = asyncio.get_running_loop()
    try:
        # Same as /api/detect: several sequential HF API calls — keep them off
        # the event loop.
        return await loop.run_in_executor(None, _register_impl, url, token)
    except _ApiError as exc:
        return JSONResponse({"error": str(exc)}, status_code=exc.status_code)
    except CatalogAddError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/stream/{job_id}")
async def stream(job_id: str) -> Any:
    """SSE stream of live log output for a running publish job."""
    _evict_stale_jobs()
    with _jobs_lock:
        q = _jobs.get(job_id)
    if q is None:
        return JSONResponse({"error": "job not found"}, status_code=404)

    async def _event_gen() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        completed = False
        try:
            while True:
                line = await loop.run_in_executor(None, q.get)
                if line is None:
                    completed = True
                    yield "data: [done]\n\n"
                    break
                yield f"data: {line}\n\n"
        finally:
            # Only drop the job once it has genuinely finished (sentinel seen).
            # On a transient SSE disconnect the background thread is still using
            # the queue, so we keep the job registered to let the UI reconnect
            # rather than orphaning a running publish (and risking a duplicate).
            # Jobs whose client never reconnects are reaped by _evict_stale_jobs.
            if completed:
                with _jobs_lock:
                    _jobs.pop(job_id, None)
                    _jobs_finished_at.pop(job_id, None)

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Static files (SPA) — mounted last so /api/* routes take priority
# ---------------------------------------------------------------------------

def _resolve_dist() -> Path:
    env_override = os.environ.get("SKULK_UI_DIST", "").strip()
    if env_override:
        return Path(env_override)
    # Default: repo-root/ui/dist (works for editable / source installs)
    return Path(__file__).parent.parent.parent.parent / "ui" / "dist"


_dist = _resolve_dist()
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _get_token() -> str | None:
    """Return the HF token from environment or ~/.config/skulk-weights/.env."""
    env_val = os.environ.get("HF_TOKEN", "").strip()
    if env_val:
        return env_val
    if _ENV_FILE.is_file():
        for raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw.startswith("HF_TOKEN="):
                val = raw.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    return None


def _save_token(token: str) -> None:
    """Write or update HF_TOKEN in ~/.config/skulk-weights/.env.

    The file holds a write-capable HF token, so it is created with owner-only
    (0600) permissions and the config dir with 0700, to avoid exposing the
    token on shared machines with a permissive default umask.
    """
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(_CONFIG_DIR, 0o700)
    pairs: dict[str, str] = {}
    if _ENV_FILE.is_file():
        for raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#") and "=" in raw:
                k, v = raw.split("=", 1)
                pairs[k.strip()] = v.strip()
    pairs["HF_TOKEN"] = token
    content = "\n".join(f"{k}={v}" for k, v in pairs.items()) + "\n"
    # Open with 0600 from the start so the token is never briefly world-readable.
    fd = os.open(_ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.chmod(_ENV_FILE, 0o600)
