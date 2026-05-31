"""FastAPI application for the skulk-ui local GUI."""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import uuid
from collections.abc import AsyncGenerator
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
    detect_assistant_model,
    detect_base_model,
    detect_mtp_keys,
    detect_quant,
    detect_tier,
    fetch_hf_model_info,
    find_builtin_catalog_path,
    parse_hf_model_id,
    quant_suffix,
    resolve_base_model,
)
from skulk_weights_publisher.catalogue import load_catalogue_view
from skulk_weights_publisher.manifest import ALLOWED_QUANTS
from skulk_weights_publisher.mtp_extractor import MtpExtractionError, extract_mtp
from skulk_weights_publisher.publisher import default_scratch_root

_FOXLIGHT_HF_OWNER = "FoxlightAI"
_CONFIG_DIR = Path.home() / ".config" / "skulk-weights"
_ENV_FILE = _CONFIG_DIR / ".env"

# Active publish jobs: job_id → line queue (None = sentinel/done)
_jobs: dict[str, queue.Queue[str | None]] = {}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Skulk Weights Publisher", docs_url=None, redoc_url=None)


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
    quant: str


class RegisterBody(BaseModel):
    """Request body for POST /api/register (catalog entry, no upload)."""

    url: str


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def status() -> dict[str, Any]:
    """Return whether an HF token is configured and whether mlx is available."""
    return {"hf_token_set": bool(_get_token()), "mlx_available": _check_mlx()}


def _check_mlx() -> bool:
    try:
        import mlx.core  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


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


@app.post("/api/detect")
async def detect(body: DetectBody) -> Any:
    """Parse a HF model URL and return full detection metadata."""
    url = body.url.strip()
    if not url:
        return JSONResponse({"error": "url is required"}, status_code=400)

    token = _get_token()
    try:
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
            sidecar_repo = (
                f"{_FOXLIGHT_HF_OWNER}/{base_model_slug(base_model)}"
                f"-mtp-{quant_suffix(quant)}"
            )
        # If no MTP tensors, check for a Gemma 4-style companion assistant.
        assistant_model_repo: str | None = None
        if not mtp_keys:
            if immediate_base:
                assistant_model_repo = detect_assistant_model(
                    immediate_base, token=token
                )
            if (
                assistant_model_repo is None
                and base_model
                and base_model != immediate_base
            ):
                assistant_model_repo = detect_assistant_model(base_model, token=token)
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
    except CatalogAddError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/publish")
async def publish(body: PublishBody) -> Any:
    """Start an async MTP extraction job and return its job ID."""
    token = _get_token()
    job_id = str(uuid.uuid4())
    q: queue.Queue[str | None] = queue.Queue()
    _jobs[job_id] = q

    scratch = default_scratch_root() / "ui-jobs" / job_id

    def _run() -> None:
        class _QueueWriter:
            def write(self, text: str) -> None:
                stripped = text.rstrip()
                if stripped:
                    q.put(stripped)

            def flush(self) -> None:
                pass

        old_stderr = sys.stderr
        sys.stderr = _QueueWriter()  # type: ignore[assignment]
        try:
            extract_mtp(
                source_repo=body.base_model,
                sidecar_repo=body.sidecar_repo,
                mtp_quant=body.quant,
                scratch_root=scratch,
                token=token,
            )
            q.put("publish complete")
        except MtpExtractionError as exc:
            q.put(f"[error]: {exc}")
        except Exception as exc:  # noqa: BLE001
            q.put(f"[error]: unexpected error: {exc}")
        finally:
            sys.stderr = old_stderr
            q.put(None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


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
    try:
        model_id = parse_hf_model_id(url)
        info = fetch_hf_model_info(model_id, token=token)
        immediate_base = detect_base_model(info)
        base_model = resolve_base_model(info, token=token)
        quant = detect_quant(info)
        if quant not in ALLOWED_QUANTS:
            return JSONResponse(
                {
                    "error": (
                        f"detected quant '{quant}' is not supported "
                        f"(allowed: {', '.join(sorted(ALLOWED_QUANTS))})"
                    )
                },
                status_code=400,
            )
        tier = detect_tier(info)

        mtp_keys: list[str] = []
        if base_model:
            mtp_keys = detect_mtp_keys(base_model, token=token)
        assistant_model_repo: str | None = None
        if not mtp_keys:
            if immediate_base:
                assistant_model_repo = detect_assistant_model(
                    immediate_base, token=token
                )
            if (
                assistant_model_repo is None
                and base_model
                and base_model != immediate_base
            ):
                assistant_model_repo = detect_assistant_model(base_model, token=token)

        key_slug = derive_key_slug(model_id, quant)
        artifact_slug = derive_artifact_slug(model_id, quant)
        effective_key = f"foxlight/{key_slug}"
        hf_repo_new = f"FoxlightAI/{artifact_slug}-vindex"
        output_name_new = f"{artifact_slug}.vindex"

        entries = load_catalogue_view().entries
        if effective_key in {e.key for e in entries}:
            return JSONResponse(
                {"error": f"'{effective_key}' already exists in the catalog"},
                status_code=409,
            )
        if hf_repo_new in {e.hf_repo for e in entries}:
            return JSONResponse(
                {"error": f"hf_repo '{hf_repo_new}' already exists in the catalog"},
                status_code=409,
            )
        if output_name_new in {e.output_name for e in entries}:
            return JSONResponse(
                {
                    "error": (
                        f"output_name '{output_name_new}' already exists "
                        "in the catalog"
                    )
                },
                status_code=409,
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
    except CatalogAddError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/stream/{job_id}")
async def stream(job_id: str) -> Any:
    """SSE stream of live log output for a running publish job."""
    q = _jobs.get(job_id)
    if q is None:
        return JSONResponse({"error": "job not found"}, status_code=404)

    async def _event_gen() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, q.get)
            if line is None:
                yield "data: [done]\n\n"
                break
            yield f"data: {line}\n\n"

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
