---
title: HTTP API Reference
---

The `skulk-ui` local GUI is a thin client over a FastAPI server
(`src/skulk_weights_publisher/server/app.py`). The server binds to `localhost`
only and reads the HuggingFace token from the environment or
`~/.config/skulk-weights/.env` — there are no accounts and no auth headers.

All `/api/*` routes return JSON. Error responses use a `{"error": "..."}`
envelope with an HTTP status code (`400` bad input, `404` not found, `409`
conflict, `500` server/config error). The static SPA is mounted last at `/`, so
`/api/*` always takes priority.

There are seven endpoints.

## `GET /api/status`

Returns whether a token is configured.

```json
{ "hf_token_set": true }
```

:::note
MTP extraction is pure numpy and cross-platform, so there is no platform
capability to report here. (Earlier releases returned an `mlx_available`
flag from the pre-streaming extraction path; it was removed along with the
GUI's mlx warning banner.)
:::

## `GET /api/config`

Returns a masked view of the stored token (or `null` when none is set).

```json
{ "hf_token_masked": "hf_AbCd...wxyz" }
```

## `POST /api/config`

Persists the HF token to `~/.config/skulk-weights/.env` (written `0600`).

Request:

```json
{ "hf_token": "hf_..." }
```

Response: `{ "ok": true }`. An empty token returns `400`
`{"error": "hf_token must not be empty"}`.

## `POST /api/detect`

Parses a HuggingFace model URL (or `owner/repo`) and returns full detection
metadata: the resolved BF16 base model, detected quant/tier, MTP key count, the
target sidecar repo (only populated when MTP tensors actually exist), and any
Gemma 4 companion assistant.

Request:

```json
{ "url": "https://huggingface.co/mlx-community/Qwen3.5-9B-OptiQ-4bit" }
```

Response:

```json
{
  "model_id": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
  "base_model": "Qwen/Qwen3.5-9B",
  "quant": "q4k",
  "tier": "smoke",
  "mtp_key_count": 12,
  "mtp_keys": ["mtp.embed_tokens.weight", "..."],
  "sidecar_repo": "FoxlightAI/qwen3-5-9b-mtp",
  "can_publish": true,
  "assistant_model_repo": null,
  "can_publish_assistant": false
}
```

- `can_publish` is `true` only when MTP tensors were found **and** a base model
  resolved — i.e. there is something to extract.
- `sidecar_repo` is `null` unless MTP tensors exist (otherwise the response would
  be self-contradictory).
- For a Gemma 4 model with no MTP heads, `mtp_key_count` is `0`, `can_publish` is
  `false`, and `assistant_model_repo` / `can_publish_assistant` are populated
  instead.

An empty `url` returns `400`; a parse error returns `400`; any other failure
returns `500` with the error envelope.

## `POST /api/publish`

Starts an **asynchronous** MTP extraction job in a background thread and returns
its job ID immediately. Progress is delivered over Server-Sent Events from
`GET /api/stream/{job_id}`.

Request:

```json
{ "base_model": "Qwen/Qwen3.5-9B", "sidecar_repo": "FoxlightAI/qwen3-5-9b-mtp" }
```

Response:

```json
{ "job_id": "0d6f...e3a1" }
```

Error responses:

- `400` — no HF token configured (the job would otherwise fail deep inside the
  upload; configure one via `POST /api/config` first).
- `429` — too many publish jobs already running (the server caps concurrent
  jobs at 2 — each can download tens of GB); retry when one finishes.

### Publish → stream flow

1. `POST /api/publish` returns a `job_id`.
2. The client opens `GET /api/stream/{job_id}` and reads SSE `data:` lines as the
   extraction runs (finding shards → downloading → streaming to disk →
   byte-level upload progress).
3. On success the stream emits a final `publish complete` line, then `[done]`.
4. On failure the stream emits a line prefixed `[error]: ` (for example
   `[error]: no MTP head tensors found in ...`, or
   `[error]: unexpected error: ...` for an unexpected exception), then `[done]`.

The `[error]: ` prefix and the `publish complete` sentinel are the contract the
GUI uses to distinguish success from failure on an otherwise opaque log stream.

## `POST /api/register`

Appends a catalog entry for a model **without uploading anything** — the GUI's
"Register in Catalog" action, used chiefly for Gemma 4 assistant pairings where
there is nothing to extract. Mirrors `skulk-weights catalog add`: detect, build
the entry block, collision-check, append to the built-in `foxlight.yaml`.

Request (the example model is **not** in the shipped catalog — registering an
already-catalogued model such as `google/gemma-4-31B-it` returns the `409`
described below instead):

```json
{ "url": "https://huggingface.co/google/gemma-4-12B-it" }
```

Response:

```json
{
  "ok": true,
  "key": "foxlight/gemma-4-12b-full-q4-k",
  "assistant_model_repo": "google/gemma-4-12B-it-assistant",
  "catalog_path": "/path/to/foxlight.yaml",
  "entry_block": "  - key: gemma-4-12b-full-q4-k\n    ..."
}
```

Error responses:

- `400` — empty/invalid `url`, or a detected quant outside `ALLOWED_QUANTS`
  (only `q4k` is allowed; 8-bit `q8k` models are rejected here).
- `409` — the derived key, `hf_repo`, or `output_name` already exists in the
  catalog.
- `500` — any other failure.

## `POST /api/catalog/find`

Reverse-lookup of catalog entries by their HuggingFace source model — the GUI
counterpart of `skulk-weights catalog find`. Read-only; never mutates the
catalog. The mapping is one-to-many (e.g. `full` + `expert-server` slices).

Request:

```json
{ "url": "google/gemma-3-4b-it" }
```

Response:

```json
{
  "source_model": "google/gemma-3-4b-it",
  "entries": [ { "key": "foxlight/gemma-3-4b-full-q4-k", "...": "..." } ]
}
```

- `400` — empty `url` or an unparseable model id.
- `404` — no catalog entry matches the source model
  (`{"error": "...", "source_model": "..."}`).
- `500` — the catalog itself failed to load (a configuration error, distinct
  from a `404` miss).

## `GET /api/stream/{job_id}`

Server-Sent Events stream of live log output for a publish job started by
`POST /api/publish`. Each progress line is sent as `data: <line>\n\n`; the
stream ends with `data: [done]\n\n`.

- Returns `404` `{"error": "job not found"}` if the job ID is unknown or has
  already been evicted.
- A transient SSE disconnect does **not** drop the job — the background thread is
  still using the queue, so the client can reconnect to the same `job_id` and
  resume reading rather than orphaning (or duplicating) a running publish.
- Finished jobs are retained for **600 seconds** after completion so a
  disconnected client has a grace window to reconnect; jobs whose client never
  reconnects within that window are reaped (`_JOB_RETENTION_SECONDS`).

Response headers set `Cache-Control: no-cache`, `Connection: keep-alive`, and
`X-Accel-Buffering: no` so intermediaries do not buffer the event stream.
