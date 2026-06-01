/**
 * Typed fetch wrappers for all /api/* endpoints.
 *
 * Each function throws an Error with the server's error message on non-2xx
 * responses, so callers can catch a single error type.
 */

import type {
  CatalogFindResponse,
  ConfigResponse,
  DetectResponse,
  PublishResponse,
  RegisterResponse,
  StatusResponse,
} from '@/types/api';

async function _post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = (await res.json()) as T & { error?: string };
  if (!res.ok) {
    throw new Error(data.error ?? `Request failed: ${res.status}`);
  }
  return data;
}

async function _get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  const data = (await res.json()) as T & { error?: string };
  if (!res.ok) {
    throw new Error(data.error ?? `Request failed: ${res.status}`);
  }
  return data;
}

/** Check whether an HF token is configured on the server. */
export function fetchStatus(): Promise<StatusResponse> {
  return _get<StatusResponse>('/api/status');
}

/** Fetch the masked HF token for display in the settings panel. */
export function fetchConfig(): Promise<ConfigResponse> {
  return _get<ConfigResponse>('/api/config');
}

/** Persist a new HF token to ~/.config/skulk-weights/.env. */
export function saveConfig(hfToken: string): Promise<{ ok: boolean }> {
  return _post<{ ok: boolean }>('/api/config', { hf_token: hfToken });
}

/** Detect model metadata from a HuggingFace URL or owner/repo string. */
export function detectModel(url: string): Promise<DetectResponse> {
  return _post<DetectResponse>('/api/detect', { url });
}

/** Start an async MTP extraction job and return the job ID. */
export function startPublish(
  baseModel: string,
  sidecarRepo: string,
  quant: string,
): Promise<PublishResponse> {
  return _post<PublishResponse>('/api/publish', {
    base_model: baseModel,
    sidecar_repo: sidecarRepo,
    quant,
  });
}

/**
 * Register a catalog entry for a model (no upload). Used by "Register in
 * Catalog" for Gemma 4 assistant-type models. Re-detects from the URL
 * server-side and appends the entry.
 */
export function registerCatalog(url: string): Promise<RegisterResponse> {
  return _post<RegisterResponse>('/api/register', { url });
}

/**
 * Reverse-lookup a catalog entry by its HuggingFace source model (URL or
 * owner/repo). Read-only — never mutates the catalog. Throws on 404 (no match)
 * and 400 (unparseable input) with the server's error message.
 */
export function findCatalog(url: string): Promise<CatalogFindResponse> {
  return _post<CatalogFindResponse>('/api/catalog/find', { url });
}

/**
 * Open an SSE stream for a running publish job.
 * Returns the EventSource — caller is responsible for closing it.
 */
export function openLogStream(jobId: string): EventSource {
  return new EventSource(`/api/stream/${jobId}`);
}
