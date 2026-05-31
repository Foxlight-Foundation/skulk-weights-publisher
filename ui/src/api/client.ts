/**
 * Typed fetch wrappers for all /api/* endpoints.
 *
 * Each function throws an Error with the server's error message on non-2xx
 * responses, so callers can catch a single error type.
 */

import type { ConfigResponse, DetectResponse, PublishResponse, StatusResponse } from '@/types/api';

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

/** Start an async publish job (MTP extraction or assistant registration) and return the job ID. */
export function startPublish(
  baseModel: string,
  sidecarRepo: string,
  quant: string,
  publishType: 'mtp' | 'assistant' = 'mtp',
): Promise<PublishResponse> {
  return _post<PublishResponse>('/api/publish', {
    base_model: baseModel,
    sidecar_repo: sidecarRepo,
    quant,
    publish_type: publishType,
  });
}

/**
 * Open an SSE stream for a running publish job.
 * Returns the EventSource — caller is responsible for closing it.
 */
export function openLogStream(jobId: string): EventSource {
  return new EventSource(`/api/stream/${jobId}`);
}
