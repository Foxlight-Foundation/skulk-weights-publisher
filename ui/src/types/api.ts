/** Response from GET /api/status */
export interface StatusResponse {
  hf_token_set: boolean;
  mlx_available: boolean;
}

/** Response from GET /api/config */
export interface ConfigResponse {
  hf_token_masked: string | null;
}

/** Response from POST /api/detect (success) */
export interface DetectResponse {
  model_id: string;
  base_model: string | null;
  quant: string;
  tier: string;
  mtp_key_count: number;
  mtp_keys: string[];
  sidecar_repo: string | null;
  can_publish: boolean;
  assistant_model_repo: string | null;
  can_publish_assistant: boolean;
}

/** Response from POST /api/publish */
export interface PublishResponse {
  job_id: string;
}

/** Response from POST /api/register */
export interface RegisterResponse {
  ok: boolean;
  key: string;
  assistant_model_repo: string | null;
  catalog_path: string;
  entry_block: string;
}

/** A resolved catalog entry, as serialized by ManifestEntry.to_dict(). */
export interface CatalogEntry {
  key: string;
  source_model: string;
  quant: string;
  tier: string;
  slices: string[];
  output_name: string;
  hf_repo: string;
  hf_collection: string | null;
  mtp_source_repo: string | null;
  mtp_sidecar_repo: string | null;
  mtp_quant: string | null;
  assistant_model_repo: string | null;
}

/** Response from POST /api/catalog/find (success) */
export interface CatalogFindResponse {
  source_model: string;
  entry: CatalogEntry;
}

/** Error envelope returned by API routes on failure */
export interface ApiError {
  error: string;
}
