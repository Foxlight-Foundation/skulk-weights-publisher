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

/** Error envelope returned by API routes on failure */
export interface ApiError {
  error: string;
}
