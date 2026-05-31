import type { DetectResponse } from '@/types/api';

export interface DetectionResultProps {
  detection: DetectResponse;
  /** Whether mlx is available on the server — gates the Publish button. */
  mlxAvailable?: boolean;
  className?: string;
}
