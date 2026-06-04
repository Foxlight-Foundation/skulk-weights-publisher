import type { DetectResponse } from '@/types/api';

export interface DetectionResultProps {
  detection: DetectResponse;
  className?: string;
}
