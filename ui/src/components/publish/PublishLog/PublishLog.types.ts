import type { PublishPhase } from '@/stores/publish.store';

export type Stage =
  | 'finding'
  | 'downloading'
  | 'extracting'
  | 'quantizing'
  | 'saving'
  | 'uploading'
  | 'confirming'
  | 'done';

export type StageStatus = 'pending' | 'active' | 'done';

export interface PublishLogProps {
  phase: PublishPhase;
  lines: string[];
  errorMessage: string | null;
  className?: string;
}
