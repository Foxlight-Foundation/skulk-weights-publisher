import { create } from 'zustand';
import { detectModel, openLogStream, registerCatalog, startPublish } from '@/api/client';
import type { DetectResponse } from '@/types/api';

export type PublishPhase = 'idle' | 'detecting' | 'detected' | 'publishing' | 'done' | 'error';

interface PublishState {
  phase: PublishPhase;
  url: string;
  detection: DetectResponse | null;
  logLines: string[];
  errorMessage: string | null;

  setUrl: (url: string) => void;
  detect: () => Promise<void>;
  publish: () => Promise<void>;
  reset: () => void;
}

export const usePublishStore = create<PublishState>()((set, get) => ({
  phase: 'idle',
  url: '',
  detection: null,
  logLines: [],
  errorMessage: null,

  setUrl: (url) => set({ url }),

  detect: async () => {
    const { url } = get();
    set({ phase: 'detecting', detection: null, errorMessage: null });
    try {
      const result = await detectModel(url);
      set({ phase: 'detected', detection: result });
    } catch (err) {
      set({
        phase: 'error',
        errorMessage: err instanceof Error ? err.message : 'Detection failed',
      });
    }
  },

  publish: async () => {
    const { detection } = get();

    // Assistant path: nothing to extract — register the companion model in the
    // catalog (a synchronous append, no SSE job).
    if (detection?.can_publish_assistant && !detection.can_publish) {
      const { url } = get();
      set({ phase: 'publishing', logLines: [], errorMessage: null });
      try {
        const res = await registerCatalog(url);
        set({
          phase: 'done',
          logLines: [
            `registered ${res.key} in catalog`,
            res.assistant_model_repo ? `assistant: ${res.assistant_model_repo}` : '',
            `written to ${res.catalog_path}`,
          ].filter(Boolean),
        });
      } catch (err) {
        set({
          phase: 'error',
          errorMessage: err instanceof Error ? err.message : 'Failed to register in catalog',
        });
      }
      return;
    }

    if (!detection?.base_model || !detection.sidecar_repo) return;

    set({ phase: 'publishing', logLines: [], errorMessage: null });

    let jobId: string;
    try {
      const res = await startPublish(detection.base_model, detection.sidecar_repo);
      jobId = res.job_id;
    } catch (err) {
      set({
        phase: 'error',
        errorMessage: err instanceof Error ? err.message : 'Failed to start publish job',
      });
      return;
    }

    const es = openLogStream(jobId);
    es.onmessage = (e: MessageEvent<string>) => {
      if (e.data === '[done]') {
        es.close();
        set({ phase: 'done' });
        return;
      }
      if (e.data.startsWith('[error]:')) {
        es.close();
        set({
          phase: 'error',
          errorMessage: e.data.replace('[error]:', '').trim(),
        });
        return;
      }
      set((s) => ({ logLines: [...s.logLines, e.data] }));
    };
    es.onerror = () => {
      es.close();
      set({ phase: 'error', errorMessage: 'Lost connection to publish stream' });
    };
  },

  reset: () =>
    set({
      phase: 'idle',
      url: '',
      detection: null,
      logLines: [],
      errorMessage: null,
    }),
}));
