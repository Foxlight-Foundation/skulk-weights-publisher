import { create } from 'zustand';
import { findCatalog } from '@/api/client';
import type { CatalogEntry } from '@/types/api';

/** Lifecycle of a reverse catalog lookup. */
export type CatalogFindPhase = 'idle' | 'finding' | 'found' | 'notFound' | 'error';

interface CatalogState {
  phase: CatalogFindPhase;
  query: string;
  entry: CatalogEntry | null;
  /** The normalized source model echoed back by the server (owner/repo). */
  sourceModel: string | null;
  errorMessage: string | null;

  setQuery: (query: string) => void;
  find: () => Promise<void>;
  reset: () => void;
}

/**
 * State for the read-only "Find in Catalog" view: given a HuggingFace source
 * model, resolve and display its catalog entry. A 404 (no match) is a normal
 * outcome, modelled as the `notFound` phase rather than `error`; `error` is
 * reserved for unexpected failures (parse errors, network).
 */
export const useCatalogStore = create<CatalogState>()((set, get) => ({
  phase: 'idle',
  query: '',
  entry: null,
  sourceModel: null,
  errorMessage: null,

  setQuery: (query) => set({ query }),

  find: async () => {
    const query = get().query.trim();
    if (!query) {
      return;
    }
    set({ phase: 'finding', entry: null, sourceModel: null, errorMessage: null });
    try {
      const result = await findCatalog(query);
      set({ phase: 'found', entry: result.entry, sourceModel: result.source_model });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Lookup failed';
      // A "no catalog entry" message is the server's 404 — a normal miss, not
      // an error state, so the UI can show a calm "not found" rather than red.
      if (message.includes('no catalog entry found')) {
        set({ phase: 'notFound', errorMessage: message });
      } else {
        set({ phase: 'error', errorMessage: message });
      }
    }
  },

  reset: () =>
    set({
      phase: 'idle',
      query: '',
      entry: null,
      sourceModel: null,
      errorMessage: null,
    }),
}));
