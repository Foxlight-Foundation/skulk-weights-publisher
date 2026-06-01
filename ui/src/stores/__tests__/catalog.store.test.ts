import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useCatalogStore } from '../catalog.store';
import * as client from '@/api/client';
import type { CatalogEntry } from '@/types/api';

vi.mock('@/api/client');

const mockFindCatalog = vi.mocked(client.findCatalog);

const ENTRY: CatalogEntry = {
  key: 'foxlight/gemma-3-4b-full-q4-k',
  source_model: 'google/gemma-3-4b-it',
  quant: 'q4k',
  tier: 'smoke',
  slices: ['full'],
  output_name: 'gemma-3-4b-full-q4-k.vindex',
  hf_repo: 'FoxlightAI/gemma-3-4b-full-q4-k-vindex',
  hf_collection: null,
  mtp_source_repo: null,
  mtp_sidecar_repo: null,
  mtp_quant: null,
  assistant_model_repo: null,
};

beforeEach(() => {
  useCatalogStore.setState({
    phase: 'idle',
    query: '',
    entries: [],
    sourceModel: null,
    errorMessage: null,
  });
  vi.clearAllMocks();
});

describe('catalog.store', () => {
  it('setQuery updates the query', () => {
    useCatalogStore.getState().setQuery('google/gemma-3-4b-it');
    expect(useCatalogStore.getState().query).toBe('google/gemma-3-4b-it');
  });

  it('find resolves to the found phase with all entries on a hit', async () => {
    mockFindCatalog.mockResolvedValue({ source_model: 'google/gemma-3-4b-it', entries: [ENTRY] });
    useCatalogStore.setState({ query: 'google/gemma-3-4b-it' });
    await useCatalogStore.getState().find();
    const state = useCatalogStore.getState();
    expect(state.phase).toBe('found');
    expect(state.entries).toEqual([ENTRY]);
    expect(state.sourceModel).toBe('google/gemma-3-4b-it');
  });

  it('find trims the query before calling the API', async () => {
    mockFindCatalog.mockResolvedValue({ source_model: 'google/gemma-3-4b-it', entries: [ENTRY] });
    useCatalogStore.setState({ query: '  google/gemma-3-4b-it  ' });
    await useCatalogStore.getState().find();
    expect(mockFindCatalog).toHaveBeenCalledWith('google/gemma-3-4b-it');
  });

  it('find is a no-op when the query is blank', async () => {
    useCatalogStore.setState({ query: '   ' });
    await useCatalogStore.getState().find();
    expect(mockFindCatalog).not.toHaveBeenCalled();
    expect(useCatalogStore.getState().phase).toBe('idle');
  });

  it('find maps a 404 message to the notFound phase, not error', async () => {
    mockFindCatalog.mockRejectedValue(new Error("no catalog entry found for source_model 'x/y'"));
    useCatalogStore.setState({ query: 'x/y' });
    await useCatalogStore.getState().find();
    const state = useCatalogStore.getState();
    expect(state.phase).toBe('notFound');
    expect(state.errorMessage).toContain('no catalog entry found');
  });

  it('find maps an unexpected failure to the error phase', async () => {
    mockFindCatalog.mockRejectedValue(new Error('boom'));
    useCatalogStore.setState({ query: 'x/y' });
    await useCatalogStore.getState().find();
    const state = useCatalogStore.getState();
    expect(state.phase).toBe('error');
    expect(state.errorMessage).toBe('boom');
  });

  it('reset returns to idle and clears fields', () => {
    useCatalogStore.setState({
      phase: 'found',
      query: 'x/y',
      entries: [ENTRY],
      sourceModel: 'x/y',
      errorMessage: null,
    });
    useCatalogStore.getState().reset();
    const state = useCatalogStore.getState();
    expect(state.phase).toBe('idle');
    expect(state.query).toBe('');
    expect(state.entries).toEqual([]);
  });
});
