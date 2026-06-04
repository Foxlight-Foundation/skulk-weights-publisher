import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useConfigStore } from '../config.store';
import * as client from '@/api/client';

vi.mock('@/api/client');

const mockFetchStatus = vi.mocked(client.fetchStatus);
const mockSaveConfig = vi.mocked(client.saveConfig);

beforeEach(() => {
  // Reset the full slice to a known baseline so state doesn't leak between tests.
  useConfigStore.setState({ hfTokenSet: false });
  vi.clearAllMocks();
});

describe('config.store', () => {
  it('fetchStatus sets hfTokenSet to true when token exists', async () => {
    mockFetchStatus.mockResolvedValue({ hf_token_set: true });
    await useConfigStore.getState().fetchStatus();
    expect(useConfigStore.getState().hfTokenSet).toBe(true);
  });

  it('fetchStatus sets hfTokenSet to false when token absent', async () => {
    mockFetchStatus.mockResolvedValue({ hf_token_set: false });
    await useConfigStore.getState().fetchStatus();
    expect(useConfigStore.getState().hfTokenSet).toBe(false);
  });

  it('saveToken calls saveConfig and sets hfTokenSet to true', async () => {
    mockSaveConfig.mockResolvedValue({ ok: true });
    await useConfigStore.getState().saveToken('hf_abc');
    expect(mockSaveConfig).toHaveBeenCalledWith('hf_abc');
    expect(useConfigStore.getState().hfTokenSet).toBe(true);
  });
});
