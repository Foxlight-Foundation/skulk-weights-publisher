import { create } from 'zustand';
import { fetchStatus, saveConfig } from '@/api/client';

interface ConfigState {
  hfTokenSet: boolean;
  fetchStatus: () => Promise<void>;
  saveToken: (token: string) => Promise<void>;
}

export const useConfigStore = create<ConfigState>()((set) => ({
  hfTokenSet: false,

  fetchStatus: async () => {
    const data = await fetchStatus();
    set({ hfTokenSet: data.hf_token_set });
  },

  saveToken: async (token: string) => {
    await saveConfig(token);
    set({ hfTokenSet: true });
  },
}));
