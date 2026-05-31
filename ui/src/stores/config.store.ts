import { create } from 'zustand';
import { fetchStatus, saveConfig } from '@/api/client';

interface ConfigState {
  hfTokenSet: boolean;
  mlxAvailable: boolean;
  fetchStatus: () => Promise<void>;
  saveToken: (token: string) => Promise<void>;
}

export const useConfigStore = create<ConfigState>()((set) => ({
  hfTokenSet: false,
  mlxAvailable: true,

  fetchStatus: async () => {
    const data = await fetchStatus();
    set({ hfTokenSet: data.hf_token_set, mlxAvailable: data.mlx_available });
  },

  saveToken: async (token: string) => {
    await saveConfig(token);
    set({ hfTokenSet: true });
  },
}));
