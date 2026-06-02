import type { Meta, StoryObj } from '@storybook/react-vite';
import { useEffect } from 'react';
import { CatalogResult } from './CatalogResult';
import { useCatalogStore } from '@/stores/catalog.store';
import type { CatalogEntry } from '@/types/api';

const SAMPLE: CatalogEntry = {
  key: 'foxlight/gemma-3-4b-full-q4-k',
  source_model: 'google/gemma-3-4b-it',
  quant: 'q4k',
  tier: 'smoke',
  slices: ['full'],
  output_name: 'gemma-3-4b-full-q4-k.vindex',
  hf_repo: 'FoxlightAI/gemma-3-4b-full-q4-k-vindex',
  hf_collection: 'FoxlightAI/vindexes-6a124406dd5fb439c431b051',
  mtp_source_repo: null,
  mtp_sidecar_repo: null,
  assistant_model_repo: null,
  vision_source_repo: null,
  vision_sidecar_repo: null,
};

const meta: Meta<typeof CatalogResult> = {
  title: 'Catalog/CatalogResult',
  component: CatalogResult,
  parameters: { layout: 'padded' },
};

export default meta;
type Story = StoryObj<typeof CatalogResult>;

export const Found: Story = {
  render: () => {
    useEffect(() => {
      useCatalogStore.setState({
        phase: 'found',
        entries: [SAMPLE],
        sourceModel: SAMPLE.source_model,
      });
    }, []);
    return <CatalogResult />;
  },
};

export const NotFound: Story = {
  render: () => {
    useEffect(() => {
      useCatalogStore.setState({
        phase: 'notFound',
        entries: [],
        sourceModel: 'mlx-community/unknown-model',
        errorMessage: "no catalog entry found for source_model 'mlx-community/unknown-model'",
      });
    }, []);
    return <CatalogResult />;
  },
};
