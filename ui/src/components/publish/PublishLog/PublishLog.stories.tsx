import type { Meta, StoryObj } from '@storybook/react-vite';
import { PublishLog } from './PublishLog';

const meta: Meta<typeof PublishLog> = {
  title: 'Publish/PublishLog',
  component: PublishLog,
  parameters: { layout: 'padded' },
};

export default meta;
type Story = StoryObj<typeof PublishLog>;

export const Idle: Story = {
  args: { phase: 'publishing', lines: [], errorMessage: null },
};

export const FindingShards: Story = {
  args: {
    phase: 'publishing',
    lines: ["mtp: found MTP tensors in 2 shard(s) (prefix: 'mtp.')"],
    errorMessage: null,
  },
};

export const Downloading: Story = {
  args: {
    phase: 'publishing',
    lines: [
      "mtp: found MTP tensors in 2 shard(s) (prefix: 'mtp.')",
      'mtp: downloading shard 1/2: model-00025-of-00026.safetensors',
    ],
    errorMessage: null,
  },
};

export const SecondShard: Story = {
  args: {
    phase: 'publishing',
    lines: [
      "mtp: found MTP tensors in 2 shard(s) (prefix: 'mtp.')",
      'mtp: downloading shard 1/2: model-00025-of-00026.safetensors',
      'mtp: shard 1/2 ready',
      'mtp: downloading shard 2/2: model-00026-of-00026.safetensors',
    ],
    errorMessage: null,
  },
};

export const Uploading: Story = {
  args: {
    phase: 'publishing',
    lines: [
      "mtp: found MTP tensors in 2 shard(s) (prefix: 'mtp.')",
      'mtp: downloading shard 1/2: model-00025-of-00026.safetensors',
      'mtp: shard 1/2 ready',
      'mtp: downloading shard 2/2: model-00026-of-00026.safetensors',
      'mtp: shard 2/2 ready',
      'mtp: streaming tensors to disk (bf16, unquantized)...',
      'mtp: saved 19 tensor(s) at bf16 (unquantized) to /tmp/skulk-scratch/mtp.safetensors',
      'mtp: uploading to hf://FoxlightAI/qwen3-6-35b-a3b-mtp/mtp.safetensors',
      'mtp: uploading 42% (3.2 GB / 7.6 GB)',
    ],
    errorMessage: null,
  },
};

export const Done: Story = {
  args: {
    phase: 'done',
    lines: [
      "mtp: found MTP tensors in 2 shard(s) (prefix: 'mtp.')",
      'mtp: downloading shard 1/2: model-00025-of-00026.safetensors',
      'mtp: shard 1/2 ready',
      'mtp: downloading shard 2/2: model-00026-of-00026.safetensors',
      'mtp: shard 2/2 ready',
      'mtp: streaming tensors to disk (bf16, unquantized)...',
      'mtp: saved 19 tensor(s) at bf16 (unquantized) to /tmp/skulk-scratch/mtp.safetensors',
      'mtp: uploading to hf://FoxlightAI/qwen3-6-35b-a3b-mtp/mtp.safetensors',
      'mtp: published to hf://FoxlightAI/qwen3-6-35b-a3b-mtp/mtp.safetensors',
      'publish complete',
    ],
    errorMessage: null,
  },
};

export const Failed: Story = {
  args: {
    phase: 'error',
    lines: [
      "mtp: found MTP tensors in 1 shard(s) (prefix: 'mtp.')",
      'mtp: downloading shard 1/1: model.safetensors',
    ],
    errorMessage: 'shards were downloaded but no mtp.* tensors could be read',
  },
};
