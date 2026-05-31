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
    lines: ['mtp: found mtp.* keys in 2 shard(s)'],
    errorMessage: null,
  },
};

export const Downloading: Story = {
  args: {
    phase: 'publishing',
    lines: [
      'mtp: found mtp.* keys in 2 shard(s)',
      'mtp: downloading shard 1/2: model-00025-of-00026.safetensors',
    ],
    errorMessage: null,
  },
};

export const SecondShard: Story = {
  args: {
    phase: 'publishing',
    lines: [
      'mtp: found mtp.* keys in 2 shard(s)',
      'mtp: downloading shard 1/2: model-00025-of-00026.safetensors',
      'mtp: shard 1/2 ready',
      'mtp: downloading shard 2/2: model-00026-of-00026.safetensors',
    ],
    errorMessage: null,
  },
};

export const Done: Story = {
  args: {
    phase: 'done',
    lines: [
      'mtp: found mtp.* keys in 2 shard(s)',
      'mtp: downloading shard 1/2: model-00025-of-00026.safetensors',
      'mtp: shard 1/2 ready',
      'mtp: downloading shard 2/2: model-00026-of-00026.safetensors',
      'mtp: shard 2/2 ready',
      'mtp: extracted 19 tensor(s)',
      'mtp: quantized to 4-bit',
      'mtp: saved to /tmp/skulk-scratch/mtp.safetensors',
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
    lines: ['mtp: found mtp.* keys in 1 shard(s)', 'mtp: downloading shard 1/1: model.safetensors'],
    errorMessage: 'shards were downloaded but no mtp.* tensors could be read',
  },
};
