import type { Meta, StoryObj } from '@storybook/react-vite';
import { DetectionResult } from './DetectionResult';
import type { DetectResponse } from '@/types/api';

const meta: Meta<typeof DetectionResult> = {
  title: 'Publish/DetectionResult',
  component: DetectionResult,
  parameters: { layout: 'padded' },
};

export default meta;
type Story = StoryObj<typeof DetectionResult>;

const withMtp: DetectResponse = {
  model_id: 'mlx-community/Qwen3-35B-A3B-4bit',
  base_model: 'Qwen/Qwen3-35B-A3B',
  quant: 'q4k',
  tier: 'smoke',
  mtp_key_count: 3,
  mtp_keys: ['mtp.0.embed_tokens.weight', 'mtp.1.embed_tokens.weight'],
  sidecar_repo: 'FoxlightAI/qwen3-35b-a3b-mtp',
  can_publish: true,
  assistant_model_repo: null,
  can_publish_assistant: false,
};

const noMtp: DetectResponse = {
  ...withMtp,
  mtp_key_count: 0,
  mtp_keys: [],
  can_publish: false,
  assistant_model_repo: null,
  can_publish_assistant: false,
};

const gemma4Assistant: DetectResponse = {
  model_id: 'mlx-community/gemma-4-27b-it-4bit',
  base_model: 'google/gemma-4-27b-it',
  quant: 'q4k',
  tier: 'moe',
  mtp_key_count: 0,
  mtp_keys: [],
  sidecar_repo: null,
  can_publish: false,
  assistant_model_repo: 'google/gemma-4-27b-it-assistant',
  can_publish_assistant: true,
};

export const Default: Story = { args: { detection: withMtp } };
export const NoMtpTensors: Story = { args: { detection: noMtp } };
export const Gemma4Assistant: Story = { args: { detection: gemma4Assistant } };
