import { render, screen } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { usePublishStore } from '@/stores/publish.store';
import { DetectionResult } from './DetectionResult';
import type { DetectResponse } from '@/types/api';

function renderWithTheme(ui: React.ReactElement) {
  return render(<ThemeProvider theme={lightTheme}>{ui}</ThemeProvider>);
}

const baseDetection: DetectResponse = {
  model_id: 'mlx-community/Qwen3-35B-A3B-4bit',
  base_model: 'Qwen/Qwen3-35B-A3B',
  quant: 'q4k',
  tier: 'smoke',
  mtp_key_count: 3,
  mtp_keys: ['mtp.0.embed_tokens.weight'],
  sidecar_repo: 'FoxlightAI/qwen3-35b-a3b-mtp',
  can_publish: true,
  assistant_model_repo: null,
  can_publish_assistant: false,
};

const gemma4Detection: DetectResponse = {
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

beforeEach(() => {
  usePublishStore.setState({ phase: 'detected' });
});

describe('DetectionResult', () => {
  it('renders model_id and base_model', () => {
    renderWithTheme(<DetectionResult detection={baseDetection} />);
    expect(screen.getByText('mlx-community/Qwen3-35B-A3B-4bit')).toBeInTheDocument();
    expect(screen.getByText('Qwen/Qwen3-35B-A3B')).toBeInTheDocument();
  });

  it('enables Publish when can_publish is true', () => {
    renderWithTheme(<DetectionResult detection={baseDetection} />);
    expect(screen.getByRole('button', { name: 'Publish MTP sidecar' })).toBeEnabled();
  });

  it('disables Publish and shows explanation when can_publish is false and no assistant', () => {
    const noMtp: DetectResponse = {
      ...baseDetection,
      mtp_key_count: 0,
      can_publish: false,
      assistant_model_repo: null,
      can_publish_assistant: false,
    };
    renderWithTheme(<DetectionResult detection={noMtp} />);
    expect(screen.getByRole('button', { name: 'Publish MTP sidecar' })).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent('No MTP tensors found');
  });

  it('shows the target sidecar repo', () => {
    renderWithTheme(<DetectionResult detection={baseDetection} />);
    expect(screen.getByText('FoxlightAI/qwen3-35b-a3b-mtp')).toBeInTheDocument();
  });

  it('shows assistant model row and Gemma 4 banner when assistant is detected', () => {
    renderWithTheme(<DetectionResult detection={gemma4Detection} />);
    expect(screen.getByText('google/gemma-4-27b-it-assistant')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Gemma 4');
    expect(screen.getByRole('alert')).toHaveTextContent('companion assistant pattern');
  });

  it('shows Register in Catalog button when can_publish_assistant is true', () => {
    renderWithTheme(<DetectionResult detection={gemma4Detection} />);
    expect(screen.getByRole('button', { name: 'Register in Catalog' })).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Publish MTP sidecar' })).not.toBeInTheDocument();
  });

  it('does not show assistant banner for MTP models', () => {
    renderWithTheme(<DetectionResult detection={baseDetection} />);
    // Should not show the Gemma 4 banner
    expect(screen.queryByText(/companion assistant pattern/)).not.toBeInTheDocument();
  });
});
