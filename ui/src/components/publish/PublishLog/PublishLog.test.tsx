import { render, screen } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { PublishLog } from './PublishLog';

function renderWithTheme(ui: React.ReactElement) {
  return render(<ThemeProvider theme={lightTheme}>{ui}</ThemeProvider>);
}

describe('PublishLog', () => {
  it('renders the log panel with accessible label', () => {
    renderWithTheme(<PublishLog phase="publishing" lines={[]} errorMessage={null} />);
    expect(screen.getByRole('region', { name: 'Publish log' })).toBeInTheDocument();
  });

  it('shows all stage labels', () => {
    renderWithTheme(<PublishLog phase="publishing" lines={[]} errorMessage={null} />);
    expect(screen.getByText('Finding MTP shards')).toBeInTheDocument();
    expect(screen.getByText('Downloading shards')).toBeInTheDocument();
    expect(screen.getByText('Uploading to HuggingFace')).toBeInTheDocument();
  });

  it('activates the downloading stage when a shard download line appears', () => {
    renderWithTheme(
      <PublishLog
        phase="publishing"
        lines={[
          "mtp: found MTP tensors in 2 shard(s) (prefix: 'mtp.')",
          'mtp: downloading shard 1/2: model-00025-of-00026.safetensors',
        ]}
        errorMessage={null}
      />,
    );
    expect(screen.getByText('Downloading shards')).toBeInTheDocument();
    expect(screen.getAllByText(/downloading shard 1\/2/).length).toBeGreaterThan(0);
  });

  // Contract test: every line below is a verbatim emit string from
  // src/skulk_weights_publisher/mtp_extractor.py (extract_mtp / _ProgressFile).
  // If a stage stays pending here, the backend and STAGE_DEFS have drifted —
  // fix them together.
  it('matches real backend emit strings — every stage completes', () => {
    const realPublishLog = [
      "mtp: found MTP tensors in 2 shard(s) (prefix: 'mtp.')",
      'mtp: downloading shard 1/2: model-00049-of-00050.safetensors',
      'mtp: shard 1/2 ready',
      'mtp: downloading shard 2/2: model-00050-of-00050.safetensors',
      'mtp: shard 2/2 ready',
      'mtp: streaming tensors to disk (bf16, unquantized)...',
      'mtp: saved 1040 tensor(s) at bf16 (unquantized) to /tmp/scratch/mtp.safetensors',
      'mtp: uploading to hf://FoxlightAI/some-model-mtp/mtp.safetensors',
      'mtp: uploading 42% (3.2 GB / 7.6 GB)',
      'mtp: published to hf://FoxlightAI/some-model-mtp/mtp.safetensors',
      'publish complete',
    ];
    renderWithTheme(<PublishLog phase="done" lines={realPublishLog} errorMessage={null} />);
    // Every non-assistant stage must render as done (✓), none stuck pending.
    const items = screen.getAllByRole('listitem');
    const pendingLabels = items
      .filter((li) => li.textContent && !li.textContent.includes('✓'))
      .map((li) => li.textContent);
    // Only the assistant-flow 'confirming' stage may remain pending.
    expect(pendingLabels.filter((l) => !/assistant/i.test(l ?? ''))).toEqual([]);
  });

  it('shows upload percentage detail while uploading', () => {
    renderWithTheme(
      <PublishLog
        phase="publishing"
        lines={[
          "mtp: found MTP tensors in 1 shard(s) (prefix: 'mtp.')",
          'mtp: streaming tensors to disk (bf16, unquantized)...',
          'mtp: saved 19 tensor(s) at bf16 (unquantized) to /tmp/scratch/mtp.safetensors',
          'mtp: uploading to hf://FoxlightAI/some-model-mtp/mtp.safetensors',
          'mtp: uploading 17% (1.3 GB / 7.6 GB)',
        ]}
        errorMessage={null}
      />,
    );
    expect(screen.getByText(/17% · 1\.3 GB \/ 7\.6 GB/)).toBeInTheDocument();
  });

  it('shows Done badge when phase is done', () => {
    renderWithTheme(<PublishLog phase="done" lines={['publish complete']} errorMessage={null} />);
    expect(screen.getByText('Done')).toBeInTheDocument();
  });

  it('shows error message when phase is error', () => {
    renderWithTheme(<PublishLog phase="error" lines={[]} errorMessage="no mtp keys found" />);
    expect(screen.getByText(/no mtp keys found/)).toBeInTheDocument();
  });

  it('shows raw log lines in the collapsible section', () => {
    renderWithTheme(
      <PublishLog
        phase="publishing"
        lines={["mtp: found MTP tensors in 1 shard(s) (prefix: 'mtp.')"]}
        errorMessage={null}
      />,
    );
    expect(screen.getAllByText(/found MTP tensors/).length).toBeGreaterThan(0);
  });

  it('shows a Registered-in-catalog completion for the assistant flow', () => {
    renderWithTheme(
      <PublishLog
        phase="done"
        lines={[
          'registered foxlight/gemma-4-27b-full-q4-k in catalog',
          'assistant: google/gemma-4-27b-it-assistant',
          'written to /pkg/catalogues/foxlight.yaml',
        ]}
        errorMessage={null}
      />,
    );
    // Registration completion shown, not the MTP stage list left pending.
    expect(screen.getByText('Registered in catalog')).toBeInTheDocument();
    expect(screen.queryByText('Finding MTP shards')).not.toBeInTheDocument();
  });
});
