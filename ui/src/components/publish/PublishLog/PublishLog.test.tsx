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
          'mtp: found mtp.* keys in 2 shard(s)',
          'mtp: downloading shard 1/2: model-00025-of-00026.safetensors',
        ]}
        errorMessage={null}
      />,
    );
    expect(screen.getByText('Downloading shards')).toBeInTheDocument();
    expect(screen.getAllByText(/downloading shard 1\/2/).length).toBeGreaterThan(0);
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
        lines={['mtp: found mtp.* keys in 1 shard(s)']}
        errorMessage={null}
      />,
    );
    expect(screen.getByText(/found mtp/)).toBeInTheDocument();
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
