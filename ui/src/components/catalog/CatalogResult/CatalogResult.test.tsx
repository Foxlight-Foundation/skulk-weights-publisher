import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { useCatalogStore } from '@/stores/catalog.store';
import type { CatalogEntry } from '@/types/api';
import { CatalogResult } from './CatalogResult';

function renderWithTheme(ui: React.ReactElement) {
  return render(<ThemeProvider theme={lightTheme}>{ui}</ThemeProvider>);
}

const ENTRY: CatalogEntry = {
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
  mtp_quant: null,
  assistant_model_repo: null,
};

beforeEach(() => {
  useCatalogStore.setState({
    phase: 'idle',
    query: '',
    entry: null,
    sourceModel: null,
    errorMessage: null,
  });
});

describe('CatalogResult', () => {
  it('renders nothing in the idle phase', () => {
    const { container } = renderWithTheme(<CatalogResult />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders the resolved entry in the found phase', () => {
    useCatalogStore.setState({ phase: 'found', entry: ENTRY, sourceModel: ENTRY.source_model });
    renderWithTheme(<CatalogResult />);
    expect(screen.getByText('foxlight/gemma-3-4b-full-q4-k')).toBeInTheDocument();
    expect(screen.getByText('FoxlightAI/gemma-3-4b-full-q4-k-vindex')).toBeInTheDocument();
    expect(screen.getByText('q4k')).toBeInTheDocument();
  });

  it('shows MTP sidecar and assistant rows only when present', () => {
    useCatalogStore.setState({
      phase: 'found',
      entry: {
        ...ENTRY,
        mtp_sidecar_repo: 'FoxlightAI/some-mtp',
        assistant_model_repo: 'google/gemma-4-27b-it-assistant',
      },
      sourceModel: ENTRY.source_model,
    });
    renderWithTheme(<CatalogResult />);
    expect(screen.getByText('FoxlightAI/some-mtp')).toBeInTheDocument();
    expect(screen.getByText('google/gemma-4-27b-it-assistant')).toBeInTheDocument();
  });

  it('shows a calm not-found notice in the notFound phase', () => {
    useCatalogStore.setState({
      phase: 'notFound',
      sourceModel: 'mlx-community/unknown',
      errorMessage: "no catalog entry found for source_model 'mlx-community/unknown'",
    });
    renderWithTheme(<CatalogResult />);
    expect(screen.getByText(/No catalog entry found for/)).toBeInTheDocument();
    expect(screen.getByText('mlx-community/unknown')).toBeInTheDocument();
  });

  it('shows the error message in the error phase', () => {
    useCatalogStore.setState({ phase: 'error', errorMessage: 'network exploded' });
    renderWithTheme(<CatalogResult />);
    expect(screen.getByText('network exploded')).toBeInTheDocument();
  });

  it('Clear button resets the store', async () => {
    const user = userEvent.setup();
    useCatalogStore.setState({ phase: 'found', entry: ENTRY, sourceModel: ENTRY.source_model });
    renderWithTheme(<CatalogResult />);
    await user.click(screen.getByRole('button', { name: 'Clear' }));
    expect(useCatalogStore.getState().phase).toBe('idle');
  });
});
