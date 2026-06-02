import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { useCatalogStore } from '@/stores/catalog.store';
import { CatalogFindForm } from './CatalogFindForm';

function renderWithTheme(ui: React.ReactElement) {
  return render(<ThemeProvider theme={lightTheme}>{ui}</ThemeProvider>);
}

beforeEach(() => {
  useCatalogStore.setState({
    phase: 'idle',
    query: '',
    entries: [],
    sourceModel: null,
    errorMessage: null,
  });
});

describe('CatalogFindForm', () => {
  it('renders the input and Find button', () => {
    renderWithTheme(<CatalogFindForm />);
    expect(screen.getByLabelText('Source model URL or owner/repo')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Find catalog entry' })).toBeInTheDocument();
  });

  it('Find button is disabled when the input is empty', () => {
    renderWithTheme(<CatalogFindForm />);
    expect(screen.getByRole('button', { name: 'Find catalog entry' })).toBeDisabled();
  });

  it('enables Find button when a query is typed', async () => {
    const user = userEvent.setup();
    renderWithTheme(<CatalogFindForm />);
    await user.type(
      screen.getByLabelText('Source model URL or owner/repo'),
      'google/gemma-3-4b-it',
    );
    expect(screen.getByRole('button', { name: 'Find catalog entry' })).toBeEnabled();
  });

  it('calls find when the Find button is clicked', async () => {
    const user = userEvent.setup();
    const find = vi.fn();
    useCatalogStore.setState({ query: 'google/gemma-3-4b-it', find });
    renderWithTheme(<CatalogFindForm />);
    await user.click(screen.getByRole('button', { name: 'Find catalog entry' }));
    expect(find).toHaveBeenCalledOnce();
  });

  it('calls find when Enter is pressed in the input', async () => {
    const user = userEvent.setup();
    const find = vi.fn();
    useCatalogStore.setState({ query: 'google/gemma-3-4b-it', find });
    renderWithTheme(<CatalogFindForm />);
    await user.click(screen.getByLabelText('Source model URL or owner/repo'));
    await user.keyboard('{Enter}');
    expect(find).toHaveBeenCalledOnce();
  });

  it('does not call find on Enter when the input is empty', async () => {
    const user = userEvent.setup();
    const find = vi.fn();
    useCatalogStore.setState({ query: '', find });
    renderWithTheme(<CatalogFindForm />);
    await user.click(screen.getByLabelText('Source model URL or owner/repo'));
    await user.keyboard('{Enter}');
    expect(find).not.toHaveBeenCalled();
  });
});
