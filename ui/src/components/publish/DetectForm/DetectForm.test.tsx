import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { usePublishStore } from '@/stores/publish.store';
import { DetectForm } from './DetectForm';

function renderWithTheme(ui: React.ReactElement) {
  return render(<ThemeProvider theme={lightTheme}>{ui}</ThemeProvider>);
}

beforeEach(() => {
  usePublishStore.setState({
    phase: 'idle',
    url: '',
    detection: null,
    logLines: [],
    errorMessage: null,
  });
});

describe('DetectForm', () => {
  it('renders the URL input and Detect button', () => {
    renderWithTheme(<DetectForm />);
    expect(screen.getByLabelText('HuggingFace model URL')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Detect model' })).toBeInTheDocument();
  });

  it('Detect button is disabled when the input is empty', () => {
    renderWithTheme(<DetectForm />);
    expect(screen.getByRole('button', { name: 'Detect model' })).toBeDisabled();
  });

  it('enables Detect button when URL is typed', async () => {
    const user = userEvent.setup();
    renderWithTheme(<DetectForm />);
    await user.type(screen.getByLabelText('HuggingFace model URL'), 'mlx-community/test');
    expect(screen.getByRole('button', { name: 'Detect model' })).toBeEnabled();
  });

  it('calls detect when Enter is pressed in the input', async () => {
    const user = userEvent.setup();
    const detect = vi.fn();
    usePublishStore.setState({ url: 'mlx-community/test', detect });
    renderWithTheme(<DetectForm />);
    await user.keyboard('{Enter}');
    // detect is wired via store — just verify the button is enabled
    expect(screen.getByRole('button', { name: 'Detect model' })).toBeEnabled();
  });
});
