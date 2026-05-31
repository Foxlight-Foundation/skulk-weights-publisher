import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { Banner } from './Banner';

function renderWithTheme(ui: React.ReactElement) {
  return render(<ThemeProvider theme={lightTheme}>{ui}</ThemeProvider>);
}

describe('Banner', () => {
  it('renders its message', () => {
    renderWithTheme(<Banner>HF token not configured</Banner>);
    expect(screen.getByRole('alert')).toHaveTextContent('HF token not configured');
  });

  it('does not render a dismiss button without onDismiss', () => {
    renderWithTheme(<Banner>Info</Banner>);
    expect(screen.queryByRole('button', { name: 'Dismiss' })).not.toBeInTheDocument();
  });

  it('calls onDismiss when the close button is clicked', async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    renderWithTheme(<Banner onDismiss={onDismiss}>Warning</Banner>);
    await user.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(onDismiss).toHaveBeenCalledOnce();
  });
});
