import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { AppHeader } from './AppHeader';

function renderWithTheme(ui: React.ReactElement) {
  return render(<ThemeProvider theme={lightTheme}>{ui}</ThemeProvider>);
}

describe('AppHeader', () => {
  it('renders the brand name', () => {
    renderWithTheme(<AppHeader onOpenSettings={() => {}} />);
    expect(screen.getByText('Skulk Weights Publisher')).toBeInTheDocument();
  });

  it('calls onOpenSettings when the gear button is clicked', async () => {
    const user = userEvent.setup();
    const onOpenSettings = vi.fn();
    renderWithTheme(<AppHeader onOpenSettings={onOpenSettings} />);
    await user.click(screen.getByRole('button', { name: 'Open settings' }));
    expect(onOpenSettings).toHaveBeenCalledOnce();
  });
});
