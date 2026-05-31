import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { SettingsPanel } from './SettingsPanel';

vi.mock('@/api/client', () => ({
  fetchConfig: vi.fn().mockResolvedValue({ hf_token_masked: null }),
  saveConfig: vi.fn().mockResolvedValue({ ok: true }),
}));

vi.mock('@/stores/config.store', () => ({
  useConfigStore: (sel: (s: { saveToken: () => Promise<void> }) => unknown) =>
    sel({ saveToken: vi.fn().mockResolvedValue(undefined) }),
}));

function renderWithTheme(ui: React.ReactElement) {
  return render(<ThemeProvider theme={lightTheme}>{ui}</ThemeProvider>);
}

describe('SettingsPanel', () => {
  it('renders nothing when closed', () => {
    renderWithTheme(<SettingsPanel open={false} onClose={() => {}} />);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('renders when open', () => {
    renderWithTheme(<SettingsPanel open onClose={() => {}} />);
    expect(screen.getByRole('dialog', { name: 'Settings' })).toBeInTheDocument();
  });

  it('calls onClose when the close button is clicked', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderWithTheme(<SettingsPanel open onClose={onClose} />);
    await user.click(screen.getByRole('button', { name: 'Close settings' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('calls onClose when the backdrop is clicked', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderWithTheme(<SettingsPanel open onClose={onClose} />);
    // The backdrop is the first focusable element behind the drawer
    const backdrop = screen.getByRole('dialog').previousElementSibling as HTMLElement;
    await user.click(backdrop);
    expect(onClose).toHaveBeenCalledOnce();
  });
});
