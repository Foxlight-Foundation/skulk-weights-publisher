import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { Field } from './Field';

function renderWithTheme(ui: React.ReactElement) {
  return render(<ThemeProvider theme={lightTheme}>{ui}</ThemeProvider>);
}

describe('Field', () => {
  it('renders an accessible text input', () => {
    renderWithTheme(<Field aria-label="Model URL" placeholder="mlx-community/..." />);
    expect(screen.getByRole('textbox', { name: 'Model URL' })).toBeInTheDocument();
  });

  it('accepts typed input', async () => {
    const user = userEvent.setup();
    renderWithTheme(<Field aria-label="URL" defaultValue="" />);
    const input = screen.getByRole('textbox');
    await user.type(input, 'mlx-community/test');
    expect(input).toHaveValue('mlx-community/test');
  });

  it('forwards the ref', () => {
    const ref = { current: null } as React.RefObject<HTMLInputElement | null>;
    render(
      <ThemeProvider theme={lightTheme}>
        <Field ref={ref} aria-label="ref test" />
      </ThemeProvider>,
    );
    expect(ref.current).toBeInstanceOf(HTMLInputElement);
  });
});
