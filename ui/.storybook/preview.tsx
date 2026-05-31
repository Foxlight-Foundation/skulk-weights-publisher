import type { Preview } from '@storybook/react';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '../src/theme/theme';
import { GlobalStyle } from '../src/theme/GlobalStyle';

const preview: Preview = {
  decorators: [
    (Story) => (
      <ThemeProvider theme={lightTheme}>
        <GlobalStyle />
        <Story />
      </ThemeProvider>
    ),
  ],
  parameters: {
    backgrounds: {
      default: 'light',
      values: [{ name: 'light', value: lightTheme.colors.bg }],
    },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
};

export default preview;
