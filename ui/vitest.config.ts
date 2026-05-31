import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      thresholds: {
        lines: 70,
        branches: 70,
        functions: 70,
        statements: 70,
      },
      exclude: [
        'node_modules',
        'dist',
        '**/*.stories.tsx',
        '**/*.types.ts',
        'src/test-setup.ts',
        'src/main.tsx',
        'src/App.tsx',
        'src/types/**',
        'src/theme/styled.d.ts',
        'src/theme/GlobalStyle.ts',
        '.storybook/**',
        'eslint.config.js',
        'vite.config.ts',
        'vitest.config.ts',
      ],
    },
  },
});
