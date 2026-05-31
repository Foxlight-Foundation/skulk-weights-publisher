import type { Meta, StoryObj } from '@storybook/react-vite';
import { AppHeader } from './AppHeader';

const meta: Meta<typeof AppHeader> = {
  title: 'Layout/AppHeader',
  component: AppHeader,
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof AppHeader>;

export const Default: Story = { args: { onOpenSettings: () => {} } };
