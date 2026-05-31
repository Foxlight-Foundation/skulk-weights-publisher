import type { Meta, StoryObj } from '@storybook/react-vite';
import { SettingsPanel } from './SettingsPanel';

const meta: Meta<typeof SettingsPanel> = {
  title: 'Layout/SettingsPanel',
  component: SettingsPanel,
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof SettingsPanel>;

export const Default: Story = { args: { open: true, onClose: () => {} } };
export const Closed: Story = { args: { open: false, onClose: () => {} } };
