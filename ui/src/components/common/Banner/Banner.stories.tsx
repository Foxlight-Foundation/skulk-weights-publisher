import type { Meta, StoryObj } from '@storybook/react-vite';
import { Banner } from './Banner';

const meta: Meta<typeof Banner> = {
  title: 'Common/Banner',
  component: Banner,
  parameters: { layout: 'padded' },
};

export default meta;
type Story = StoryObj<typeof Banner>;

export const Default: Story = {
  args: { children: 'HF token not set. Open Settings to configure your HuggingFace token.' },
};

export const WithDismiss: Story = {
  args: {
    children: 'HF token not set. Open Settings to configure your HuggingFace token.',
    onDismiss: () => {},
  },
};

export const Error: Story = {
  args: { severity: 'error', children: 'Detection failed: model not found.' },
};

export const Info: Story = {
  args: { severity: 'info', children: 'No MTP tensors detected in this model.' },
};
