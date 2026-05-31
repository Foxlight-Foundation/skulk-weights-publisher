import type { Meta, StoryObj } from '@storybook/react-vite';
import { Field } from './Field';

const meta: Meta<typeof Field> = {
  title: 'Common/Field',
  component: Field,
  parameters: { layout: 'centered' },
};

export default meta;
type Story = StoryObj<typeof Field>;

export const Default: Story = {
  args: { placeholder: 'mlx-community/Qwen3-35B-A3B-4bit', style: { width: 360 } },
};

export const Password: Story = {
  args: { type: 'password', placeholder: 'hf_...', style: { width: 360 } },
};

export const Disabled: Story = {
  args: { placeholder: 'Disabled', disabled: true, style: { width: 360 } },
};
