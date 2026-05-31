import type { Meta, StoryObj } from '@storybook/react-vite';
import { DetectForm } from './DetectForm';

const meta: Meta<typeof DetectForm> = {
  title: 'Publish/DetectForm',
  component: DetectForm,
  parameters: { layout: 'padded' },
};

export default meta;
type Story = StoryObj<typeof DetectForm>;

export const Default: Story = {};
