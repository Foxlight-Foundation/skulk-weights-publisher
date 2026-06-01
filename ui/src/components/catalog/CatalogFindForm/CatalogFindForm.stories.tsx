import type { Meta, StoryObj } from '@storybook/react-vite';
import { CatalogFindForm } from './CatalogFindForm';

const meta: Meta<typeof CatalogFindForm> = {
  title: 'Catalog/CatalogFindForm',
  component: CatalogFindForm,
  parameters: { layout: 'padded' },
};

export default meta;
type Story = StoryObj<typeof CatalogFindForm>;

export const Default: Story = {};
