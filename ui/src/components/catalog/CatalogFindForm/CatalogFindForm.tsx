import { useRef } from 'react';
import styled from 'styled-components';
import { useCatalogStore } from '@/stores/catalog.store';
import { Button } from '@/components/common/Button/Button';
import { Field } from '@/components/common/Field/Field';
import type { CatalogFindFormProps } from './CatalogFindForm.types';

const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const FormLabel = styled.label`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-weight: 500;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const InputRow = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.sm};
`;

/**
 * Source-model input and Find button for the read-only catalog reverse-lookup.
 * Reads/writes the catalog store directly.
 */
export const CatalogFindForm = ({ className }: CatalogFindFormProps) => {
  const phase = useCatalogStore((s) => s.phase);
  const query = useCatalogStore((s) => s.query);
  const setQuery = useCatalogStore((s) => s.setQuery);
  const find = useCatalogStore((s) => s.find);
  const inputRef = useRef<HTMLInputElement>(null);

  const finding = phase === 'finding';

  const handleFind = () => {
    void find();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !finding && query.trim()) {
      handleFind();
    }
  };

  return (
    <Wrapper className={className}>
      <FormLabel htmlFor="catalog-find-input">Source model URL or owner/repo</FormLabel>
      <InputRow>
        <Field
          ref={inputRef}
          id="catalog-find-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="google/gemma-3-4b-it"
          style={{ flex: 1 }}
          aria-label="Source model URL or owner/repo"
        />
        <Button
          onClick={handleFind}
          loading={finding}
          disabled={!query.trim()}
          aria-label="Find catalog entry"
        >
          Find
        </Button>
      </InputRow>
    </Wrapper>
  );
};
