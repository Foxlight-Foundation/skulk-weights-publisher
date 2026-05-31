import { useRef } from 'react';
import styled from 'styled-components';
import { usePublishStore } from '@/stores/publish.store';
import { Button } from '@/components/common/Button/Button';
import { Field } from '@/components/common/Field/Field';
import type { DetectFormProps } from './DetectForm.types';

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
 * URL input and Detect button. Reads/writes the publish store directly.
 */
export const DetectForm = ({ className }: DetectFormProps) => {
  const phase = usePublishStore((s) => s.phase);
  const url = usePublishStore((s) => s.url);
  const setUrl = usePublishStore((s) => s.setUrl);
  const detect = usePublishStore((s) => s.detect);
  const inputRef = useRef<HTMLInputElement>(null);

  const detecting = phase === 'detecting';
  const busy = phase === 'publishing' || phase === 'done';

  const handleDetect = () => {
    void detect();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !detecting && !busy && url.trim()) {
      handleDetect();
    }
  };

  return (
    <Wrapper className={className}>
      <FormLabel htmlFor="model-url-input">HuggingFace model URL</FormLabel>
      <InputRow>
        <Field
          ref={inputRef}
          id="model-url-input"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="mlx-community/Qwen3-35B-A3B-4bit"
          disabled={busy}
          style={{ flex: 1 }}
          aria-label="HuggingFace model URL"
        />
        <Button
          onClick={handleDetect}
          loading={detecting}
          disabled={!url.trim() || busy}
          aria-label="Detect model"
        >
          Detect
        </Button>
      </InputRow>
    </Wrapper>
  );
};
