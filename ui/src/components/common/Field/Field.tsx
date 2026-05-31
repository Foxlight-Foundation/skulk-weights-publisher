import { forwardRef } from 'react';
import styled, { css } from 'styled-components';
import type { FieldProps, FieldSize } from './Field.types';

const sizeMap: Record<FieldSize, { height: string; fontSize: string; padding: string }> = {
  sm: { height: '30px', fontSize: '13px', padding: '0 8px' },
  md: { height: '36px', fontSize: '14px', padding: '0 12px' },
  lg: { height: '42px', fontSize: '15px', padding: '0 14px' },
};

const Wrapper = styled.div<{ $size: FieldSize; $disabled: boolean }>`
  display: flex;
  align-items: center;
  gap: 6px;
  height: ${({ $size }) => sizeMap[$size].height};
  padding: ${({ $size }) => sizeMap[$size].padding};
  font-size: ${({ $size }) => sizeMap[$size].fontSize};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  transition: border-color 0.15s;

  &:focus-within {
    border-color: ${({ theme }) => theme.colors.gold};
    outline: none;
  }

  ${({ $disabled }) =>
    $disabled &&
    css`
      opacity: 0.5;
      cursor: not-allowed;
    `}
`;

const Input = styled.input`
  flex: 1;
  min-width: 0;
  background: transparent;
  border: none;
  outline: none;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: inherit;
  color: ${({ theme }) => theme.colors.text};

  &::placeholder {
    color: ${({ theme }) => theme.colors.textMuted};
  }

  &:disabled {
    cursor: not-allowed;
  }
`;

const Slot = styled.span`
  display: flex;
  align-items: center;
  color: ${({ theme }) => theme.colors.textMuted};
  flex-shrink: 0;
`;

/**
 * Text/password input with optional leading icon and trailing element.
 * Exposes the underlying input ref for programmatic focus.
 */
export const Field = forwardRef<HTMLInputElement, FieldProps>(
  ({ size = 'md', icon, rightElement, disabled, className, ...inputProps }, ref) => (
    <Wrapper $size={size} $disabled={!!disabled} className={className}>
      {icon && <Slot>{icon}</Slot>}
      <Input ref={ref} disabled={disabled} {...inputProps} />
      {rightElement && <Slot>{rightElement}</Slot>}
    </Wrapper>
  ),
);

Field.displayName = 'Field';
