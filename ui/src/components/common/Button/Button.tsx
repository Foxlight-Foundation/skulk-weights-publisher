import { forwardRef } from 'react';
import styled, { css } from 'styled-components';
import type { ButtonProps, ButtonSize, ButtonVariant } from './Button.types';
import { Spinner } from '@/components/common/Spinner/Spinner';

const sizeMap: Record<ButtonSize, { height: string; padding: string; fontSize: string }> = {
  sm: { height: '30px', padding: '0 10px', fontSize: '13px' },
  md: { height: '36px', padding: '0 14px', fontSize: '14px' },
  lg: { height: '42px', padding: '0 18px', fontSize: '15px' },
};

const variantStyles: Record<ButtonVariant, ReturnType<typeof css>> = {
  primary: css`
    background: ${({ theme }) => theme.colors.gold};
    color: ${({ theme }) => theme.colors.textOnAccent};
    border: 1px solid transparent;
    &:hover:not(:disabled) {
      background: ${({ theme }) => theme.colors.goldStrong};
    }
  `,
  outline: css`
    background: transparent;
    color: ${({ theme }) => theme.colors.gold};
    border: 1px solid ${({ theme }) => theme.colors.borderStrong};
    &:hover:not(:disabled) {
      background: ${({ theme }) => theme.colors.goldBg};
    }
  `,
  ghost: css`
    background: transparent;
    color: ${({ theme }) => theme.colors.textSecondary};
    border: 1px solid transparent;
    &:hover:not(:disabled) {
      background: ${({ theme }) => theme.colors.surfaceHover};
      color: ${({ theme }) => theme.colors.text};
    }
  `,
};

const StyledButton = styled.button<{
  $variant: ButtonVariant;
  $size: ButtonSize;
  $block: boolean;
  $loading: boolean;
}>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border-radius: ${({ theme }) => theme.radii.md};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 500;
  cursor: pointer;
  transition:
    background 0.15s,
    color 0.15s,
    border-color 0.15s;
  white-space: nowrap;
  position: relative;

  height: ${({ $size }) => sizeMap[$size].height};
  padding: ${({ $size }) => sizeMap[$size].padding};
  font-size: ${({ $size }) => sizeMap[$size].fontSize};
  width: ${({ $block }) => ($block ? '100%' : 'auto')};

  ${({ $variant }) => variantStyles[$variant]}

  &:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  ${({ $loading }) =>
    $loading &&
    css`
      color: transparent;
      pointer-events: none;
    `}
`;

const SpinnerOverlay = styled.span`
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
`;

/**
 * Primary action button used throughout skulk-ui.
 *
 * Use `variant="primary"` for the single most important action on a view.
 * Use `variant="ghost"` for secondary or destructive secondary actions.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      loading = false,
      block = false,
      disabled,
      children,
      ...rest
    },
    ref,
  ) => (
    <StyledButton
      ref={ref}
      $variant={variant}
      $size={size}
      $block={block}
      $loading={loading}
      disabled={disabled || loading}
      {...rest}
    >
      {children}
      {loading && (
        <SpinnerOverlay>
          <Spinner size={16} />
        </SpinnerOverlay>
      )}
    </StyledButton>
  ),
);

Button.displayName = 'Button';
