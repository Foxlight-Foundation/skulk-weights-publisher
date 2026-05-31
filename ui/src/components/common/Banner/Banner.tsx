import styled, { css } from 'styled-components';
import type { BannerProps, BannerSeverity } from './Banner.types';

const severityStyles: Record<BannerSeverity, ReturnType<typeof css>> = {
  warning: css`
    background: ${({ theme }) => theme.colors.warningBg};
    border-color: ${({ theme }) => theme.colors.warning};
    color: ${({ theme }) => theme.colors.warningText};
  `,
  error: css`
    background: ${({ theme }) => theme.colors.errorBg};
    border-color: ${({ theme }) => theme.colors.error};
    color: ${({ theme }) => theme.colors.errorText};
  `,
  info: css`
    background: ${({ theme }) => theme.colors.infoBg};
    border-color: ${({ theme }) => theme.colors.info};
    color: ${({ theme }) => theme.colors.info};
  `,
};

const Strip = styled.div<{ $severity: BannerSeverity }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${({ theme }) => theme.spacing.md};
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  ${({ $severity }) => severityStyles[$severity]}
`;

const DismissButton = styled.button`
  background: none;
  border: none;
  cursor: pointer;
  color: inherit;
  opacity: 0.7;
  font-size: 16px;
  line-height: 1;
  padding: 0 2px;
  flex-shrink: 0;

  &:hover {
    opacity: 1;
  }
`;

/**
 * Full-width status strip for warnings, errors, and info messages.
 * Pass `onDismiss` to render a close button.
 */
export const Banner = ({ severity = 'warning', children, onDismiss, className }: BannerProps) => (
  <Strip $severity={severity} className={className} role="alert">
    <span>{children}</span>
    {onDismiss && (
      <DismissButton onClick={onDismiss} aria-label="Dismiss">
        ✕
      </DismissButton>
    )}
  </Strip>
);
