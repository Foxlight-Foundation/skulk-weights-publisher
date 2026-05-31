import styled, { keyframes } from 'styled-components';
import type { SpinnerProps } from './Spinner.types';

const spin = keyframes`
  to { transform: rotate(360deg); }
`;

const Ring = styled.span<{ $size: number }>`
  display: inline-block;
  width: ${({ $size }) => $size}px;
  height: ${({ $size }) => $size}px;
  border-radius: 50%;
  border: 2px solid ${({ theme }) => theme.colors.borderLight};
  border-top-color: ${({ theme }) => theme.colors.gold};
  animation: ${spin} 0.7s linear infinite;
  flex-shrink: 0;
`;

/**
 * Animated loading ring used in buttons and inline loading states.
 */
export const Spinner = ({ size = 20, className }: SpinnerProps) => (
  <Ring $size={size} className={className} role="status" aria-label="Loading" />
);
