import styled from 'styled-components';
import type { AppHeaderProps } from './AppHeader.types';

const Bar = styled.header`
  position: sticky;
  top: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 52px;
  padding: 0 ${({ theme }) => theme.spacing.lg};
  background: ${({ theme }) => theme.colors.header};
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid ${({ theme }) => theme.colors.borderLight};
`;

const Brand = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const BrandName = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text};
  letter-spacing: -0.01em;
`;

const Badge = styled.span`
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.gold};
  background: ${({ theme }) => theme.colors.goldBg};
  border: 1px solid ${({ theme }) => theme.colors.borderStrong};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 1px 5px;
`;

const SettingsButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: ${({ theme }) => theme.radii.md};
  background: transparent;
  border: 1px solid transparent;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 16px;
  transition:
    background 0.15s,
    color 0.15s;

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
    color: ${({ theme }) => theme.colors.text};
  }
`;

/**
 * Top navigation bar with brand name and settings gear.
 */
export const AppHeader = ({ onOpenSettings }: AppHeaderProps) => (
  <Bar>
    <Brand>
      <BrandName>Skulk Weights Publisher</BrandName>
      <Badge>MTP</Badge>
    </Brand>
    <SettingsButton onClick={onOpenSettings} aria-label="Open settings">
      ⚙
    </SettingsButton>
  </Bar>
);
