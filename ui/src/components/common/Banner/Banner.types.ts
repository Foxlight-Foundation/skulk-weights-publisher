export type BannerSeverity = 'warning' | 'error' | 'info';

export interface BannerProps {
  severity?: BannerSeverity;
  children: React.ReactNode;
  onDismiss?: () => void;
  className?: string;
}
