import { useEffect, useState } from 'react';
import { ThemeProvider } from 'styled-components';
import { lightTheme } from '@/theme/theme';
import { GlobalStyle } from '@/theme/GlobalStyle';
import { useConfigStore } from '@/stores/config.store';
import { usePublishStore } from '@/stores/publish.store';
import { AppHeader } from '@/components/layout/AppHeader/AppHeader';
import { SettingsPanel } from '@/components/layout/SettingsPanel/SettingsPanel';
import { Banner } from '@/components/common/Banner/Banner';
import { DetectForm } from '@/components/publish/DetectForm/DetectForm';
import { DetectionResult } from '@/components/publish/DetectionResult/DetectionResult';
import { PublishLog } from '@/components/publish/PublishLog/PublishLog';
import styled from 'styled-components';

const Main = styled.main`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  padding: ${({ theme }) => theme.spacing.xl} ${({ theme }) => theme.spacing.lg};
  gap: ${({ theme }) => theme.spacing.lg};
  max-width: 720px;
  width: 100%;
  margin: 0 auto;
`;

const Card = styled.div`
  width: 100%;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.xl};
  box-shadow:
    0 1px 3px ${({ theme }) => theme.colors.shadow},
    0 4px 12px ${({ theme }) => theme.colors.shadow};
  padding: ${({ theme }) => theme.spacing.lg};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
`;

const CardTitle = styled.h2`
  font-size: ${({ theme }) => theme.fontSizes.lg};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text};
`;

function AppContent() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [tokenWarningDismissed, setTokenWarningDismissed] = useState(false);

  const hfTokenSet = useConfigStore((s) => s.hfTokenSet);
  const mlxAvailable = useConfigStore((s) => s.mlxAvailable);
  const fetchStatus = useConfigStore((s) => s.fetchStatus);

  const phase = usePublishStore((s) => s.phase);
  const detection = usePublishStore((s) => s.detection);
  const logLines = usePublishStore((s) => s.logLines);
  const errorMessage = usePublishStore((s) => s.errorMessage);

  useEffect(() => {
    void fetchStatus();
  }, [fetchStatus]);

  const showTokenWarning = !hfTokenSet && !tokenWarningDismissed;
  const showDetection =
    detection !== null &&
    (phase === 'detected' || phase === 'publishing' || phase === 'done' || phase === 'error');
  const showLog = phase === 'publishing' || phase === 'done' || phase === 'error';

  return (
    <>
      <AppHeader onOpenSettings={() => setSettingsOpen(true)} />
      <Main>
        {showTokenWarning && (
          <Banner onDismiss={() => setTokenWarningDismissed(true)}>
            <strong>HF token not configured.</strong> Open{' '}
            <button
              onClick={() => setSettingsOpen(true)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                textDecoration: 'underline',
                color: 'inherit',
                font: 'inherit',
                padding: 0,
              }}
            >
              Settings
            </button>{' '}
            to configure your HuggingFace token before publishing.
          </Banner>
        )}
        {!mlxAvailable && (
          <Banner severity="error">
            <strong>mlx is not installed.</strong> MTP extraction requires Apple Silicon and the{' '}
            <code>[mtp]</code> extras: <code>uv sync --extra ui --extra mtp</code>
          </Banner>
        )}

        <Card>
          <CardTitle>Publish MTP Sidecar</CardTitle>
          <DetectForm />
          {showDetection && detection && (
            <DetectionResult detection={detection} mlxAvailable={mlxAvailable} />
          )}
        </Card>

        {showLog && <PublishLog phase={phase} lines={logLines} errorMessage={errorMessage} />}
      </Main>

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
}

export function App() {
  return (
    <ThemeProvider theme={lightTheme}>
      <GlobalStyle />
      <AppContent />
    </ThemeProvider>
  );
}
