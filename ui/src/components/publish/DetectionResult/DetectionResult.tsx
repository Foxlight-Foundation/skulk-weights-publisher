import styled from 'styled-components';
import { usePublishStore } from '@/stores/publish.store';
import { Button } from '@/components/common/Button/Button';
import { Banner } from '@/components/common/Banner/Banner';
import type { DetectionResultProps } from './DetectionResult.types';

const Card = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
  padding: ${({ theme }) => theme.spacing.md};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.lg};
`;

const Grid = styled.dl`
  display: grid;
  grid-template-columns: 130px 1fr;
  gap: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.md};
  align-items: baseline;
`;

const GridLabel = styled.dt`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textMuted};
  text-transform: uppercase;
  letter-spacing: 0.03em;
`;

const GridValue = styled.dd`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.text};
`;

const Mono = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
`;

const Pill = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  font-weight: 600;
  padding: 1px 7px;
  border-radius: 999px;
  background: ${({ theme }) => theme.colors.goldBg};
  color: ${({ theme }) => theme.colors.gold};
  border: 1px solid ${({ theme }) => theme.colors.borderStrong};
  text-transform: uppercase;
  letter-spacing: 0.03em;
`;

const Actions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: ${({ theme }) => theme.spacing.sm};
  padding-top: ${({ theme }) => theme.spacing.sm};
  border-top: 1px solid ${({ theme }) => theme.colors.borderLight};
`;

/**
 * Shows detected model metadata and exposes the Publish MTP button.
 * The Publish button is disabled with an explanation when no MTP tensors are found.
 */
export const DetectionResult = ({
  detection,
  mlxAvailable = true,
  className,
}: DetectionResultProps) => {
  const phase = usePublishStore((s) => s.phase);
  const publish = usePublishStore((s) => s.publish);
  const reset = usePublishStore((s) => s.reset);
  const publishing = phase === 'publishing';

  return (
    <Card className={className}>
      <Grid>
        <GridLabel>Model</GridLabel>
        <GridValue>
          <Mono>{detection.model_id}</Mono>
        </GridValue>

        <GridLabel>Base model</GridLabel>
        <GridValue>
          {detection.base_model ? (
            <Mono>{detection.base_model}</Mono>
          ) : (
            <span style={{ color: 'var(--warning)' }}>Not detected</span>
          )}
        </GridValue>

        <GridLabel>Quant / Tier</GridLabel>
        <GridValue>
          <Pill>{detection.quant}</Pill> <Pill>{detection.tier}</Pill>
        </GridValue>

        <GridLabel>MTP tensors</GridLabel>
        <GridValue>
          {detection.mtp_key_count > 0 ? (
            <>{detection.mtp_key_count} keys found</>
          ) : (
            <span>None</span>
          )}
        </GridValue>

        {detection.sidecar_repo && (
          <>
            <GridLabel>Target repo</GridLabel>
            <GridValue>
              <Mono>{detection.sidecar_repo}</Mono>
            </GridValue>
          </>
        )}

        {detection.assistant_model_repo && (
          <>
            <GridLabel>Assistant model</GridLabel>
            <GridValue>
              <Mono>{detection.assistant_model_repo}</Mono>
            </GridValue>
          </>
        )}
      </Grid>

      {detection.assistant_model_repo && (
        <Banner severity="info">
          This model uses Gemma 4&apos;s companion assistant pattern. The assistant model is
          pre-published by Google — no tensor extraction needed.
        </Banner>
      )}
      {!detection.can_publish && !detection.can_publish_assistant && (
        <Banner severity="info">
          No MTP tensors found in the base model. This model does not have native MTP heads and
          cannot be published as an MTP sidecar.
        </Banner>
      )}
      {detection.can_publish && !mlxAvailable && (
        <Banner severity="error">
          mlx is not installed. Install it with <code>uv sync --extra ui --extra mtp</code> then
          restart skulk-ui.
        </Banner>
      )}

      <Actions>
        <Button variant="ghost" size="sm" onClick={reset} disabled={publishing}>
          Reset
        </Button>
        {detection.can_publish_assistant && !detection.can_publish ? (
          <Button
            variant="primary"
            onClick={() => void publish()}
            disabled={false}
            loading={publishing}
            aria-label="Register in Catalog"
          >
            Register in Catalog
          </Button>
        ) : (
          <Button
            variant="primary"
            onClick={() => void publish()}
            disabled={!detection.can_publish || !mlxAvailable}
            loading={publishing}
            aria-label="Publish MTP sidecar"
          >
            Publish MTP
          </Button>
        )}
      </Actions>
    </Card>
  );
};
