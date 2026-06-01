import styled from 'styled-components';
import { useCatalogStore } from '@/stores/catalog.store';
import { Button } from '@/components/common/Button/Button';
import { Banner } from '@/components/common/Banner/Banner';
import type { CatalogEntry } from '@/types/api';
import type { CatalogResultProps } from './CatalogResult.types';

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

const MatchCount = styled.p`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const Divider = styled.hr`
  border: none;
  border-top: 1px solid ${({ theme }) => theme.colors.borderLight};
  margin: ${({ theme }) => theme.spacing.sm} 0;
`;

const Row = ({ label, value }: { label: string; value: string }) => (
  <>
    <GridLabel>{label}</GridLabel>
    <GridValue>
      <Mono>{value}</Mono>
    </GridValue>
  </>
);

const EntryGrid = ({ entry }: { entry: CatalogEntry }) => (
  <Grid>
    <Row label="Key" value={entry.key} />
    <Row label="Source model" value={entry.source_model} />

    <GridLabel>Quant / Tier</GridLabel>
    <GridValue>
      <Pill>{entry.quant}</Pill> <Pill>{entry.tier}</Pill>
    </GridValue>

    <GridLabel>Slices</GridLabel>
    <GridValue>{entry.slices.join(', ')}</GridValue>

    <Row label="Vindex repo" value={entry.hf_repo} />
    {entry.mtp_sidecar_repo && <Row label="MTP sidecar" value={entry.mtp_sidecar_repo} />}
    {entry.assistant_model_repo && <Row label="Assistant" value={entry.assistant_model_repo} />}
    {entry.hf_collection && <Row label="Collection" value={entry.hf_collection} />}
  </Grid>
);

/**
 * Renders the outcome of a catalog reverse-lookup: the resolved entry grid on a
 * hit, a calm "not found" notice on a miss (a normal outcome), or an error
 * banner on an unexpected failure. Reads the catalog store directly.
 */
export const CatalogResult = ({ className }: CatalogResultProps) => {
  const phase = useCatalogStore((s) => s.phase);
  const entries = useCatalogStore((s) => s.entries);
  const sourceModel = useCatalogStore((s) => s.sourceModel);
  const errorMessage = useCatalogStore((s) => s.errorMessage);
  const reset = useCatalogStore((s) => s.reset);

  if (phase === 'notFound') {
    return (
      <Card className={className}>
        <Banner severity="info">
          No catalog entry found for <Mono>{sourceModel ?? 'that source model'}</Mono>. It may not
          be published yet, or the source model id may differ from the one recorded in the catalog.
        </Banner>
        <Actions>
          <Button variant="ghost" size="sm" onClick={reset}>
            Clear
          </Button>
        </Actions>
      </Card>
    );
  }

  if (phase === 'error') {
    return (
      <Card className={className}>
        <Banner severity="error">{errorMessage ?? 'Lookup failed.'}</Banner>
        <Actions>
          <Button variant="ghost" size="sm" onClick={reset}>
            Clear
          </Button>
        </Actions>
      </Card>
    );
  }

  if (phase !== 'found' || entries.length === 0) {
    return null;
  }

  return (
    <Card className={className}>
      {entries.length > 1 && (
        <MatchCount>
          {entries.length} entries match <Mono>{sourceModel ?? 'this source model'}</Mono>
        </MatchCount>
      )}
      {entries.map((entry, index) => (
        <div key={entry.key}>
          {index > 0 && <Divider />}
          <EntryGrid entry={entry} />
        </div>
      ))}
      <Actions>
        <Button variant="ghost" size="sm" onClick={reset}>
          Clear
        </Button>
      </Actions>
    </Card>
  );
};
