import { useEffect, useRef } from 'react';
import styled from 'styled-components';
import { Spinner } from '@/components/common/Spinner/Spinner';
import type { PublishLogProps, Stage, StageStatus } from './PublishLog.types';

// ---------------------------------------------------------------------------
// Stage parsing
// ---------------------------------------------------------------------------

const STAGE_DEFS: { id: Stage; label: string; trigger: RegExp }[] = [
  { id: 'finding', label: 'Finding MTP shards', trigger: /mtp: found mtp\.\* keys/ },
  { id: 'downloading', label: 'Downloading shards', trigger: /mtp: downloading shard/ },
  {
    id: 'extracting',
    label: 'Extracting tensors',
    trigger: /mtp: extracted|mtp: streaming tensors/,
  },
  { id: 'quantizing', label: 'Quantizing', trigger: /mtp: quantized/ },
  { id: 'saving', label: 'Saving locally', trigger: /mtp: saved/ },
  { id: 'uploading', label: 'Uploading to HuggingFace', trigger: /mtp: uploading/ },
  {
    id: 'confirming',
    label: 'Confirming assistant model',
    trigger: /assistant: confirming companion model/,
  },
  { id: 'done', label: 'Published', trigger: /mtp: published|publish complete|assistant: ready/ },
];

function deriveStages(
  lines: string[],
  isError: boolean,
): { stage: Stage; label: string; status: StageStatus; detail?: string }[] {
  const reached = new Set<Stage>();
  let latestReached: Stage | null = null;

  for (const line of lines) {
    for (const def of STAGE_DEFS) {
      if (def.trigger.test(line)) {
        reached.add(def.id);
        latestReached = def.id;
      }
    }
  }

  // Also mark downloading active as soon as finding is done
  if (reached.has('finding') && !reached.has('extracting')) {
    reached.add('downloading');
    latestReached = 'downloading';
  }

  return STAGE_DEFS.map((def) => {
    const isDone = reached.has(def.id) && def.id !== latestReached;
    const isActive = def.id === latestReached && !isError && !reached.has('done');
    const isComplete = def.id === latestReached && (reached.has('done') || def.id === 'done');

    let status: StageStatus = 'pending';
    if (isDone || isComplete) status = 'done';
    else if (isActive) status = 'active';

    // Pull detail text for the current active stage
    let detail: string | undefined;
    if (def.id === 'downloading' && isActive) {
      const lastDownload = [...lines].reverse().find((l) => /mtp: downloading shard/.test(l));
      if (lastDownload) detail = lastDownload.replace('mtp: ', '');
    }
    if (def.id === 'extracting' && status === 'done') {
      const line =
        lines.find((l) => /mtp: extracted/.test(l)) ?? lines.find((l) => /mtp: saved/.test(l));
      if (line) detail = line.replace('mtp: ', '');
    }
    if (def.id === 'quantizing' && status === 'done') {
      const line = lines.find((l) => /mtp: quantized/.test(l));
      if (line) detail = line.replace('mtp: ', '');
    }
    if (def.id === 'uploading' && isActive) {
      const prog = [...lines].reverse().find((l) => /mtp: uploading \d+%/.test(l));
      if (prog) {
        const m = prog.match(/uploading (\d+)%\s*\(([^)]+)\)/);
        if (m) detail = `${m[1]}% · ${m[2]}`;
      }
    }

    return { stage: def.id, label: def.label, status, detail };
  });
}

// ---------------------------------------------------------------------------
// Styled components
// ---------------------------------------------------------------------------

const Panel = styled.div`
  display: flex;
  flex-direction: column;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  overflow: hidden;
`;

const PanelHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border-bottom: 1px solid ${({ theme }) => theme.colors.borderLight};
`;

const PanelTitle = styled.h3`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
  text-transform: uppercase;
  letter-spacing: 0.04em;
`;

const StatusBadge = styled.span<{ $phase: string }>`
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  background: ${({ $phase, theme }) => {
    if ($phase === 'done') return theme.colors.accentBg;
    if ($phase === 'error') return theme.colors.errorBg;
    return theme.colors.goldBg;
  }};
  color: ${({ $phase, theme }) => {
    if ($phase === 'done') return theme.colors.accent;
    if ($phase === 'error') return theme.colors.errorText;
    return theme.colors.gold;
  }};
`;

const StageList = styled.ol`
  list-style: none;
  padding: ${({ theme }) => theme.spacing.md};
  display: flex;
  flex-direction: column;
  gap: 2px;
`;

const StageItem = styled.li<{ $status: StageStatus }>`
  display: flex;
  align-items: flex-start;
  gap: ${({ theme }) => theme.spacing.sm};
  padding: 6px ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ $status, theme }) =>
    $status === 'active'
      ? theme.colors.goldBg
      : $status === 'done'
        ? 'transparent'
        : 'transparent'};
  opacity: ${({ $status }) => ($status === 'pending' ? 0.35 : 1)};
  transition: all 0.2s;
`;

const StageIcon = styled.span<{ $status: StageStatus }>`
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  flex-shrink: 0;
  margin-top: 1px;
  background: ${({ $status, theme }) =>
    $status === 'done'
      ? theme.colors.accent
      : $status === 'active'
        ? theme.colors.goldBg
        : theme.colors.surfaceHover};
  color: ${({ $status, theme }) =>
    $status === 'done'
      ? '#fff'
      : $status === 'active'
        ? theme.colors.gold
        : theme.colors.textMuted};
  border: 1px solid
    ${({ $status, theme }) =>
      $status === 'done'
        ? theme.colors.accent
        : $status === 'active'
          ? theme.colors.borderStrong
          : theme.colors.borderLight};
`;

const SpinIcon = styled.span`
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 1px;
`;

const StageText = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
`;

const StageName = styled.span<{ $status: StageStatus }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-weight: ${({ $status }) => ($status === 'active' ? '600' : '400')};
  color: ${({ $status, theme }) =>
    $status === 'active'
      ? theme.colors.gold
      : $status === 'done'
        ? theme.colors.text
        : theme.colors.textMuted};
`;

const StageDetail = styled.span`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textMuted};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const Divider = styled.div`
  height: 1px;
  background: ${({ theme }) => theme.colors.borderLight};
  margin: 0 ${({ theme }) => theme.spacing.md};
`;

const RawLog = styled.pre`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  line-height: 1.6;
  color: ${({ theme }) => theme.colors.textMuted};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  margin: 0;
  max-height: 120px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
`;

const DownloadHint = styled.p`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  padding: 0 ${({ theme }) => theme.spacing.md} ${({ theme }) => theme.spacing.sm};
  font-style: italic;
`;

const UploadHint = styled.p`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  padding: 0 ${({ theme }) => theme.spacing.md} ${({ theme }) => theme.spacing.sm};
  font-style: italic;
`;

const ErrorBlock = styled.div`
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.errorOnSurface};
  background: ${({ theme }) => theme.colors.errorBg};
  font-family: ${({ theme }) => theme.fonts.mono};
  border-top: 1px solid ${({ theme }) => theme.colors.errorBg};
`;

/**
 * Catalog registration (Gemma 4 assistant path) is synchronous — there are no
 * MTP extraction stages. Render a single completion item driven by the store's
 * "registered ..." log line instead of the MTP stage list, so the structured
 * view reflects completion rather than sitting in "pending".
 */
function registrationStages(
  lines: string[],
): { stage: Stage; label: string; status: StageStatus; detail?: string }[] {
  const detail =
    lines.find((l) => l.startsWith('assistant: ')) ??
    lines.find((l) => l.startsWith('written to '));
  return [{ stage: 'done', label: 'Registered in catalog', status: 'done', detail }];
}

function statusLabel(phase: string): string {
  if (phase === 'publishing') return 'Publishing…';
  if (phase === 'done') return 'Done';
  if (phase === 'error') return 'Failed';
  return 'Log';
}

/**
 * Structured publish progress panel. Parses the raw log stream into
 * discrete stages and renders each with a spinner or checkmark.
 */
export const PublishLog = ({ phase, lines, errorMessage, className }: PublishLogProps) => {
  const rawRef = useRef<HTMLPreElement>(null);
  const isError = phase === 'error';
  // The assistant registration flow is synchronous and emits "registered ..."
  // rather than MTP progress lines; show a registration completion instead of
  // the (irrelevant) MTP stage list.
  const isRegistration = lines.some((l) => l.startsWith('registered '));
  const stages = isRegistration ? registrationStages(lines) : deriveStages(lines, isError);
  const activeStage = stages.find((s) => s.status === 'active');
  const isDownloading = activeStage?.stage === 'downloading';
  const isUploading = activeStage?.stage === 'uploading';

  useEffect(() => {
    if (rawRef.current) rawRef.current.scrollTop = rawRef.current.scrollHeight;
  }, [lines]);

  return (
    <Panel className={className} role="region" aria-label="Publish log">
      <PanelHeader>
        <PanelTitle>Publish log</PanelTitle>
        <StatusBadge $phase={phase}>{statusLabel(phase)}</StatusBadge>
      </PanelHeader>

      <StageList>
        {stages.map(({ stage, label, status, detail }) => (
          <StageItem key={stage} $status={status}>
            {status === 'active' ? (
              <SpinIcon>
                <Spinner size={16} />
              </SpinIcon>
            ) : (
              <StageIcon $status={status}>{status === 'done' ? '✓' : '·'}</StageIcon>
            )}
            <StageText>
              <StageName $status={status}>{label}</StageName>
              {detail && <StageDetail>{detail}</StageDetail>}
            </StageText>
          </StageItem>
        ))}
      </StageList>

      {isDownloading && (
        <DownloadHint>
          Downloading BF16 shards from HuggingFace — this can take several minutes for large models.
        </DownloadHint>
      )}

      {isUploading && (
        <UploadHint>
          Uploading to HuggingFace — large models can take 20–40 minutes on a typical connection.
        </UploadHint>
      )}

      {isError && errorMessage && <ErrorBlock>Error: {errorMessage}</ErrorBlock>}

      {lines.length > 0 && (
        <>
          <Divider />
          <RawLog ref={rawRef}>{lines.join('\n')}</RawLog>
        </>
      )}
    </Panel>
  );
};
