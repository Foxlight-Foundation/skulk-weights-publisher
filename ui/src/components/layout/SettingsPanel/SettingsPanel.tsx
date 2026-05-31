import { useEffect, useState } from 'react';
import styled, { keyframes } from 'styled-components';
import { useConfigStore } from '@/stores/config.store';
import { fetchConfig } from '@/api/client';
import { Button } from '@/components/common/Button/Button';
import { Field } from '@/components/common/Field/Field';
import type { SettingsPanelProps } from './SettingsPanel.types';

const fadeIn = keyframes`from { opacity: 0 } to { opacity: 1 }`;
const slideIn = keyframes`from { transform: translateX(100%) } to { transform: translateX(0) }`;

const Backdrop = styled.div`
  position: fixed;
  inset: 0;
  background: ${({ theme }) => theme.colors.overlay};
  z-index: 200;
  animation: ${fadeIn} 0.15s ease;
`;

const Drawer = styled.div`
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  width: 360px;
  background: ${({ theme }) => theme.colors.surface};
  border-left: 1px solid ${({ theme }) => theme.colors.border};
  box-shadow: -4px 0 24px ${({ theme }) => theme.colors.shadowStrong};
  z-index: 201;
  display: flex;
  flex-direction: column;
  animation: ${slideIn} 0.2s ease;
`;

const DrawerHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => theme.spacing.lg};
  border-bottom: 1px solid ${({ theme }) => theme.colors.borderLight};
`;

const DrawerTitle = styled.h2`
  font-size: ${({ theme }) => theme.fontSizes.lg};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text};
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: 18px;
  padding: 4px;
  border-radius: ${({ theme }) => theme.radii.sm};
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
    background: ${({ theme }) => theme.colors.surfaceHover};
  }
`;

const DrawerBody = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: ${({ theme }) => theme.spacing.lg};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
`;

const Section = styled.fieldset`
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing.md};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
`;

const SectionLegend = styled.legend`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0 4px;
`;

const FormRow = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.xs};
`;

const Label = styled.label`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-weight: 500;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const Hint = styled.p`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const InputRow = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const SaveStatus = styled.p<{ $ok: boolean }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ $ok, theme }) => ($ok ? theme.colors.healthy : theme.colors.error)};
`;

/**
 * Slide-in settings drawer for configuring the HuggingFace token.
 * Token is persisted to ~/.config/skulk-weights/.env on the server.
 */
export const SettingsPanel = ({ open, onClose }: SettingsPanelProps) => {
  const saveToken = useConfigStore((s) => s.saveToken);
  const [tokenDraft, setTokenDraft] = useState('');
  const [maskedToken, setMaskedToken] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'ok' | 'error'>('idle');

  useEffect(() => {
    if (!open) return;
    fetchConfig()
      .then((c) => setMaskedToken(c.hf_token_masked))
      .catch(() => {});
  }, [open]);

  const handleSave = async () => {
    try {
      await saveToken(tokenDraft);
      setTokenDraft('');
      setSaveStatus('ok');
      const updated = await fetchConfig();
      setMaskedToken(updated.hf_token_masked);
    } catch {
      setSaveStatus('error');
    }
  };

  if (!open) return null;

  return (
    <>
      <Backdrop onClick={onClose} />
      <Drawer role="dialog" aria-label="Settings">
        <DrawerHeader>
          <DrawerTitle>Settings</DrawerTitle>
          <CloseButton onClick={onClose} aria-label="Close settings">
            ✕
          </CloseButton>
        </DrawerHeader>
        <DrawerBody>
          <Section>
            <SectionLegend>HuggingFace</SectionLegend>
            <FormRow>
              <Label htmlFor="hf-token">API Token</Label>
              {maskedToken && <Hint>Current: {maskedToken}</Hint>}
              <InputRow>
                <Field
                  id="hf-token"
                  type="password"
                  placeholder="hf_..."
                  value={tokenDraft}
                  onChange={(e) => {
                    setTokenDraft(e.target.value);
                    setSaveStatus('idle');
                  }}
                  style={{ flex: 1 }}
                />
                <Button variant="outline" size="md" onClick={() => void handleSave()}>
                  Save
                </Button>
              </InputRow>
              {saveStatus === 'ok' && <SaveStatus $ok>Token saved.</SaveStatus>}
              {saveStatus === 'error' && <SaveStatus $ok={false}>Failed to save token.</SaveStatus>}
              <Hint>Stored in ~/.config/skulk-weights/.env</Hint>
            </FormRow>
          </Section>
        </DrawerBody>
      </Drawer>
    </>
  );
};
