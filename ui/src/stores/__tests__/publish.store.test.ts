import { beforeEach, describe, expect, it, vi } from 'vitest';
import { usePublishStore } from '../publish.store';
import * as client from '@/api/client';

vi.mock('@/api/client');

const mockDetectModel = vi.mocked(client.detectModel);
const mockStartPublish = vi.mocked(client.startPublish);
const mockOpenLogStream = vi.mocked(client.openLogStream);
const mockRegisterCatalog = vi.mocked(client.registerCatalog);

beforeEach(() => {
  usePublishStore.setState({
    phase: 'idle',
    url: '',
    detection: null,
    logLines: [],
    errorMessage: null,
  });
  vi.clearAllMocks();
});

describe('publish.store', () => {
  it('setUrl updates the url field', () => {
    usePublishStore.getState().setUrl('mlx-community/test');
    expect(usePublishStore.getState().url).toBe('mlx-community/test');
  });

  it('reset clears all state', () => {
    usePublishStore.setState({ phase: 'done', url: 'something', errorMessage: 'err' });
    usePublishStore.getState().reset();
    const s = usePublishStore.getState();
    expect(s.phase).toBe('idle');
    expect(s.url).toBe('');
    expect(s.errorMessage).toBeNull();
  });

  describe('detect', () => {
    it('transitions to detected on success', async () => {
      const result = {
        model_id: 'mlx-community/test',
        base_model: 'owner/base',
        quant: 'q4k',
        tier: 'smoke',
        mtp_key_count: 2,
        mtp_keys: ['mtp.0.key'],
        sidecar_repo: 'FoxlightAI/base-mtp',
        can_publish: true,
        assistant_model_repo: null,
        can_publish_assistant: false,
      };
      mockDetectModel.mockResolvedValue(result);
      usePublishStore.setState({ url: 'mlx-community/test' });

      await usePublishStore.getState().detect();

      expect(usePublishStore.getState().phase).toBe('detected');
      expect(usePublishStore.getState().detection).toEqual(result);
    });

    it('transitions to error when detection fails', async () => {
      mockDetectModel.mockRejectedValue(new Error('not found'));
      usePublishStore.setState({ url: 'bad/model' });

      await usePublishStore.getState().detect();

      expect(usePublishStore.getState().phase).toBe('error');
      expect(usePublishStore.getState().errorMessage).toBe('not found');
    });
  });

  describe('publish', () => {
    it('does nothing if detection has no base_model and no assistant', async () => {
      usePublishStore.setState({
        phase: 'detected',
        detection: {
          model_id: 'x',
          base_model: null,
          quant: 'q4k',
          tier: 'smoke',
          mtp_key_count: 0,
          mtp_keys: [],
          sidecar_repo: null,
          can_publish: false,
          assistant_model_repo: null,
          can_publish_assistant: false,
        },
      });

      await usePublishStore.getState().publish();

      expect(usePublishStore.getState().phase).toBe('detected');
      expect(mockStartPublish).not.toHaveBeenCalled();
    });

    it('transitions to error when startPublish fails', async () => {
      usePublishStore.setState({
        phase: 'detected',
        detection: {
          model_id: 'x',
          base_model: 'owner/base',
          quant: 'q4k',
          tier: 'smoke',
          mtp_key_count: 2,
          mtp_keys: ['k'],
          sidecar_repo: 'FoxlightAI/base-mtp',
          can_publish: true,
          assistant_model_repo: null,
          can_publish_assistant: false,
        },
      });
      mockStartPublish.mockRejectedValue(new Error('server error'));

      await usePublishStore.getState().publish();

      expect(usePublishStore.getState().phase).toBe('error');
      expect(usePublishStore.getState().errorMessage).toBe('server error');
    });

    it('opens SSE stream and appends log lines', async () => {
      usePublishStore.setState({
        phase: 'detected',
        detection: {
          model_id: 'x',
          base_model: 'owner/base',
          quant: 'q4k',
          tier: 'smoke',
          mtp_key_count: 2,
          mtp_keys: ['k'],
          sidecar_repo: 'FoxlightAI/base-mtp',
          can_publish: true,
          assistant_model_repo: null,
          can_publish_assistant: false,
        },
      });
      mockStartPublish.mockResolvedValue({ job_id: 'job-1' });

      let capturedOnMessage: ((e: MessageEvent<string>) => void) | null = null;
      const fakeEs = {
        onerror: null as (() => void) | null,
        close: vi.fn(),
        set onmessage(fn: ((e: MessageEvent<string>) => void) | null) {
          capturedOnMessage = fn;
        },
        get onmessage() {
          return capturedOnMessage;
        },
      };
      mockOpenLogStream.mockReturnValue(fakeEs as unknown as EventSource);

      await usePublishStore.getState().publish();
      expect(usePublishStore.getState().phase).toBe('publishing');

      type Handler = (e: MessageEvent<string>) => void;
      // Simulate a log line arriving
      (fakeEs.onmessage as Handler | null)?.({
        data: 'mtp: downloaded shard',
      } as MessageEvent<string>);
      expect(usePublishStore.getState().logLines).toContain('mtp: downloaded shard');

      // Simulate done
      (fakeEs.onmessage as Handler | null)?.({ data: '[done]' } as MessageEvent<string>);
      expect(usePublishStore.getState().phase).toBe('done');
      expect(fakeEs.close).toHaveBeenCalled();
    });

    it('registers in catalog (no SSE) for a Gemma 4 companion model', async () => {
      usePublishStore.setState({
        url: 'mlx-community/gemma-4-27b-it-4bit',
        phase: 'detected',
        detection: {
          model_id: 'mlx-community/gemma-4-27b-it-4bit',
          base_model: 'google/gemma-4-27b-it',
          quant: 'q4k',
          tier: 'moe',
          mtp_key_count: 0,
          mtp_keys: [],
          sidecar_repo: null,
          can_publish: false,
          assistant_model_repo: 'google/gemma-4-27b-it-assistant',
          can_publish_assistant: true,
        },
      });
      mockRegisterCatalog.mockResolvedValue({
        ok: true,
        key: 'foxlight/gemma-4-27b-full-q4-k',
        assistant_model_repo: 'google/gemma-4-27b-it-assistant',
        catalog_path: '/pkg/catalogues/foxlight.yaml',
        entry_block: '\n  - key: gemma-4-27b-full-q4-k\n',
      });

      await usePublishStore.getState().publish();

      expect(mockRegisterCatalog).toHaveBeenCalledWith('mlx-community/gemma-4-27b-it-4bit');
      expect(mockStartPublish).not.toHaveBeenCalled();
      expect(usePublishStore.getState().phase).toBe('done');
      expect(usePublishStore.getState().logLines.join('\n')).toContain(
        'registered foxlight/gemma-4-27b-full-q4-k',
      );
    });

    it('transitions to error when catalog registration fails', async () => {
      usePublishStore.setState({
        url: 'mlx-community/gemma-4-27b-it-4bit',
        phase: 'detected',
        detection: {
          model_id: 'mlx-community/gemma-4-27b-it-4bit',
          base_model: 'google/gemma-4-27b-it',
          quant: 'q4k',
          tier: 'moe',
          mtp_key_count: 0,
          mtp_keys: [],
          sidecar_repo: null,
          can_publish: false,
          assistant_model_repo: 'google/gemma-4-27b-it-assistant',
          can_publish_assistant: true,
        },
      });
      mockRegisterCatalog.mockRejectedValue(new Error('already exists in the catalog'));

      await usePublishStore.getState().publish();

      expect(usePublishStore.getState().phase).toBe('error');
      expect(usePublishStore.getState().errorMessage).toBe('already exists in the catalog');
    });
  });
});
