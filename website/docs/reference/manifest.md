---
title: Manifest Reference
---

A manifest source file contains a top-level `models` list. Each item describes
one publishable vindex and the slice shape Skulk can later place on runtime
hardware.

Manifests are source files. The catalog is the merged view built from the
packaged Foxlight manifest plus any operator manifests listed in
`skulk-weights.yaml`.

## Example Source Entry

```yaml
models:
  - key: gemma-3-4b-full-q4-k
    source_model: google/gemma-3-4b-it
    quant: q4k
    tier: smoke
    slices:
      - full
    output_name: gemma-3-4b-it-full-q4-k.vindex
    hf_repo: FoxlightAI/gemma-3-4b-it-full-q4-k-vindex
    hf_collection: FoxlightAI/vindexes-6a124406dd5fb439c431b051
```

If that source is loaded under the `foxlight` namespace, the effective
catalog key is:

```text
foxlight/gemma-3-4b-full-q4-k
```

If an operator source is loaded under `my-org`, the same short key becomes:

```text
my-org/gemma-3-4b-full-q4-k
```

## Field Reference

### Vindex fields

| Field | Meaning |
|---|---|
| `key` | Stable short selector before the catalog namespace is added |
| `source_model` | Hugging Face model ID passed to `larql extract` |
| `quant` | Quantization passed to LARQL |
| `tier` | Publication group, currently `smoke` or `moe` |
| `slices` | LARQL slice mode, currently `full` or `expert-server`; this is the runtime placement shape |
| `output_name` | Local vindex directory basename under scratch storage |
| `hf_repo` | Hugging Face repository passed to `larql publish` |
| `hf_collection` | Optional Hugging Face collection slug updated after publish succeeds |

### MTP sidecar fields

These three fields must all be set together or all omitted.

| Field | Meaning |
|---|---|
| `mtp_source_repo` | Hugging Face model ID of the original BF16 checkpoint that contains `mtp.*` tensor keys |
| `mtp_sidecar_repo` | Hugging Face repository where `mtp.safetensors` will be uploaded |
| `mtp_quant` | Quantization scheme for the extracted MTP weights, currently `q4k` or `q8k` |

The `mtp_source_repo` is often different from `source_model`. `source_model` is typically an
mlx-converted or community checkpoint; `mtp_source_repo` must be the original PyTorch BF16 release
because mlx-lm's `sanitize()` strips `mtp.*` keys during conversion.

Example entry with MTP sidecar:

```yaml
models:
  - key: qwen3-6b-full-q4-k
    source_model: acme/qwen3-6b-mlx-q4k
    quant: q4k
    tier: smoke
    slices:
      - full
    output_name: qwen3-6b-full-q4-k.vindex
    hf_repo: acme/qwen3-6b-full-q4-k-vindex
    mtp_source_repo: Qwen/Qwen3-6B
    mtp_sidecar_repo: acme/qwen3-6b-mtp-q4k
    mtp_quant: q4k
```

## Validation Rules

- `key` must be lowercase kebab-case and unique within its source
- effective catalog keys must be unique after namespaces are applied
- `source_model` must look like `owner/name`
- `quant` currently supports `q4k`
- `tier` must be `smoke` or `moe`
- `slices` must be non-empty
- `full` cannot be combined with other slices
- `output_name` must be a `.vindex` basename and unique in the merged catalog
- `hf_repo` must look like `owner/name` and be unique in the merged catalog
- operator `hf_repo` owners must match the source `hf_owner` in `skulk-weights.yaml`
- `hf_collection` must look like `owner/slug`
- operator `hf_collection` owners must match the source `hf_owner` in `skulk-weights.yaml`
- `mtp_source_repo` and `mtp_sidecar_repo` must look like `owner/name`
- `mtp_quant` currently supports `q4k` and `q8k`
- `mtp_source_repo`, `mtp_sidecar_repo`, and `mtp_quant` must all be set together or all omitted

## Generated Commands

This entry:

```yaml
key: gemma-3-4b-full-q4-k
source_model: google/gemma-3-4b-it
quant: q4k
slices:
  - full
output_name: gemma-3-4b-it-full-q4-k.vindex
hf_repo: FoxlightAI/gemma-3-4b-it-full-q4-k-vindex
hf_collection: FoxlightAI/vindexes-6a124406dd5fb439c431b051
```

produces this command shape:

```bash
larql extract google/gemma-3-4b-it \
  -o .scratch/gemma-3-4b-it-full-q4-k.vindex \
  --quant q4k

larql publish .scratch/gemma-3-4b-it-full-q4-k.vindex \
  --repo FoxlightAI/gemma-3-4b-it-full-q4-k-vindex \
  --slices none
```

After the LARQL publish command succeeds, the publisher adds the repository to
the configured collection using the Hugging Face Hub API.

For `slices: [full]`, the publisher sends `--slices none` because LARQL treats
that as the complete vindex publish path.

For `slices: [expert-server]`, the output is meant for MoE expert weight
serving from CPU/high-memory LARQL servers instead of forcing those weights into
GPU memory.
