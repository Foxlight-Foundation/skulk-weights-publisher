---
title: Manifest Reference
---

`models.yaml` contains a top-level `models` list. Each item describes one
publishable vindex and the slice shape Skulk can later place on runtime
hardware.

```yaml
models:
  - key: gemma-3-4b-full-q4-k
    source_model: google/gemma-3-4b-it
    quant: q4k
    tier: smoke
    slices:
      - full
    output_name: gemma-3-4b-it-full-q4-k.vindex
    hf_repo: skulk/gemma-3-4b-it-full-q4-k-vindex
```

## Field Reference

| Field | Meaning |
|---|---|
| `key` | Stable CLI and workflow selector for this vindex |
| `source_model` | Hugging Face model ID passed to `larql extract` |
| `quant` | Quantization passed to LARQL |
| `tier` | Publication group, currently `smoke` or `moe` |
| `slices` | LARQL slice mode, currently `full` or `expert-server`; this is the runtime placement shape |
| `output_name` | Local vindex directory basename under scratch storage |
| `hf_repo` | Hugging Face repository passed to `larql publish` |

## Validation Rules

- `key` must be lowercase kebab-case and unique
- `source_model` must look like `owner/name`
- `quant` currently supports `q4k`
- `tier` must be `smoke` or `moe`
- `slices` must be non-empty
- `full` cannot be combined with other slices
- `output_name` must be a `.vindex` basename
- `hf_repo` must look like `owner/name` and be unique

## Generated Commands

This entry:

```yaml
key: gemma-3-4b-full-q4-k
source_model: google/gemma-3-4b-it
quant: q4k
slices:
  - full
output_name: gemma-3-4b-it-full-q4-k.vindex
hf_repo: skulk/gemma-3-4b-it-full-q4-k-vindex
```

produces this command shape:

```bash
larql extract google/gemma-3-4b-it \
  -o .scratch/gemma-3-4b-it-full-q4-k.vindex \
  --quant q4k

larql publish .scratch/gemma-3-4b-it-full-q4-k.vindex \
  --repo skulk/gemma-3-4b-it-full-q4-k-vindex \
  --slices none
```

For `slices: [full]`, the publisher sends `--slices none` because LARQL treats
that as the complete vindex publish path.

For `slices: [expert-server]`, the output is meant for MoE expert weight
serving from CPU/high-memory LARQL servers instead of forcing those weights into
GPU memory.
