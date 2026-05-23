---
title: Manifest Reference
---

`models.yaml` contains a top-level `models` list.

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

Validation rules:

- `key` must be lowercase kebab-case and unique
- `source_model` must look like `owner/name`
- `quant` currently supports `q4k`
- `tier` must be `smoke` or `moe`
- `slices` must be non-empty
- `full` cannot be combined with other slices
- `output_name` must be a `.vindex` basename
- `hf_repo` must look like `owner/name` and be unique
