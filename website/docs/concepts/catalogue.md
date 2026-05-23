---
title: Catalogue
---

`models.yaml` is the source of truth for publishable artifacts.

Each entry has:

- `key`: stable selector used by the CLI and GitHub Actions
- `source_model`: upstream Hugging Face model ID
- `quant`: target LARQL quantization
- `tier`: `smoke` for small scheduled targets or `moe` for manual large targets
- `slices`: LARQL slice presets to publish
- `output_name`: local vindex output directory name
- `hf_repo`: target Hugging Face repository

The CLI validates the catalogue before any publish command is planned.

```bash
skulk-vindex manifest validate
skulk-vindex manifest get --key gemma-3-4b-full-q4-k
```
