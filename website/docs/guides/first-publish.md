---
title: First Publish
---

Do the first real publish with one smoke-tier entry. Do not start with an MoE
entry.

## Preflight

```bash
skulk-vindex manifest validate
skulk-vindex doctor --publish
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

## Real Publish

```bash
export HF_TOKEN=...
export SKULK_VINDEX_SCRATCH=/fast/scratch/skulk-vindexes
skulk-vindex publish --model gemma-3-4b-full-q4-k
```

The command refuses to overwrite an existing output path. Use `--force` only
when you intentionally want to replace a local extraction output.
