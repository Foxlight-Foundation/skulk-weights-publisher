---
title: Runner Setup
---

Real publication is designed for a self-hosted runner.

Required labels:

```text
self-hosted
linux
larql
vindex
```

Runner requirements:

- `larql` on `PATH`
- Python 3.11 or newer
- stable network access to Hugging Face
- `HF_TOKEN` configured as a GitHub Actions secret
- fast scratch storage

Set `SKULK_VINDEX_SCRATCH` if the scratch directory should live outside the
checkout.
