---
title: Runner Setup
---

Real publication needs a machine that can run LARQL, write large temporary
vindex directories, and upload to Hugging Face. In GitHub Actions, that machine
is a self-hosted runner.

The runner is separate from ordinary PR validation. PR validation can build docs,
validate the merged catalogue, and dry-run commands on hosted runners. Real
publication uses the self-hosted runner because extraction is disk-heavy and
credentialed. The runner is also separate from the eventual Skulk runtime
placement: it creates the published vindex that later lets CPU/high-memory
LARQL servers host weight-heavy FFN or expert pieces while GPU nodes handle the
inference hot path.

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

GPU hardware is not a runner requirement for publication. The important
capacity is disk for extraction and network for upload.

Set `SKULK_VINDEX_SCRATCH` if the scratch directory should live outside the
checkout.

## Minimum First Target

Start by publishing one `smoke` entry. Provision at least 200 GB of fast scratch
storage for that first path.

MoE and expert-server entries require substantially more scratch space. Treat
them as explicit operator actions, not background validation.

## Runner Preflight

Run this on the runner before dispatching a real publish:

```bash
python3 -m pip install -e .
skulk-vindex doctor --publish
skulk-vindex publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```
