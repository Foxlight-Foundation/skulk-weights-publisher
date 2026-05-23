# Operator Setup

This repository is designed for a self-hosted runner because vindex extraction
is disk-heavy and can exceed commodity hosted-runner limits.

## GitHub Actions Runner

Register a Linux self-hosted runner with these labels:

```text
self-hosted
linux
larql
vindex
```

Phase 1 smoke-tier capacity target:

- at least 200 GB scratch space
- stable network path to HuggingFace
- `larql` available on `PATH`

MoE-tier entries are manual by default and require substantially more scratch
space than the smoke tier. Do not enable broad MoE publication until the runner
has enough disk for the selected model family.

Set `SKULK_VINDEX_SCRATCH` on the runner if the scratch directory should live
outside the checkout.

## Secrets

Configure this GitHub Actions secret:

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace token with write access to the target `skulk/` repos |

Do not commit tokens to this repository.

## Validation

Before enabling the scheduled workflow, run:

```bash
python3 -m pip install -r requirements.txt
scripts/doctor.sh
python3 scripts/manifest.py validate
scripts/publish-vindex.sh --model gemma-3-4b-full-q4-k --dry-run
```

Before real publication, run the stricter preflight:

```bash
scripts/doctor.sh --publish
scripts/publish-vindex.sh --model gemma-3-4b-full-q4-k --dry-run
```

Then use manual workflow dispatch for a single smoke-tier entry before enabling
the full weekly run.

## Workflow Dispatch

- `model=all`, `tier=smoke`, `dry_run=false` publishes the scheduled smoke set.
- `model=<manifest-key>` publishes one entry and ignores `tier`.
- `dry_run=true` runs the same manifest resolution path but only prints LARQL
  commands.
- `tier=moe` is for explicit large-model publication after capacity has been
  verified.
