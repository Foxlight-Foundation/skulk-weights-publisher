---
title: Install
---

Install the publisher when you want to inspect the catalog, dry-run a publish,
or run the publishing workflow on a configured runner. The CLI exists to make
vindex publication repeatable before Skulk relies on those vindexes for
GPU/CPU runtime placement.

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). Commands run
through `uv run`, which resolves the environment on demand.

## Local Development Install

```bash
uv sync --extra dev
```

This installs the CLI plus development tools used by CI: tests, linting, and
type checking. Add `--extra mtp` for MTP sidecar extraction (`safetensors`,
`numpy`, `mlx` — Apple Silicon only). Add `--extra ui` for the local GUI (it
already includes the mtp deps). Vision sidecars and vindex publishing need no
extra — `huggingface_hub` is a base dependency.

## Runner Install

```bash
uv sync
```

Use the bare sync on a self-hosted publishing runner that only needs the product
CLI. A Linux runner can publish vindexes; real MTP and vision publishing also
needs `--extra mtp` and a macOS Apple Silicon host (see
[Runner Setup](runner-setup.md)).

## Check The Install

```bash
uv run skulk-weights doctor
uv run skulk-weights catalog validate
```

Then run one dry-run:

```bash
uv run skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

To add your own operator catalog, initialize a config after install:

```bash
uv run skulk-weights catalog init
```

## Compatibility Wrappers

The repository keeps a few wrapper scripts under `scripts/` for automation that
has not moved to the package CLI yet:

```bash
scripts/doctor.sh
scripts/manifest.py validate
scripts/publish-vindex.sh --model foxlight/gemma-3-4b-full-q4-k --dry-run
scripts/publish-weights.sh --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

New documentation and new automation should use `skulk-weights`.
