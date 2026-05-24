---
title: Install
---

Install the publisher when you want to inspect the catalog, dry-run a publish,
or run the publishing workflow on a configured runner. The CLI exists to make
vindex publication repeatable before Skulk relies on those vindexes for
GPU/CPU runtime placement.

## Local Development Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

This installs the CLI plus development tools used by CI: tests, linting, and
type checking.

## Runner Install

```bash
python3 -m pip install -e .
```

Use the runner install on a self-hosted publishing runner that only needs the
product CLI.

## Check The Install

```bash
skulk-vindex doctor
skulk-vindex catalog validate
```

Then run one dry-run:

```bash
skulk-vindex publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

To add your own operator catalog, initialize a config after install:

```bash
skulk-vindex catalog init
```

## Compatibility Wrappers

The repository includes older script names for automation that has not moved to
the package CLI yet:

```bash
scripts/doctor.sh
scripts/manifest.py validate
scripts/publish-vindex.sh --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

New documentation and new automation should use `skulk-vindex`.
