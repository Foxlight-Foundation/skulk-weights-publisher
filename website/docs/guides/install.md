---
title: Install
---

Install the publisher when you want to inspect the catalogue, dry-run a publish,
or run the publishing workflow on a configured runner.

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
skulk-vindex manifest validate
```

Then run one dry-run:

```bash
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

## Compatibility Wrappers

The repository includes older script names for automation that has not moved to
the package CLI yet:

```bash
scripts/doctor.sh
scripts/manifest.py validate
scripts/publish-vindex.sh --model gemma-3-4b-full-q4-k --dry-run
```

New documentation and new automation should use `skulk-vindex`.
