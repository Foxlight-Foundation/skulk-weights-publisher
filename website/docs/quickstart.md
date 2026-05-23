---
title: Quickstart
---

This guide gets you from a clean checkout to a safe dry-run. A dry-run prints
the LARQL commands that would run, but it does not extract model weights and
does not upload anything.

## Requirements

- Python 3.11 or newer
- Node.js 20 or newer if you want to build the docs site
- `larql` only for real publication
- a Hugging Face token only for real publication

## Install The CLI

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Validate The Catalogue

```bash
skulk-vindex manifest validate
skulk-vindex manifest list --tier smoke
```

## Run The Doctor

```bash
skulk-vindex doctor
```

Use the stricter publishing checks only on a machine that is supposed to publish:

```bash
skulk-vindex doctor --publish
```

## Dry-Run A Publish

```bash
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

You should see the exact `larql extract` and `larql publish` commands. If that
output is not what you expect, stop there and fix the catalogue entry before
running a real publish.
