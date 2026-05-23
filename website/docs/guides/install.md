---
title: Install
---

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For CI or a publishing runner:

```bash
python3 -m pip install -e .
```

The repository still includes compatibility wrappers:

```bash
scripts/doctor.sh
scripts/manifest.py validate
scripts/publish-vindex.sh --model gemma-3-4b-full-q4-k --dry-run
```

Prefer `skulk-vindex` for new automation.
