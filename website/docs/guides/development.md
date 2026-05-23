---
title: Development
---

Use this workflow when changing the publisher code, catalogue validation,
publishing workflow, or documentation.

The development goal is simple: every change should preserve the ability to
explain a vindex publish plan before any real extraction or upload happens.

## Python Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Python Checks

Run the same checks used by pull-request validation:

```bash
ruff check
basedpyright
pytest
bash -n scripts/doctor.sh scripts/publish-vindex.sh
```

Then validate the catalogue and every dry-run path:

```bash
skulk-vindex manifest validate
skulk-vindex manifest list --tier all | while IFS= read -r key; do
  [ -n "$key" ] || continue
  skulk-vindex publish --model "$key" --dry-run >/dev/null
done
```

That loop proves every catalogue entry can produce a dry-run plan.

## Documentation Setup

```bash
cd website
npm ci
npm run build
```

Use `npm run start` when editing docs locally.

Pushes to feature branches publish preview docs under:

```text
https://foxlight-foundation.github.io/skulk-vindex-publisher/previews/<branch>/
```

Pushes to `main` publish the production docs root.

## Compatibility Wrappers

The `scripts/` commands are retained for existing automation. They call the
same Python package used by `skulk-vindex`, so behavior should be tested through
both the package CLI and at least one wrapper dry-run before changing release
automation.
