---
title: Development
---

Use this workflow when changing the publisher code, catalog validation,
publishing workflow, or documentation.

The development goal is simple: every change should preserve the ability to
explain a vindex publish plan before any real extraction or upload happens.
That plan is not only a file-transfer plan; it is how the project records which
published vindex Skulk can later place on GPU inference nodes or CPU/high-memory
weight-serving nodes.

## Python Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

Add `--extra ui` and `--extra mtp` when working on the GUI or the MTP/vision
extraction paths.

## Python Checks

Run the same checks used by pull-request validation:

```bash
uv run ruff check src/ tests/
uv run --extra dev basedpyright
uv run pytest tests/
bash -n scripts/doctor.sh scripts/publish-vindex.sh scripts/publish-weights.sh
```

The full test suite, including the GUI and MTP paths, is:

```bash
uv run --extra ui --extra mtp --extra dev pytest tests/
```

Then validate the catalog and every dry-run path:

```bash
uv run skulk-weights catalog validate
uv run skulk-weights catalog list --tier all | while IFS= read -r key; do
  [ -n "$key" ] || continue
  uv run skulk-weights publish --model "$key" --dry-run >/dev/null
done
```

That loop proves every effective catalog entry can produce a dry-run plan.
Use `--config skulk-weights.yaml` in both commands when testing operator
catalog sources.

## Documentation Setup

The documentation site (this Docusaurus site under `website/`) uses npm:

```bash
cd website
npm ci
npm run build
```

Use `npm run start` when editing docs locally.

Pushes to feature branches publish preview docs under:

```text
https://foxlight-foundation.github.io/skulk-weights-publisher/previews/<branch>/
```

Pushes to `main` publish the production docs root.

## Compatibility Wrappers

The `scripts/` commands — `doctor.sh`, `manifest.py`, `publish-vindex.sh`, and
`publish-weights.sh` — are retained for existing automation. They call the same
Python package used by `skulk-weights`, so behavior should be tested through
both the package CLI and at least one wrapper dry-run before changing release
automation.
