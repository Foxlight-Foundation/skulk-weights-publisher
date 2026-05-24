---
title: Workflow Reference
---

The repository has two GitHub Actions workflows: one for vindex publication and
one for documentation.

## Publish Workflow

`.github/workflows/publish.yml` has two jobs:

- `validate`: runs safe checks on hosted GitHub runners
- `publish`: runs LARQL publication on the self-hosted runner

Manual dispatch inputs:

- `model`: one catalog key or `all`
- `catalog_config`: optional `skulk-vindex.yaml` path for operator sources
- `tier`: used when `model` is `all`
- `dry_run`: prints commands without publishing

The validate job installs the package, lints the code, type-checks it, runs unit
tests, validates the effective catalog, and dry-runs every catalog entry.

The publish job uses the `self-hosted`, `linux`, `larql`, and `vindex` runner
labels. It resolves the requested catalog keys and runs the same CLI operators
use locally. After each successful LARQL publish, the CLI adds the published
model repo to the entry's configured Hugging Face collection.

By default, the workflow uses the built-in Foxlight catalog. Pass
`catalog_config` when dispatching the workflow to include operator sources
from a checked-in `skulk-vindex.yaml`; Foxlight entries remain included.
Set the repository variable `SKULK_VINDEX_COLLECTION` only when you need to
override the collection target for the selected run.

This job publishes the vindexes Skulk will later place across runtime hardware.
The runner performs extraction and upload; it is not required to be the GPU node
or the CPU/high-memory LARQL server that eventually consumes the output.

## Docs Workflow

`.github/workflows/docs.yml` builds the Docusaurus site on pull requests and
pushes to branches.

Pull requests upload `website/build` as an artifact so reviewers can inspect the
generated static site.

Pushes to `main` publish the production site from the `gh-pages` branch root:

```text
https://foxlight-foundation.github.io/skulk-vindex-publisher/
```

Pushes to other branches publish branch previews under `previews/<branch>`:

```text
https://foxlight-foundation.github.io/skulk-vindex-publisher/previews/feature-publisher-production-foundation/
```

The workflow sets Docusaurus `baseUrl` per branch so links and static assets use
the same path that GitHub Pages serves.
