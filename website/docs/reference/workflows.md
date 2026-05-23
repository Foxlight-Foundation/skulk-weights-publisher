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

- `model`: one manifest key or `all`
- `tier`: used when `model` is `all`
- `dry_run`: prints commands without publishing

The validate job installs the package, lints the code, type-checks it, runs unit
tests, validates `models.yaml`, and dry-runs every catalogue entry.

The publish job uses the `self-hosted`, `linux`, `larql`, and `vindex` runner
labels. It resolves the requested manifest keys and runs the same CLI operators
use locally.

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
