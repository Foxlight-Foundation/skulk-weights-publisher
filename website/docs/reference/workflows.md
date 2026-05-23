---
title: Workflow Reference
---

## Publish Workflow

`.github/workflows/publish.yml` has two jobs:

- `validate`: runs on hosted GitHub runners and performs safe checks
- `publish`: runs only for manual dispatch or schedule on the self-hosted runner

Manual dispatch inputs:

- `model`: one manifest key or `all`
- `tier`: used when `model` is `all`
- `dry_run`: prints commands without publishing

Scheduled publication is intentionally scoped to smoke-tier entries. MoE entries
should stay manual until the runner has enough disk, memory, and network
capacity for the selected model family.

## Docs Workflow

`.github/workflows/docs.yml` builds the Docusaurus site on pull requests and
pushes to `main`.

Pull requests upload `website/build` as an artifact so reviewers can inspect the
generated static site.

Pushes to `main` upload the same build output to GitHub Pages and deploy it with
`actions/deploy-pages`.
