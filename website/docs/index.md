---
slug: /
title: Skulk Vindex Publisher
---

Skulk Vindex Publisher builds and publishes LARQL vindex artifacts that Skulk can
download at runtime.

If you are new to the project, start with the [quickstart](quickstart.md). It
shows the safe dry-run path before any model extraction or Hugging Face upload
happens.

## What This Tool Does

- validates the vindex catalogue in `models.yaml`
- plans deterministic `larql extract` and `larql publish` commands
- checks local runner prerequisites
- runs safe dry-runs in pull requests
- supports real publishing from a controlled self-hosted runner

## What This Tool Does Not Do Yet

- it does not prove Skulk's future MLX-to-LARQL delegation path
- it does not make Skulk load vindexes on the MLX head
- it does not bypass Hugging Face access, license, or token requirements

The publisher is product infrastructure. It should make artifact creation
repeatable and auditable before Skulk depends on those artifacts.
