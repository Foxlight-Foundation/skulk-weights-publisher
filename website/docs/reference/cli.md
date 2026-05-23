---
title: CLI Reference
---

The `skulk-vindex` CLI has three jobs:

- inspect the catalogue
- check the local environment
- plan or run one vindex publication

Those jobs keep publication reviewable before LARQL extracts large weight
directories. The published vindex is the stable object Skulk can later place
across GPU inference nodes and CPU/high-memory weight-serving nodes.

## `skulk-vindex manifest validate`

Validates `models.yaml`. Run this after changing the catalogue and before
publishing.

Example:

```bash
skulk-vindex manifest validate
```

## `skulk-vindex manifest list`

Lists manifest keys. Use this to see which vindexes are available by tier.

Options:

- `--tier all`
- `--tier smoke`
- `--tier moe`

Example:

```bash
skulk-vindex manifest list --tier smoke
```

## `skulk-vindex manifest get --key KEY`

Prints one manifest entry as JSON. Use this when you want to inspect exactly
what a key resolves to before publishing.

Example:

```bash
skulk-vindex manifest get --key gemma-3-4b-full-q4-k
```

## `skulk-vindex doctor`

Checks local prerequisites that are safe on any machine: Python dependencies,
scratch directory access, and manifest validity.

## `skulk-vindex doctor --publish`

Adds publication-specific checks for `larql` and `HF_TOKEN`.

## `skulk-vindex publish --model KEY`

Builds the publish plan for one manifest entry. With `--dry-run`, it only prints
the plan. Without `--dry-run`, it runs LARQL extraction and publication.
Review the slice mode before publishing; it is the part of the manifest that
connects this vindex to the intended runtime hardware role.

Options:

- `--dry-run`: print commands without running LARQL
- `--force`: replace an existing local output path
- `--scratch PATH`: override `SKULK_VINDEX_SCRATCH`

Examples:

```bash
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
skulk-vindex publish --model gemma-3-4b-full-q4-k --scratch /fast/skulk-vindexes
```

Expected dry-run output includes:

- manifest key
- source model
- local output path
- target Hugging Face repository
- `larql extract` command
- `larql publish` command

## Global Options

All commands accept `--manifest PATH` before the subcommand:

```bash
skulk-vindex --manifest /path/to/models.yaml manifest validate
```
