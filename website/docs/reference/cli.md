---
title: CLI Reference
---

## `skulk-vindex manifest validate`

Validates `models.yaml`.

Use this before every publish and after every catalogue edit:

```bash
skulk-vindex manifest validate
```

## `skulk-vindex manifest list`

Lists manifest keys.

Options:

- `--tier all`
- `--tier smoke`
- `--tier moe`

Example:

```bash
skulk-vindex manifest list --tier smoke
```

## `skulk-vindex manifest get --key KEY`

Prints one manifest entry as JSON.

Example:

```bash
skulk-vindex manifest get --key gemma-3-4b-full-q4-k
```

## `skulk-vindex doctor`

Checks local prerequisites that are safe on any machine.

## `skulk-vindex doctor --publish`

Adds publication-specific checks for `larql` and `HF_TOKEN`.

## `skulk-vindex publish --model KEY`

Runs extraction and publication for one manifest entry.

Options:

- `--dry-run`: print commands without running LARQL
- `--force`: replace an existing local output path
- `--scratch PATH`: override `SKULK_VINDEX_SCRATCH`

Examples:

```bash
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
skulk-vindex publish --model gemma-3-4b-full-q4-k --scratch /fast/skulk-vindexes
```

## Global Options

All commands accept `--manifest PATH` before the subcommand:

```bash
skulk-vindex --manifest /path/to/models.yaml manifest validate
```
