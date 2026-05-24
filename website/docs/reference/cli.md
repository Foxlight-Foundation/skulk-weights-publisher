---
title: CLI Reference
---

The `skulk-vindex` CLI has three jobs:

- inspect the merged catalog
- check the local environment
- plan or run one vindex publication

Those jobs keep publication reviewable before LARQL extracts large weight
directories. The published vindex is the stable object Skulk can later place
across GPU inference nodes and CPU/high-memory weight-serving nodes.

## Global Options

Global options come before the subcommand.

`--config PATH` loads `PATH` as `skulk-vindex.yaml`. The built-in Foxlight
catalog is still included, and the config can add operator catalog sources.

```bash
skulk-vindex --config skulk-vindex.yaml catalog validate
```

`--manifest PATH` is a legacy single-file mode. It bypasses the merged
catalog and reads one manifest source directly.

```bash
skulk-vindex --manifest /path/to/models.yaml manifest validate
```

## `skulk-vindex catalog validate`

Validates the effective catalog. Run this after changing
`skulk-vindex.yaml` or any source file.

Example:

```bash
skulk-vindex catalog validate
skulk-vindex --config skulk-vindex.yaml catalog validate
```

## `skulk-vindex catalog sources`

Prints the sources that contributed entries to the merged catalog.

Example:

```bash
skulk-vindex catalog sources
```

## `skulk-vindex catalog list`

Lists effective catalog keys. Use this to see which vindexes are available by
tier.

Options:

- `--tier all`
- `--tier smoke`
- `--tier moe`

Example:

```bash
skulk-vindex catalog list --tier smoke
```

## `skulk-vindex catalog get --key KEY`

Prints one catalog entry as JSON. Use this when you want to inspect exactly
what a namespaced key resolves to before publishing.

Example:

```bash
skulk-vindex catalog get --key foxlight/gemma-3-4b-full-q4-k
```

## `skulk-vindex catalog init`

Writes a starter `skulk-vindex.yaml`. The generated file is valid immediately
because the Foxlight catalog is included automatically. Add operator sources
when you are ready.

Options:

- `--output PATH`: write to a path other than `skulk-vindex.yaml`
- `--force`: replace an existing file

Example:

```bash
skulk-vindex catalog init
```

## `skulk-vindex manifest ...`

`manifest` is a compatibility alias for the catalog inspection commands.
Prefer `catalog` in new automation.

Use `--manifest PATH` when you need true single-file legacy behavior:

```bash
skulk-vindex --manifest models.yaml manifest list --tier smoke
```

## `skulk-vindex doctor`

Checks local prerequisites that are safe on any machine: Python dependencies,
scratch directory access, and catalog validity.

## `skulk-vindex doctor --publish`

Adds publication-specific checks for `larql`, `HF_TOKEN`, and the
`huggingface_hub` package used for collection updates.

## `skulk-vindex publish --model KEY`

Builds the publish plan for one catalog entry. With `--dry-run`, it only
prints the plan. Without `--dry-run`, it runs LARQL extraction and publication,
then adds the published repository to the configured Hugging Face collection.
Review the slice mode before publishing; it is the part of the catalog that
connects this vindex to the intended runtime hardware role.

Options:

- `--dry-run`: print commands without running LARQL
- `--force`: replace an existing local output path
- `--scratch PATH`: override `SKULK_VINDEX_SCRATCH`

Examples:

```bash
skulk-vindex publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
skulk-vindex --config skulk-vindex.yaml publish \
  --model my-org/llama-3-8b-full-q4-k \
  --scratch /fast/skulk-vindexes
```

Expected dry-run output includes:

- catalog key
- source model
- local output path
- target Hugging Face repository
- target Hugging Face collection
- `larql extract` command
- `larql publish` command
