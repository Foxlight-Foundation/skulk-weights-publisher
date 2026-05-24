---
title: CLI Reference
---

The `skulk-vindex` CLI has three jobs:

- inspect the merged catalogue
- check the local environment
- plan or run one vindex publication

Those jobs keep publication reviewable before LARQL extracts large weight
directories. The published vindex is the stable object Skulk can later place
across GPU inference nodes and CPU/high-memory weight-serving nodes.

## Global Options

Global options come before the subcommand.

`--config PATH` loads `PATH` as `skulk-vindex.yaml`. The built-in Foxlight
catalogue is still included, and the config can add operator catalogue sources.

```bash
skulk-vindex --config skulk-vindex.yaml catalogue validate
```

`--manifest PATH` is a legacy single-file mode. It bypasses the merged
catalogue and reads one manifest source directly.

```bash
skulk-vindex --manifest /path/to/models.yaml manifest validate
```

## `skulk-vindex catalogue validate`

Validates the effective catalogue. Run this after changing
`skulk-vindex.yaml` or any source file.

Example:

```bash
skulk-vindex catalogue validate
skulk-vindex --config skulk-vindex.yaml catalogue validate
```

## `skulk-vindex catalogue sources`

Prints the sources that contributed entries to the merged catalogue.

Example:

```bash
skulk-vindex catalogue sources
```

## `skulk-vindex catalogue list`

Lists effective catalogue keys. Use this to see which vindexes are available by
tier.

Options:

- `--tier all`
- `--tier smoke`
- `--tier moe`

Example:

```bash
skulk-vindex catalogue list --tier smoke
```

## `skulk-vindex catalogue get --key KEY`

Prints one catalogue entry as JSON. Use this when you want to inspect exactly
what a namespaced key resolves to before publishing.

Example:

```bash
skulk-vindex catalogue get --key foxlight/gemma-3-4b-full-q4-k
```

## `skulk-vindex catalogue init`

Writes a starter `skulk-vindex.yaml`. The generated file is valid immediately
because the Foxlight catalogue is included automatically. Add operator sources
when you are ready.

Options:

- `--output PATH`: write to a path other than `skulk-vindex.yaml`
- `--force`: replace an existing file

Example:

```bash
skulk-vindex catalogue init
```

## `skulk-vindex manifest ...`

`manifest` is a compatibility alias for the catalogue inspection commands.
Prefer `catalogue` in new automation.

Use `--manifest PATH` when you need true single-file legacy behavior:

```bash
skulk-vindex --manifest models.yaml manifest list --tier smoke
```

## `skulk-vindex doctor`

Checks local prerequisites that are safe on any machine: Python dependencies,
scratch directory access, and catalogue validity.

## `skulk-vindex doctor --publish`

Adds publication-specific checks for `larql`, `HF_TOKEN`, and the
`huggingface_hub` package used for collection updates.

## `skulk-vindex publish --model KEY`

Builds the publish plan for one catalogue entry. With `--dry-run`, it only
prints the plan. Without `--dry-run`, it runs LARQL extraction and publication,
then adds the published repository to the configured Hugging Face collection.
Review the slice mode before publishing; it is the part of the catalogue that
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

- catalogue key
- source model
- local output path
- target Hugging Face repository
- target Hugging Face collection
- `larql extract` command
- `larql publish` command
