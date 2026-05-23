---
title: Troubleshooting
---

Most publisher errors fall into one of three groups:

- the command cannot find an operator config or manifest source
- the local machine is missing a publishing prerequisite
- the scratch output path already exists

## `skulk-vindex.yaml not found`

The built-in Foxlight catalogue does not need a config file. You only need
`--config PATH` when you are adding operator catalogue sources. If you pass
`--config`, make sure the path exists:

```bash
skulk-vindex --config skulk-vindex.yaml catalogue validate
```

## `models.yaml not found`

This error comes from legacy single-manifest mode or from an operator source
listed in `skulk-vindex.yaml`. Run commands from the repository root, fix the
source path, or pass `--manifest PATH` when you intentionally want legacy mode.

## `missing command: larql`

Install LARQL and make sure it is on `PATH`. This is required only for real
publication, not for normal validation or dry-runs.

Run the safe command first:

```bash
skulk-vindex publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

Then run the publishing preflight on the runner:

```bash
skulk-vindex doctor --publish
```

## `HF_TOKEN is not set`

Real publication needs a Hugging Face token with write access to the target
repository.

In GitHub Actions, configure `HF_TOKEN` as a repository secret. On a manual
runner shell, export it before publishing:

```bash
export HF_TOKEN=...
```

## `output path already exists`

The publisher refuses to overwrite local extraction output by default. Remove
the directory manually, choose another scratch root, or rerun with `--force`
when replacement is intentional.

This check matters because extraction output can be large and may correspond to
a specific runtime role. Accidentally reusing a path can leave operators unsure
which vindex should be placed on GPU inference nodes or CPU/high-memory LARQL
servers.

Use a different scratch root when you want to keep the old output:

```bash
skulk-vindex publish \
  --model foxlight/gemma-3-4b-full-q4-k \
  --scratch /fast/scratch/skulk-vindexes-2
```
