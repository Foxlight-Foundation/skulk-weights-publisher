---
title: Error Reference
---

## `models.yaml not found; run from the repository root`

The default manifest path is `models.yaml` in the current directory. Run the
command from the repository root or pass `--manifest PATH`.

## `key must be lowercase kebab-case`

Manifest keys are stable automation selectors. Use lowercase letters, numbers,
and dashes.

## `full must not be combined with other slices`

`full` represents the complete vindex artifact. Create a separate manifest entry
for an `expert-server` slice.

## `larql is required for non-dry-run publishing`

The command is trying to publish for real. Install LARQL and make sure
`larql` is on `PATH`, or rerun with `--dry-run`.

## `HF_TOKEN is required for non-dry-run publishing`

Set `HF_TOKEN` to a Hugging Face token with write access to the target
repository.

## `output path already exists`

The local scratch output directory already exists. Remove the directory, choose
another scratch root, or rerun with `--force` if replacing the local output is
intentional.
