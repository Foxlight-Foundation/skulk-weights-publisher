---
title: Error Reference
---

Use this page when a publisher command exits before running the publish you
expected.

## `skulk-vindex.yaml not found`

The built-in Foxlight catalogue works without a config file. This error means
you passed `--config PATH` and that path does not exist.

```bash
skulk-vindex --config skulk-vindex.yaml catalogue validate
```

## `models.yaml not found; run from the repository root`

This error appears when legacy `--manifest` mode or an operator source path
points at a missing manifest file. Run the command from the repository root or
pass the correct path:

```bash
skulk-vindex --manifest /path/to/models.yaml manifest validate
```

## `hf_repo owner must be '...'`

An operator source in `skulk-vindex.yaml` declares `hf_owner`, and one entry is
trying to publish to a different Hugging Face owner. Fix `hf_repo` or use a
different source block with the correct owner.

## `hf_collection owner must be '...'`

An operator source declares `hf_owner`, and its collection points at a different
Hugging Face owner. Use a collection owned by the same account or organization,
or move the entry into a separate source block.

## `key must be lowercase kebab-case`

Manifest keys are stable automation selectors. Use lowercase letters, numbers,
and dashes.

## `full must not be combined with other slices`

`full` represents the complete vindex. Create a separate manifest entry for an
`expert-server` slice.

## `larql is required for non-dry-run publishing`

The command is trying to publish for real. Install LARQL and make sure
`larql` is on `PATH`, or rerun with `--dry-run`.

## `HF_TOKEN is required for non-dry-run publishing`

Set `HF_TOKEN` to a Hugging Face token with write access to the target
repository and collection.

## `failed to add ... to collection ...`

The vindex repo was published, but the follow-up collection update failed.
Check that `HF_TOKEN` can write to the collection and that the collection slug is
correct.

## `output path already exists`

The local scratch output directory already exists. Remove the directory, choose
another scratch root, or rerun with `--force` if replacing the local output is
intentional.

Use `--force` only when replacing the local extraction output is expected:

```bash
skulk-vindex publish --model foxlight/gemma-3-4b-full-q4-k --force
```
