---
title: Error Reference
---

Use this page when a publisher command exits before running the publish you
expected.

## `skulk-weights.yaml not found`

The built-in Foxlight catalog works without a config file. This error means
you passed `--config PATH` and that path does not exist.

```bash
skulk-weights --config skulk-weights.yaml catalog validate
```

## `models.yaml not found; run from the repository root`

This error appears when legacy `--manifest` mode or an operator source path
points at a missing manifest file. Run the command from the repository root or
pass the correct path:

```bash
skulk-weights --manifest /path/to/models.yaml catalog validate
```

## `hf_repo owner must be '...'`

An operator source in `skulk-weights.yaml` declares `hf_owner`, and one entry is
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

## `larql is required for vindex publishing`

The command is trying to publish a vindex for real. Install LARQL and make sure
`larql` is on `PATH`, or rerun with `--dry-run`.

## `HF_TOKEN is required for non-dry-run publishing`

Set `HF_TOKEN` to a Hugging Face token with write access to the target
repository and collection.

## `no MTP sidecar configured for <key>; add mtp_source_repo and mtp_sidecar_repo to the catalog entry`

You ran `--artifact mtp` against an entry that has no MTP sidecar configured.
Add `mtp_source_repo` and `mtp_sidecar_repo` to the catalog entry, or select a
different artifact.

## `no vision sidecar configured for <key>; add vision_source_repo and vision_sidecar_repo to the catalog entry`

You ran `--artifact vision` against an entry that has no vision sidecar
configured. Add `vision_source_repo` and `vision_sidecar_repo` to the catalog
entry, or select a different artifact.

## `no MTP head tensors found in <repo>`

The MTP source repo does not contain any detectable MTP tensors. SWP checks two
layouts: `mtp.*` / `.mtp.*` keys (Qwen3, DeepSeek V4-Flash), and
`model.layers.{num_hidden_layers}.*` via `config.json` (DeepSeek V3/V3-0324).

Point `mtp_source_repo` at the original BF16 or FP8 release rather than an
mlx-converted checkpoint — mlx-lm strips MTP tensors during conversion.

## `mlx is required for reading MTP weights`

MTP extraction uses mlx for safetensors output. Run the `mtp` artifact on a
host with `mlx` installed (`uv sync --extra mtp`).

## `no .safetensors weights found in <repo>`

The vision source repo has no `.safetensors` weights to mirror. Confirm
`vision_source_repo` points at a repo that publishes its weights in
`safetensors` format.

## `SKULK_WEIGHTS_COLLECTION must look like owner/slug or be 'none'`

`SKULK_WEIGHTS_COLLECTION` was set to something that is neither a valid
`owner/slug` collection slug nor a disable value. Fix the slug, or set it to one
of `none`, `0`, `false`, `no`, `off`, or `disabled` to skip collection filing.

## `failed to ensure collection ...`

A sidecar (mtp or vision) was published, but resolving or creating its per-type
collection (`MTP Sidecars` / `Vision Sidecars`) failed. Check that `HF_TOKEN`
can create and read collections for the target owner.

## `failed to add ... to collection ...`

The repo was published, but the follow-up collection update failed.
Check that `HF_TOKEN` can write to the collection and that the collection slug is
correct.

## `output path already exists`

The local scratch output directory already exists. Remove the directory, choose
another scratch root, or rerun with `--force` if replacing the local output is
intentional.

Use `--force` only when replacing the local extraction output is expected:

```bash
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --force
```
