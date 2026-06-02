---
title: Troubleshooting
---

Most publisher errors fall into one of three groups:

- the command cannot find an operator config or manifest source
- the local machine is missing a publishing prerequisite
- the scratch output path already exists
- Hugging Face accepted the publish but rejected the collection update

## `skulk-weights.yaml not found`

The built-in Foxlight catalog does not need a config file. You only need
`--config PATH` when you are adding operator catalog sources. If you pass
`--config`, make sure the path exists:

```bash
uv run skulk-weights --config skulk-weights.yaml catalog validate
```

## `models.yaml not found`

This error comes from legacy single-manifest mode or from an operator source
listed in `skulk-weights.yaml`. Run commands from the repository root, fix the
source path, or pass `--manifest PATH` when you intentionally want legacy mode.

## `missing command: larql`

Install LARQL and make sure it is on `PATH`. This is required only for real
publication, not for normal validation or dry-runs.

Run the safe command first:

```bash
uv run skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

Then run the publishing preflight on the runner:

```bash
uv run skulk-weights doctor --publish
```

## `HF_TOKEN is not set`

Real publication needs a Hugging Face token with write access to the target
repository and collection.

In GitHub Actions, configure `HF_TOKEN` as a repository secret. On a manual
runner shell, export it before publishing:

```bash
export HF_TOKEN=...
```

## `failed to add ... to collection ...`

The LARQL publish finished, but the Hugging Face collection update failed.
Confirm that the collection slug exists and that `HF_TOKEN` has permission to
modify it. The built-in Foxlight entries target:

```text
FoxlightAI/vindexes-6a124406dd5fb439c431b051
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
uv run skulk-weights publish \
  --model foxlight/gemma-3-4b-full-q4-k \
  --scratch /fast/scratch/skulk-weights-2
```

## `no MTP sidecar configured for ...`

This error occurs during real (non-dry-run) publishing. You ran `--artifact mtp`
on an entry that does not have `mtp_source_repo` and `mtp_sidecar_repo` set in
the catalog. In dry-run mode the publisher instead prints
`mtp step: not configured for this entry` and exits cleanly.

Either add those fields to the entry or use `--artifact vindex` to publish only
the vindex.

See the [MTP sidecar guide](guides/mtp-sidecar.md) for the required catalog
fields and how to structure the entry.

## `no mtp.* keys found in ...`

SWP downloaded the source checkpoint but found no tensors matching `mtp.*`. This
means either:

- the `mtp_source_repo` points to a quantized checkpoint rather than the
  original BF16 weights — MTP tensors are stripped by quantization pipelines,
  so `mtp_source_repo` must always point to the BF16 source, not a derivative.
- the model was not trained with native MTP heads at all.

Confirm that the source repo contains the original checkpoint (usually the
upstream model card on Hugging Face will say so) and update `mtp_source_repo`
accordingly.

## `no vision sidecar configured for ...`

This error occurs during real (non-dry-run) publishing. You ran
`--artifact vision` on an entry that does not have both `vision_source_repo` and
`vision_sidecar_repo` set in the catalog. The two fields are both-or-none.

Either add those fields to the entry or use `--artifact vindex` to publish only
the vindex. See the [Vision sidecar guide](guides/vision-sidecar.md) for the
required catalog fields and how to structure the entry.

## `no .safetensors weights found in ...`

SWP resolved the `vision_source_repo` but found no `.safetensors` files to
mirror. A vision sidecar mirrors the encoder weights byte-for-byte, so the
source repo must actually contain `.safetensors` weights. Confirm that
`vision_source_repo` points at the repository that holds the vision-encoder
weights (often a third-party repo, not the mlx-community quant that omits them).
