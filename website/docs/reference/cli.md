---
title: CLI Reference
---

The `skulk-weights` CLI has three jobs:

- inspect the merged catalog
- check the local environment
- plan or run one weight publication

Those jobs keep publication reviewable before LARQL extracts large weight
directories. The published artifacts are the stable objects Skulk can later place
across GPU inference nodes and CPU/high-memory weight-serving nodes.

## Global Options

Global options come before the subcommand.

`--config PATH` loads `PATH` as `skulk-weights.yaml`. The built-in Foxlight
catalog is still included, and the config can add operator catalog sources.

```bash
skulk-weights --config skulk-weights.yaml catalog validate
```

`--manifest PATH` is a legacy single-file mode. It bypasses the merged
catalog and reads one manifest source directly.

```bash
skulk-weights --manifest /path/to/models.yaml catalog validate
```

## `skulk-weights catalog validate`

Validates the effective catalog. Run this after changing
`skulk-weights.yaml` or any source file.

Example:

```bash
skulk-weights catalog validate
skulk-weights --config skulk-weights.yaml catalog validate
```

## `skulk-weights catalog sources`

Prints the sources that contributed entries to the merged catalog.

Example:

```bash
skulk-weights catalog sources
```

## `skulk-weights catalog list`

Lists effective catalog keys. Use this to see which entries are available by
tier.

Options:

- `--tier all`
- `--tier smoke`
- `--tier moe`

Example:

```bash
skulk-weights catalog list --tier smoke
```

## `skulk-weights catalog show KEY`

Prints one catalog entry as JSON. Use this when you want to inspect exactly
what a namespaced key resolves to before publishing.

Example:

```bash
skulk-weights catalog show foxlight/gemma-3-4b-full-q4-k
```

## `skulk-weights catalog init`

Writes a starter `skulk-weights.yaml`. The generated file is valid immediately
because the Foxlight catalog is included automatically. Add operator sources
when you are ready.

Options:

- `--output PATH`: write to a path other than `skulk-weights.yaml`
- `--force`: replace an existing file

Example:

```bash
skulk-weights catalog init
```

## `skulk-weights catalog add HF_MODEL_ID`

Adds a new Foxlight catalog entry by fetching metadata from a Hugging Face
model repo and generating the YAML block automatically. Use this instead of
editing `foxlight.yaml` by hand.

Options:

- `--dry-run`: print the generated YAML block without writing anything
- `--yes` / `-y`: skip the confirmation prompt before writing

What it does:

1. Resolves the HF model ID (accepts bare `owner/repo` strings)
2. Fetches model card tags to detect quant scheme and tier
3. Checks the base model for `mtp.*` tensor keys and populates MTP sidecar fields if found
4. Derives `key`, `output_name`, and `hf_repo` from the model ID
5. Validates no key, output name, or repo collisions against the existing catalog
6. Appends the entry to the built-in `foxlight.yaml`

Examples:

```bash
skulk-weights catalog add mlx-community/Qwen3-6B-4bit --dry-run
skulk-weights catalog add mlx-community/Qwen3-6B-4bit
```

The command exits 1 if the detected quant scheme is not supported, or if the
generated key, `output_name`, or `hf_repo` would collide with an existing entry.
Always use `--dry-run` first to review the generated block before writing.

Setting `HF_TOKEN` in the environment is recommended — it allows the MTP key
scan to access gated base model repos without hitting rate limits.

## `skulk-weights scratch clean`

Deletes the scratch directory and all cached weight shards inside it. Use this
to reclaim disk space after a publish run or to force a clean re-download.

Options:

- `--scratch PATH`: override `SKULK_WEIGHTS_SCRATCH` for this operation
- `--yes` / `-y`: skip the confirmation prompt

Examples:

```bash
skulk-weights scratch clean
skulk-weights scratch clean --yes
skulk-weights scratch clean --scratch /fast/skulk-weights --yes
```

The command refuses to delete paths that are too broad: home directory, root,
current working directory, any ancestor of the current working directory, or
any path fewer than three components deep.

## `skulk-weights doctor`

Checks local prerequisites that are safe on any machine: Python dependencies,
scratch directory access, and catalog validity.

## `skulk-weights doctor --publish`

Adds publication-specific checks for `larql`, `HF_TOKEN`, and the
`huggingface_hub` package used for collection updates.

## `skulk-weights publish --model KEY`

Builds the publish plan for one catalog entry. With `--dry-run`, it only
prints the plan. Without `--dry-run`, it runs the selected artifact step:

- `vindex`: runs `larql extract`, `larql publish`, and adds the repository to
  the configured Hugging Face collection.
- `mtp`: downloads only the shards that contain `mtp.*` tensor keys from the
  original BF16 checkpoint, quantizes them, and uploads `mtp.safetensors` to
  the sidecar repository. Requires `mtp_source_repo`, `mtp_sidecar_repo`, and
  `mtp_quant` on the catalog entry.
- `vision`: not yet implemented; raises an error if passed without `--dry-run`.
- `all` (default): runs `vindex`, then `mtp` if configured on the entry.

Options:

- `--artifact vindex|mtp|vision|all`: publish only the named artifact, or all
  declared artifacts when omitted.
- `--dry-run`: print the plan without running any extraction or upload
- `--force`: replace an existing local output path
- `--scratch PATH`: override `SKULK_WEIGHTS_SCRATCH`

Examples:

```bash
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k
skulk-weights publish --model my-org/qwen3-6b-full-q4-k --artifact mtp --dry-run
skulk-weights publish --model my-org/qwen3-6b-full-q4-k --artifact mtp
skulk-weights --config skulk-weights.yaml publish \
  --model my-org/llama-3-8b-full-q4-k \
  --scratch /fast/skulk-weights
```

Expected dry-run output includes:

- catalog key
- tier
- artifact selection
- source model
- local output path
- target Hugging Face repository
- target Hugging Face collection
- `larql extract` command (vindex artifact)
- `larql publish` command (vindex artifact)
- MTP source repo, sidecar repo, quant, and output path (mtp artifact)
- note when mtp or vision is not configured for the entry
