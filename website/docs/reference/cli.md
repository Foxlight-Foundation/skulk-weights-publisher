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

`catalogue` is a legacy alias for `catalog`; every `catalog` subcommand below
also works under `catalogue`.

## Global Options

Global options come before the subcommand. `--config` and `--manifest` are
mutually exclusive.

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

## `skulk-weights catalog find HF_URL_OR_OWNER/REPO`

Reverse lookup: given an upstream source model, prints every catalog entry
whose `source_model` matches. The lookup is one-to-many — a single source model
can map to several entries (for example a `full` slice plus an `expert-server`
slice) — so each matching entry is printed as a JSON object, one per line.

Accepts either a bare `owner/repo` string or a full `https://huggingface.co/...`
URL. Respects `--config`, so it searches operator sources too.

Exits 1 with a stderr message when no entry matches the given source model.

Example:

```bash
skulk-weights catalog find google/gemma-3-4b-it
skulk-weights catalog find https://huggingface.co/google/gemma-3-4b-it
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
4. If no `mtp.*` keys are found, checks for a Gemma-4-style companion model
   named `{model}-assistant` and, when present, writes `assistant_model_repo`
   instead of any MTP fields
5. Derives `key`, `output_name`, and `hf_repo` from the model ID
6. Validates no key, output name, or repo collisions against the existing catalog
7. Appends the entry to the built-in `foxlight.yaml`

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

- `vindex`: runs `larql extract`, `larql publish`, and files the repository
  into the configured Hugging Face collection.
- `mtp`: downloads only the shards that contain `mtp.*` tensor keys from the
  original BF16 checkpoint and uploads them at full precision (bf16,
  unquantized) as `mtp.safetensors` to the sidecar repository. Requires
  `mtp_source_repo` and `mtp_sidecar_repo` on the catalog entry.
- `vision`: mirrors the vision source repo's weights and configs into the
  vision sidecar repo byte-for-byte — no quantization and no dtype conversion.
  Requires `vision_source_repo` and `vision_sidecar_repo` on the catalog entry;
  raises an error only when those are not configured.
- `all` (default): runs `vindex`, then `mtp`, then `vision`, skipping any
  artifact not configured on the entry.

### Model cards

Every real publish (vindex, mtp, or vision) also uploads a self-describing
`README.md` model card to the published repo, with frontmatter recording the
`base_model`, `tags`, the inherited source `license`, and a `foxlight:` block
(artifact type, source repo and pinned source revision, target model, quant,
catalog key, and generation timestamp). The source revision and license are
resolved best-effort from the Hub using `HF_TOKEN`. See the
[Manifest Reference](./manifest.md) for the entry fields these cards describe.

### Collections

Each artifact is filed into the Hugging Face collection for its type:

- the vindex is filed into the configured slug exactly — the entry's
  `hf_collection` or the `SKULK_WEIGHTS_COLLECTION` override;
- mtp and vision sidecars are filed into their per-type collections,
  `MTP Sidecars` and `Vision Sidecars`, resolved by title (created if missing,
  reused if present).

Filing is disabled when no collection is configured for the entry, or when
`SKULK_WEIGHTS_COLLECTION` is set to a disable value (see the
[Environment Reference](./environment.md)).

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
- the vindex collection (`vindex collection: <slug>`, or `collection: disabled`)
- `larql extract` command (vindex artifact)
- `larql publish` command (vindex artifact)
- MTP source repo, sidecar repo, precision (`bf16 (unquantized)`), and output
  path, plus `mtp collection: MTP Sidecars` (mtp artifact)
- vision source repo, sidecar repo, the mirror note, plus
  `vision collection: Vision Sidecars` (vision artifact)
- a note when mtp or vision is not configured for the entry
