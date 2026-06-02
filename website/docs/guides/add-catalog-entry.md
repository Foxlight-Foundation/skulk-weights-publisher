---
title: Add A Catalog Entry
---

Use this guide when you want to add an operator-owned vindex to the merged
catalog. The Foxlight catalog is already included; your job is to describe
the additional vindex library your organization wants to publish.

:::tip Adding to the Foxlight built-in catalog?

If you are a Foxlight contributor adding a model to the built-in catalog rather
than setting up an operator-owned catalog, use `catalog add` instead of editing
`foxlight.yaml` by hand:

```bash
uv run skulk-weights catalog add mlx-community/Qwen3-6B-4bit --dry-run
```

See the [`catalog add` CLI reference](../reference/cli.md) for details. The
rest of this guide covers the operator workflow — adding your own catalog
source alongside the built-in Foxlight entries.

:::

## 1. Create The Catalog Config

Start with a config file:

```bash
uv run skulk-weights catalog init
```

Edit `skulk-weights.yaml` so it points at your operator source:

```yaml
catalogs:
  - path: ./operator-vindexes.yaml
    namespace: my-org
    hf_owner: my-org
    hf_collection: my-org/vindexes-0123456789abcdef01234567
```

`namespace` is the prefix operators type in CLI commands. `hf_owner` is the
Hugging Face account or organization every entry in that source must publish
to. `hf_collection` is optional and points successful publishes from this source
at your own public or private vindex collection. These guards catch accidental
publishes to the wrong namespace before LARQL does any expensive work.

## 2. Choose The Stable Key

Add a short key to `operator-vindexes.yaml`:

```yaml
models:
  - key: llama-3-8b-full-q4-k
```

Use lowercase letters, numbers, and dashes. Include enough detail that the key
distinguishes model family, size, slice mode, and quantization. With the config
above, the effective CLI key is:

```text
my-org/llama-3-8b-full-q4-k
```

## 3. Pick The Source Model

`source_model` is the Hugging Face model LARQL reads from:

```yaml
source_model: meta-llama/Llama-3.1-8B-Instruct
```

If the upstream model is gated, make sure the publishing token has accepted the
model terms before running a real publish.

## 4. Select Quant And Slices

`quant` describes how LARQL stores the extracted vindex. The current publisher
accepts:

```yaml
quant: q4k
```

`slices` describes the vindex shape:

- `full`: publish the complete vindex
- `expert-server`: publish an MoE expert-server slice for CPU/high-memory
  weight serving

`full` cannot be combined with another slice in the same entry.

## 5. Assign The Tier

The tier controls how operators select groups of entries.

Use `smoke` for smaller vindexes that are appropriate for first publication
tests:

```yaml
tier: smoke
```

Use `moe` for larger MoE vindexes that should remain manual until the runner
has enough disk, memory, and network capacity. These are usually the entries
most relevant to keeping expert weights out of expensive GPU memory.

## 6. Set Output, Repository, And Collection Names

`output_name` is the local directory LARQL writes under scratch storage. It must
end in `.vindex` and must not include a slash.

`hf_repo` is the target Hugging Face repository, and its owner must match
`hf_owner` from `skulk-weights.yaml`:

```yaml
output_name: llama-3-1-8b-instruct-full-q4-k.vindex
hf_repo: my-org/llama-3-1-8b-instruct-full-q4-k-vindex
```

If the source config sets `hf_collection`, the published repo is added to that
collection after `larql publish` succeeds. You can also set `hf_collection` on a
single manifest entry when one entry needs a different collection.

## Optional Sidecar Fields

Beyond the vindex fields above, an entry can carry optional fields that describe
companion artifacts. These are mutually exclusive groups — set a full group or
none of it:

- `mtp_source_repo` / `mtp_sidecar_repo` / `mtp_quant` — an MTP sidecar
  extracted and quantized from the BF16 checkpoint. See the
  [MTP sidecar guide](mtp-sidecar.md).
- `vision_source_repo` / `vision_sidecar_repo` — a byte-for-byte mirror of a
  VLM's vision encoder. See the [Vision sidecar guide](vision-sidecar.md).
- `assistant_model_repo` — a companion `{model}-assistant` drafter model (the
  Gemma 4 pattern), recorded rather than extracted. Cannot be combined with any
  `mtp_*` field. See the [Gemma 4 assistant guide](gemma4-assistant.md).

`skulk-weights catalog add` auto-detects which of these applies when you add a
model, so you usually do not write them by hand.

## 7. Validate And Dry-Run

After editing the operator source, run:

```bash
uv run skulk-weights --config skulk-weights.yaml catalog validate
uv run skulk-weights --config skulk-weights.yaml catalog show \
  my-org/llama-3-8b-full-q4-k
uv run skulk-weights --config skulk-weights.yaml publish \
  --model my-org/llama-3-8b-full-q4-k \
  --dry-run
```

Commit the config and source file only after the dry-run command matches the
vindex you intend to build, publish, and place in Skulk.
