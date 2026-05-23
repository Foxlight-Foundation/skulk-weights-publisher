---
title: Add A Catalogue Entry
---

Use this guide when you want to add an operator-owned vindex to the merged
catalogue. The Foxlight catalogue is already included; your job is to describe
the additional vindex library your organization wants to publish.

## 1. Create The Catalogue Config

Start with a config file:

```bash
skulk-vindex catalogue init
```

Edit `skulk-vindex.yaml` so it points at your operator source:

```yaml
catalogues:
  - path: ./operator-vindexes.yaml
    namespace: my-org
    hf_owner: my-org
```

`namespace` is the prefix operators type in CLI commands. `hf_owner` is the
Hugging Face account or organization every entry in that source must publish
to. This guard catches accidental publishes to the wrong namespace before LARQL
does any expensive work.

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

## 6. Set Output And Repository Names

`output_name` is the local directory LARQL writes under scratch storage. It must
end in `.vindex` and must not include a slash.

`hf_repo` is the target Hugging Face repository, and its owner must match
`hf_owner` from `skulk-vindex.yaml`:

```yaml
output_name: llama-3-1-8b-instruct-full-q4-k.vindex
hf_repo: my-org/llama-3-1-8b-instruct-full-q4-k-vindex
```

## 7. Validate And Dry-Run

After editing the operator source, run:

```bash
skulk-vindex --config skulk-vindex.yaml catalogue validate
skulk-vindex --config skulk-vindex.yaml catalogue get \
  --key my-org/llama-3-8b-full-q4-k
skulk-vindex --config skulk-vindex.yaml publish \
  --model my-org/llama-3-8b-full-q4-k \
  --dry-run
```

Commit the config and source file only after the dry-run command matches the
vindex you intend to build, publish, and place in Skulk.
