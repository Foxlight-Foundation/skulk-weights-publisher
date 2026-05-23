---
title: Add A Catalogue Entry
---

Use this guide when you want to add a new publishable vindex.

A catalogue entry should be boring and specific. It should tell a future
operator exactly which model LARQL will extract, how the vindex should be
stored, what local directory will be created, and where the vindex will be
published. For sliced entries, it should also make the intended runtime role
obvious: complete model representation or expert-server weight serving.

## 1. Choose The Stable Key

The `key` is the name operators type:

```yaml
key: llama-3-2-3b-full-q4-k
```

Use lowercase letters, numbers, and dashes. Include enough detail that the key
distinguishes model family, size, slice mode, and quantization.

## 2. Pick The Source Model

`source_model` is the Hugging Face model LARQL reads from:

```yaml
source_model: meta-llama/Llama-3.2-3B-Instruct
```

If the upstream model is gated, make sure the publishing token has accepted the
model terms before running a real publish.

## 3. Select Quant And Slices

`quant` describes how LARQL stores the extracted vindex. The current catalogue
uses:

```yaml
quant: q4k
```

`slices` describes the vindex shape:

- `full`: publish the complete vindex
- `expert-server`: publish an MoE expert-server slice for CPU/high-memory
  weight serving

`full` cannot be combined with another slice in the same entry.

## 4. Assign The Tier

The tier controls how operators select groups of entries.

Use `smoke` for smaller vindexes that are appropriate for first publication
tests:

```yaml
tier: smoke
```

Use `moe` for larger MoE vindexes that should remain manual until the runner
has enough disk, memory, and network capacity. These are usually the entries
most relevant to keeping expert weights out of expensive GPU memory.

## 5. Set Output And Repository Names

`output_name` is the local directory LARQL writes under scratch storage. It must
end in `.vindex` and must not include a slash.

`hf_repo` is the target Hugging Face repository:

```yaml
output_name: llama-3-2-3b-instruct-full-q4-k.vindex
hf_repo: skulk/llama-3-2-3b-instruct-full-q4-k-vindex
```

## 6. Validate And Dry-Run

After editing `models.yaml`, run:

```bash
skulk-vindex manifest validate
skulk-vindex manifest get --key llama-3-2-3b-full-q4-k
skulk-vindex publish --model llama-3-2-3b-full-q4-k --dry-run
```

Commit the manifest change only after the dry-run command matches the vindex
you intend to build, publish, and place in Skulk.
