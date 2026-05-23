---
title: Add A Catalogue Entry
---

Use this guide when you want to add a new publishable vindex.

A catalogue entry should be boring and specific. It should tell a future
operator exactly which model is being transformed, how LARQL will prepare it,
what local directory will be created, and where the artifact will be published.

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

`quant` describes how LARQL prepares the artifact. The current catalogue uses:

```yaml
quant: q4k
```

`slices` describes the artifact shape:

- `full`: publish the complete vindex artifact
- `expert-server`: publish an MoE expert-server slice

`full` cannot be combined with another slice in the same entry.

## 4. Assign The Tier

The tier controls how operators select groups of entries.

Use `smoke` for smaller artifacts that are appropriate for first publication
tests:

```yaml
tier: smoke
```

Use `moe` for larger MoE artifacts that should remain manual until the runner
has enough disk, memory, and network capacity.

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

Commit the manifest change only after the dry-run command matches the artifact
you intend to build and publish.
