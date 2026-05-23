---
title: Add A Catalogue Entry
---

Use this guide when you want the publisher to know about a new vindex artifact.

## 1. Choose The Stable Key

The `key` is what operators type in CLI commands and workflow dispatch. Keep it
lowercase, descriptive, and stable:

```yaml
key: llama-3-2-3b-full-q4-k
```

Do not change an existing key after it has appeared in automation or docs.

## 2. Pick The Source Model

Set `source_model` to the upstream Hugging Face model ID consumed by LARQL:

```yaml
source_model: meta-llama/Llama-3.2-3B-Instruct
```

If the upstream model is gated, make sure the publishing token has accepted the
model terms before running a real publish.

## 3. Select Quant And Slices

The current catalogue supports `q4k` quantization and these slice modes:

- `full`: publish the complete vindex artifact
- `expert-server`: publish an MoE expert-server slice for delegation testing

`full` cannot be combined with another slice in the same entry.

## 4. Assign The Tier

Use `smoke` for small artifacts that can run in scheduled validation and early
publication tests.

Use `moe` for larger MoE artifacts that should remain manual until the runner
has enough disk, memory, and network capacity.

## 5. Set Output And Repository Names

`output_name` is the local scratch directory basename. It must end in
`.vindex` and must not include a slash.

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
