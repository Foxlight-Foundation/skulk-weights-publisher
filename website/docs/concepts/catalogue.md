---
title: The Catalogue
---

The catalogue is `models.yaml`. It is the list of vindexes this project knows
how to build and publish for Skulk.

Each row answers one question: "If an operator asks for this vindex, exactly
what should be built, where should it go, and what runtime role is it meant to
support?"

Example:

```yaml
models:
  - key: gemma-3-4b-full-q4-k
    source_model: google/gemma-3-4b-it
    quant: q4k
    tier: smoke
    slices:
      - full
    output_name: gemma-3-4b-it-full-q4-k.vindex
    hf_repo: skulk/gemma-3-4b-it-full-q4-k-vindex
```

## Fields In Plain Language

- `key`: the short name operators type in CLI commands and workflow dispatch
- `source_model`: the Hugging Face model LARQL reads from
- `quant`: the quantization LARQL uses when extracting the vindex
- `tier`: whether this is a small first-test vindex or a larger manual target
- `slices`: the vindex shape LARQL should publish for the intended runtime role
- `output_name`: the local vindex directory name created under scratch storage
- `hf_repo`: the Hugging Face repository that receives the published vindex

## Tiers

The `smoke` tier is for the first practical publish path. These entries are
small enough to validate runner setup and publication behavior.

The `moe` tier is for larger mixture-of-experts targets. These are the entries
most directly tied to the cost goal: expert weights are large and can be served
from CPU/high-memory machines instead of forcing every weight-heavy role onto
GPU memory. They need more disk, more network time, and more operator attention.

## Slice Modes

`full` means "publish the whole vindex." Use it when the full model
representation should be available under one repository.

`expert-server` is a specialized slice used for MoE expert-server publication.
It gets its own catalogue entry so the published repository name, output
directory, and workflow selection stay explicit for CPU/high-memory expert
serving.

The CLI validates the catalogue before any publish command is planned.

```bash
skulk-vindex manifest validate
skulk-vindex manifest get --key gemma-3-4b-full-q4-k
```

When you change `models.yaml`, run a dry-run before you commit:

```bash
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

The dry-run is how you confirm that the catalogue entry produces the LARQL
commands and runtime shape you intended.
