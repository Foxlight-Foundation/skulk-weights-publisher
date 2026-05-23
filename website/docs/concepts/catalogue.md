---
title: The Catalogue
---

The catalogue is `models.yaml`. It is the list of vindex artifacts this project
knows how to build and publish.

Each row answers one question: "If an operator asks for this vindex, exactly
what should be built and where should it go?"

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
- `quant`: the quantization LARQL uses when building the artifact
- `tier`: whether this is a small first-test artifact or a larger manual target
- `slices`: the artifact shape LARQL should publish
- `output_name`: the local vindex directory name created under scratch storage
- `hf_repo`: the Hugging Face repository that receives the published artifact

## Tiers

The `smoke` tier is for the first practical publish path. These entries are
small enough to validate runner setup and publication behavior.

The `moe` tier is for larger mixture-of-experts targets. These need more disk,
more network time, and more operator attention.

## Slice Modes

`full` means "publish the whole vindex artifact."

`expert-server` is a specialized slice used for MoE expert-server publication.
It gets its own catalogue entry so the published repository name, output
directory, and workflow selection stay explicit.

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
commands you intended.
