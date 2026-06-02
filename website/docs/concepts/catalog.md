---
title: The Catalog
---

The catalog is the library of vindexes the publisher knows how to build for
Skulk. Each entry answers one operational question: if an operator selects this
key, exactly which Hugging Face model should LARQL extract, which vindex shape
should be published, what local output should be written, and which Hugging
Face repository should receive it?

## Built-In Foxlight Catalog

The Foxlight catalog is packaged with the CLI and loaded automatically. Those
entries publish to the `FoxlightAI` Hugging Face organization and are added to
the public
[`Vindexes`](https://huggingface.co/collections/FoxlightAI/vindexes-6a124406dd5fb439c431b051)
collection. You do not need a local config file to inspect or dry-run them:

```bash
skulk-weights catalog validate
skulk-weights catalog list --tier smoke
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

Foxlight entries use the `foxlight/` namespace in the merged catalog. The
source file stores the short key, and the catalog layer exposes the effective
key:

```yaml
models:
  - key: gemma-3-4b-full-q4-k
    source_model: google/gemma-3-4b-it
    quant: q4k
    tier: smoke
    slices:
      - full
    output_name: gemma-3-4b-it-full-q4-k.vindex
    hf_repo: FoxlightAI/gemma-3-4b-it-full-q4-k-vindex
    hf_collection: FoxlightAI/vindexes-6a124406dd5fb439c431b051
```

Effective CLI key:

```text
foxlight/gemma-3-4b-full-q4-k
```

## Operator Catalogs

Operators can add their own catalog sources without forking the Foxlight
catalog. Create `skulk-weights.yaml`:

```bash
skulk-weights catalog init
```

Then point it at an operator-owned manifest file:

```yaml
catalogs:
  - path: ./operator-vindexes.yaml
    namespace: my-org
    hf_owner: my-org
    hf_collection: my-org/vindexes-0123456789abcdef01234567
```

`namespace` controls the effective CLI key. `hf_owner` controls which Hugging
Face account or organization the entries are allowed to publish into.
`hf_collection` is optional; when set, successful publishes from that source are
added to the collection after `larql publish` finishes. With the config above,
a source entry with `key: llama-3-8b-full-q4-k` becomes:

```text
my-org/llama-3-8b-full-q4-k
```

The Foxlight catalog is still included automatically when you pass the config.

## Fields In Plain Language

- `key`: the short source key before the namespace is added
- `source_model`: the Hugging Face model LARQL reads from
- `quant`: the quantization LARQL uses when extracting the vindex
- `tier`: whether this is a small first-test vindex or a larger manual target
- `slices`: the vindex shape LARQL should publish for the intended runtime role
- `output_name`: the local vindex directory name created under scratch storage
- `hf_repo`: the Hugging Face repository that receives the published vindex
- `hf_collection`: the Hugging Face collection that receives the published vindex
- `mtp_source_repo`: the BF16 checkpoint the MTP heads are extracted from
- `mtp_sidecar_repo`: the repository that receives the published bf16
  `mtp.safetensors` (one sidecar per base model, shared across all quantizations)
- `assistant_model_repo`: the companion `{model}-assistant` drafter model (a
  pointer to a Google-published model; no tensors are extracted for it)
- `vision_source_repo`: the third-party repository whose vision weights are mirrored
- `vision_sidecar_repo`: the Foxlight-owned repository that receives the byte-for-byte mirror

Every real publish (vindex, MTP, or vision) also uploads a self-describing
`README.md` model card carrying its provenance and the source model's
license—see [the publishing lifecycle](publishing-lifecycle.md). Each published
repo is filed into the Hugging Face collection for its artifact type
(`Vindexes`, `MTP Sidecars`, `Vision Sidecars`); the vindex collection is the
configured `hf_collection` slug.

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
It gets its own catalog entry so the published repository name, output
directory, and workflow selection stay explicit for CPU/high-memory expert
serving.

The CLI validates the merged catalog before any publish command is planned:

```bash
skulk-weights --config skulk-weights.yaml catalog validate
skulk-weights --config skulk-weights.yaml catalog show my-org/llama-3-8b-full-q4-k
```

When you change a source file, run a dry-run before you commit:

```bash
skulk-weights --config skulk-weights.yaml publish \
  --model my-org/llama-3-8b-full-q4-k \
  --dry-run
```

The dry-run is how you confirm that the catalog entry produces the LARQL
commands and runtime shape you intended.
