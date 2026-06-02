---
title: MTP Sidecar Guide
---

Multi-Token Prediction (MTP) heads allow a model to draft multiple tokens in a
single forward pass, which Skulk uses to accelerate speculative decoding. Some
model families—Qwen3, DeepSeek V3/R1, and similar architectures—ship native
MTP heads as `mtp.*` tensor keys baked into the BF16 checkpoint.

mlx-lm's `sanitize()` strips `mtp.*` keys during conversion, so the converted
checkpoint used for vindex publication does not contain them. The MTP sidecar
step re-extracts them from the original BF16 checkpoint and publishes them at
**full precision (bf16, unquantized)** as `mtp.safetensors` to a separate HF
repository.

The heads are **not** quantized. They are the speculative *drafter*, and their
only job is to maximize draft *acceptance*—the entire point of the feature.
Quantizing them would degrade acceptance to save only tens of MB on a multi-GB
model, a bad trade. The heads are also independent of the target model's
quantization: the Skulk runtime loads the sidecar through its own path, so
**one bf16 sidecar serves every quantization of the base model**. There is one
sidecar per base model, quant-independent.

## Prerequisites

The same prerequisites as regular vindex publishing apply, plus the `mtp`
extras (`huggingface_hub`, `safetensors`, `mlx`):

```bash
uv sync --extra mtp
```

- `huggingface_hub` Python package
- `safetensors` Python package
- `mlx` Python package (used for safetensors I/O)
- Read access to the source BF16 checkpoint on Hugging Face
- Write access to the sidecar repository on Hugging Face

## Catalog Entry

Add two fields to the catalog entry. Both must be present or both must be
absent:

```yaml
models:
  - key: qwen3-6b-full-q4-k
    source_model: acme/qwen3-6b-mlx-q4k
    quant: q4k
    tier: smoke
    slices:
      - full
    output_name: qwen3-6b-full-q4-k.vindex
    hf_repo: acme/qwen3-6b-full-q4-k-vindex
    mtp_source_repo: Qwen/Qwen3-6B       # original BF16 checkpoint
    mtp_sidecar_repo: acme/qwen3-6b-mtp  # where mtp.safetensors lands
```

`mtp_source_repo` is typically the canonical model owner's BF16 release, not an
mlx-converted community checkpoint. For Qwen3 models, this is the `Qwen/` namespace
release. For DeepSeek V3/R1, this is the `deepseek-ai/` namespace release.

The sidecar repo carries no quant suffix. Because the bf16 sidecar is shared
across every quantization of the base model, the name is `<owner>/<base-slug>-mtp`
(here `acme/qwen3-6b-mtp`)—there is one sidecar per base model, not one per
quant tier. Every vindex entry for the same base model points at the same
`mtp_sidecar_repo`.

## Dry Run

Preview what the step will do without downloading anything:

```bash
uv run skulk-weights publish --model acme/qwen3-6b-full-q4-k --artifact mtp --dry-run
```

Output:

```
mtp source repo:  hf://Qwen/Qwen3-6B
mtp sidecar repo: hf://acme/qwen3-6b-mtp/mtp.safetensors
mtp precision:    bf16 (unquantized)
mtp output path:  /path/to/scratch/acme--qwen3-6b-mtp-mtp.safetensors
```

## Extract And Publish

```bash
uv run skulk-weights publish \
  --model acme/qwen3-6b-full-q4-k \
  --artifact mtp
```

The extractor:

1. Fetches `model.safetensors.index.json` from `mtp_source_repo` and reads the
   `weight_map` to identify which shard files contain `mtp.*` keys. For single-file
   checkpoints it falls back to checking `model.safetensors` directly.
2. Downloads only those shards into the scratch directory (`.scratch/_hf_cache/`
   by default). Large models ship dozens of shards; the extractor avoids downloading
   the entire 60–70 GB by filtering on the index.
3. Reads all tensors whose key starts with `mtp.` or contains `.mtp.`.
4. Saves the heads at full precision (bf16, unquantized). Nothing is quantized:
   preserving the drafter's fidelity is what keeps draft acceptance high.
5. Saves the result as a single `mtp.safetensors` file in scratch storage.
6. Uploads it to `mtp_sidecar_repo` on Hugging Face, alongside a self-describing
   `README.md` model card. The card inherits the source model's license
   unchanged and carries a Foxlight provenance block pinning the source SHA.
7. Files the repo into the `MTP Sidecars` Hugging Face collection (unless
   `SKULK_WEIGHTS_COLLECTION` is set to a disabling value).

## Publishing Only The MTP Sidecar

The `--artifact mtp` flag publishes the MTP sidecar without re-running vindex
extraction or publication. This is useful when the vindex already exists on HF
and you only need to add the sidecar:

```bash
# vindex already published, add the MTP sidecar
uv run skulk-weights publish --model acme/qwen3-6b-full-q4-k --artifact mtp
```

Compare with:

```bash
# publish every configured artifact for the entry in one run
uv run skulk-weights publish --model acme/qwen3-6b-full-q4-k --artifact all
```

## Error Cases

**Entry has no MTP fields:**

```
PublishError: no MTP sidecar configured for acme/qwen3-6b-full-q4-k;
add mtp_source_repo and mtp_sidecar_repo to the catalog entry
```

**No `mtp.*` keys found in source repo:**

```
no mtp.* keys found in Qwen/Qwen3-6B; confirm this model has native MTP heads
```

This happens when `mtp_source_repo` points to an mlx-converted checkpoint instead
of the original BF16 release, or when the model genuinely does not have MTP heads.

**Missing Python dependencies:**

```
huggingface_hub is required for MTP extraction
safetensors is required for single-file model inspection
mlx is required for MTP safetensors I/O
```

Install the extras with `uv sync --extra mtp`.

## Scratch Storage

Sidecar output lands in `SKULK_WEIGHTS_SCRATCH` (or `.scratch/` by default) as:

```
<sidecar_repo_with_slashes_replaced_by_dashes>-mtp.safetensors
```

For `mtp_sidecar_repo: acme/qwen3-6b-mtp` that is:

```
acme--qwen3-6b-mtp-mtp.safetensors
```

The file is not removed after upload. Delete it manually if scratch disk is
constrained.
