---
title: MTP Sidecar Guide
---

Multi-Token Prediction (MTP) heads allow a model to draft multiple tokens in a
single forward pass, which Skulk uses to accelerate speculative decoding. Some
model families ship native MTP heads baked into the base checkpoint:

- **Qwen3, DeepSeek V4-Flash, and similar**: heads stored as `mtp.*` tensor keys.
- **DeepSeek V3 / V3-0324**: heads stored as `model.layers.{num_hidden_layers}.*`
  (one extra transformer layer beyond the main stack); SWP detects this
  automatically by reading `config.json`.

mlx-lm's `sanitize()` strips MTP tensors during conversion, so the converted
checkpoint used for vindex publication does not contain them. The MTP sidecar
step re-extracts them from the original checkpoint and publishes them at
**full precision (bf16, unquantized)** as `mtp.safetensors` to a separate HF
repository. Tensors stored in quantised formats (FP8/INT8) are dequantised to
BF16 during extraction.

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

1. Fetches `model.safetensors.index.json` from `mtp_source_repo` and scans the
   `weight_map` for MTP tensors using two detection strategies:
   - **New-style**: looks for `mtp.*` or `.mtp.*` keys directly.
   - **Old-style** (DeepSeek V3/V3-0324): if no `mtp.*` keys are found, fetches
     `config.json` and checks `num_nextn_predict_layers > 0`; if set, treats
     `model.layers.{num_hidden_layers}.*` as the MTP layer.
   For single-file checkpoints it falls back to checking `model.safetensors` directly
   (new-style keys only).
2. Downloads only those shards into the scratch directory (`.scratch/_hf_cache/`
   by default). Large models ship dozens of shards; the extractor avoids downloading
   the entire 60–70 GB by filtering on the index.
3. Reads the MTP tensors out of each shard. Tensors in quantised formats (FP8 E4M3
   with E8M0 or F32 block scales, INT8 with E8M0 scales) are dequantised to BF16
   on the fly. BF16, F16, and F32 tensors are passed through directly.
4. Saves the result as a single `mtp.safetensors` file at full precision (bf16,
   unquantized). Preserving fidelity is what keeps draft acceptance high.
5. Uploads it to `mtp_sidecar_repo` on Hugging Face, alongside a self-describing
   `README.md` model card. The card inherits the source model's license
   unchanged and carries a Foxlight provenance block pinning the source SHA.
6. Files the repo into the `MTP Sidecars` Hugging Face collection (unless
   `SKULK_WEIGHTS_COLLECTION` is set to a disabling value).
7. Deletes the local `.safetensors` file and shard cache from scratch. Skulk
   owns the artifact lifecycle; SWP's job ends when the push completes.

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

**No MTP head tensors found in source repo:**

```
no MTP head tensors found in Qwen/Qwen3-6B; confirm this model has native MTP heads
(checked for mtp.* keys and model.layers.{N}.* via config.json)
```

This happens when `mtp_source_repo` points to an mlx-converted checkpoint instead
of the original BF16/FP8 release (mlx-lm strips MTP tensors during conversion),
or when the model genuinely does not have MTP heads.

**Missing Python dependencies:**

```
huggingface_hub is required for MTP extraction
safetensors is required for single-file model inspection
mlx is required for reading MTP weights
```

Install the extras with `uv sync --extra mtp`.

## Scratch Storage

Sidecar output and the shard cache are created in `SKULK_WEIGHTS_SCRATCH`
(or `.scratch/` by default) during extraction, then **deleted automatically
after a successful upload**. Skulk owns the artifact lifecycle; SWP does not
retain local copies.

If extraction fails before the upload completes, the scratch files are left in
place for inspection and retry. Use `skulk-weights scratch clean` to remove
them manually if needed.
