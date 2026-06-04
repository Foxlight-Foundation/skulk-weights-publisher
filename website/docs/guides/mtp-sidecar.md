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

Extraction is **pure-numpy and cross-platform** — it requires only `numpy`,
`safetensors`, and `huggingface_hub`, with no `mlx` dependency, so it runs the
same on Linux/x86 as on macOS.

The heads are **not** quantized. They are the speculative *drafter*, and their
only job is to maximize draft *acceptance*—the entire point of the feature.
Quantizing them would degrade acceptance to save only tens of MB on a multi-GB
model, a bad trade. The heads are also independent of the target model's
quantization: the Skulk runtime loads the sidecar through its own path, so
**one bf16 sidecar serves every quantization of the base model**. There is one
sidecar per base model, quant-independent.

## Prerequisites

The same prerequisites as regular vindex publishing apply, plus the `mtp`
extras (`numpy`, `safetensors`, on top of the base `huggingface_hub`):

```bash
uv sync --extra mtp
```

- `huggingface_hub` Python package (base dependency)
- `numpy` Python package (FP8/INT8 decode and BF16 encode)
- `safetensors` Python package (single-file checkpoint inspection)
- Read access to the source BF16 checkpoint on Hugging Face
- Write access to the sidecar repository on Hugging Face

There is no `mlx` requirement and no platform restriction — extraction is
pure-numpy and runs on Linux/x86 as well as macOS.

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
3. Reads the MTP tensors out of each shard and dequantises any quantised formats
   to BF16 on the fly (see [Dequantisation](#dequantisation) below). BF16, F16,
   and F32 tensors are converted to BF16 directly.
4. Streams the result into a single `mtp.safetensors` file at full precision
   (bf16, unquantized), **one tensor at a time** (see
   [Streaming extraction](#streaming-extraction) below). Preserving fidelity is
   what keeps draft acceptance high.
5. Uploads it to `mtp_sidecar_repo` on Hugging Face, emitting byte-level progress
   lines during the LFS transfer (see [Upload progress](#upload-progress) below),
   alongside a self-describing `README.md` model card. The card inherits the
   source model's license unchanged and carries a Foxlight provenance block
   pinning the source SHA.
6. Files the repo into the `MTP Sidecars` Hugging Face collection (unless
   `SKULK_WEIGHTS_COLLECTION` is set to a disabling value).
7. Deletes the local `.safetensors` file and shard cache from scratch. Skulk
   owns the artifact lifecycle; SWP's job ends when the push completes.

### Already-published skip

Because one bf16 sidecar covers every quantization of a base model, extraction
is **skipped** when `mtp.safetensors` already exists on the sidecar repo. The
extractor checks the Hub up front and, if the file is present, prints a message
that the sidecar already covers the source model and exits without
re-downloading or re-uploading. Pass `--force` to re-extract and overwrite it.
The check is best-effort: any lookup failure (repo absent, no network) is
treated as "not published" so extraction proceeds normally.

### Streaming extraction

Tensors are streamed from the source shards into the output file **one at a
time**, so peak memory is bounded to a single tensor regardless of how many
tensors there are or how large the source shards are. Shards are read via
memory-friendly per-tensor access (the extractor seeks to each tensor's byte
range in the shard rather than loading the whole file), and the output
`mtp.safetensors` is written incrementally. This is what makes extraction safe
for very large models (DeepSeek V4-Pro, V3, etc.) where accumulating every
dequantised tensor in a dict before saving would need tens of GB of RAM.

### Upload progress

During the LFS upload pass the extractor emits byte-level progress lines so the
CLI and GUI can track the real network transfer:

```
mtp: uploading 12% (1.3 GB / 10.6 GB)
mtp: uploading 14% (1.5 GB / 10.6 GB)
```

Progress is armed only on the actual upload pass, not the fast CPU hash pass
that precedes it, so the percentage reflects bytes pushed to Hugging Face.

### Dequantisation

The extractor handles a broad range of quantised source dtypes, always producing
BF16 output:

- **FP8 E4M3** weights, paired with a block scale stored as **F8 E8M0**, **F32**
  `_scale_inv` (DeepSeek V3 — the stored value is `1/scale`, so the reciprocal is
  taken), or **BF16**.
- **INT8** weights, paired with an **F8 E8M0** block scale.
- **BF16 / F16 / F32** tensors pass straight through (converted to BF16).

Block scales are applied in one of two layouts:

- **2D MX tiled** — for a 2D weight `[R, C]`, one scale per `B×B` tile, with
  `ceil(R/B) × ceil(C/B)` scales. The tile size `B` is inferred from the scale
  count (tried in order 128, 64, 256, 32); partial right/bottom-edge tiles are
  zero-padded to a full block boundary, scaled, then stripped. This 2D layout is
  checked first so axis-aligned tile boundaries are respected even when the flat
  layout would also divide evenly.
- **1D flat** — `n_weight` divisible by the scale count, with
  `block_size = n_weight / n_scale`. Used by DeepSeek V3 (`_scale_inv`) and
  V4-Flash/Pro (`.scale`), or as the fallback when no 2D tiling matches.

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
```

Install the extras with `uv sync --extra mtp` (`numpy` + `safetensors`, on top
of the base `huggingface_hub`). `safetensors` is needed only for the single-file
checkpoint fallback; sharded checkpoints inspect their index directly.

## Scratch Storage

Sidecar output and the shard cache are created in `SKULK_WEIGHTS_SCRATCH`
(or `.scratch/` by default) during extraction, then **deleted automatically
after a successful upload**. Skulk owns the artifact lifecycle; SWP does not
retain local copies.

If extraction fails before the upload completes, the scratch files are left in
place for inspection and retry. Use `skulk-weights scratch clean` to remove
them manually if needed.
