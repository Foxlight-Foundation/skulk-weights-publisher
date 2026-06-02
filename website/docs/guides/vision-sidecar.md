---
title: Vision Sidecar Guide
---

A vision sidecar publishes the vision-encoder weights of a vision-language model
(VLM) so the multimodal path has no external dependency. mlx-community VLM quants
frequently ship only the language tower and omit the vision encoder, leaving the
encoder weights to live in a third-party repository. Kimi K2.5 is a typical case:
its vision weights are hosted in a separate, non-Foxlight repo.

The vision sidecar step mirrors those encoder weights **byte-for-byte** into a
Foxlight-owned repository. There is no quantization and no dtype conversion — the
mirror is an exact copy — so the Skulk multimodal path can load the encoder
without reaching out to an upstream third party.

## Prerequisites

The same prerequisites as regular vindex publishing apply, plus `huggingface_hub`
(part of the `mtp` extras):

```bash
uv sync --extra mtp
```

- `huggingface_hub` Python package
- Read access to the source vision-encoder repository on Hugging Face
- Write access to the sidecar repository on Hugging Face

Unlike the MTP sidecar, **no `mlx` is required** — there is no quantization, so a
vision sidecar can be published from any platform with `huggingface_hub`.

## Catalog Entry

Add two fields to the catalog entry. Both must be present or both must be absent:

```yaml
models:
  - key: kimi-k2-5-full-q4-k
    source_model: mlx-community/Kimi-K2.5-4bit
    quant: q4k
    tier: moe
    slices:
      - full
    output_name: kimi-k2-5-full-q4-k.vindex
    hf_repo: FoxlightAI/kimi-k2-5-full-q4-k-vindex
    vision_source_repo: third-party/kimi-k2-5-vision   # where the encoder weights live
    vision_sidecar_repo: FoxlightAI/kimi-k2-5-vision    # Foxlight-owned mirror
```

`vision_source_repo` is the repository that actually holds the vision-encoder
`.safetensors` weights — often a third party, not the mlx-community quant (which
is exactly what the sidecar removes the dependency on).

`vision_sidecar_repo` is the mirror's destination. Its owner **must equal the
owner of `hf_repo`** so the encoder lands in the same namespace as the vindex.
The conventional name is `<source-slug>-vision`.

## Dry Run

Preview what the step will do without downloading anything. `--model` takes the
**catalog key** (e.g. `foxlight/kimi-k2-5-full-q4-k`), not a HuggingFace repo
name:

```bash
uv run skulk-weights publish --model foxlight/kimi-k2-5-full-q4-k --artifact vision --dry-run
```

Output:

```
vision source repo:  hf://third-party/kimi-k2-5-vision
vision sidecar repo: hf://FoxlightAI/kimi-k2-5-vision
```

## Mirror And Publish

```bash
uv run skulk-weights publish \
  --model foxlight/kimi-k2-5-full-q4-k \
  --artifact vision
```

The mirror step:

1. Resolves the `.safetensors` weights in `vision_source_repo`.
2. Copies them **byte-for-byte** into `vision_sidecar_repo` — no quantization,
   no dtype conversion. The mirror is an exact copy of the source.
3. Prunes stale files on republish, so the sidecar repo never accumulates
   leftovers from a previous mirror.
4. Uploads a self-describing `README.md` model card alongside the weights. The
   card inherits the source model's license unchanged and carries a Foxlight
   provenance block pinning the source SHA.
5. Files the repo into the `Vision Sidecars` Hugging Face collection (unless
   `SKULK_WEIGHTS_COLLECTION` is set to a disabling value).

## Publishing Only The Vision Sidecar

The `--artifact vision` flag mirrors the encoder without re-running vindex
extraction or publication. This is useful when the vindex already exists and you
only need to add the vision sidecar:

```bash
# vindex already published, add the vision sidecar
uv run skulk-weights publish --model foxlight/kimi-k2-5-full-q4-k --artifact vision
```

Compare with:

```bash
# publish every configured artifact for the entry in one run
uv run skulk-weights publish --model foxlight/kimi-k2-5-full-q4-k --artifact all
```

## Error Cases

**Entry has no vision fields:**

```
no vision sidecar configured for FoxlightAI/kimi-k2-5-full-q4-k...
```

Add `vision_source_repo` and `vision_sidecar_repo` to the catalog entry, or use
`--artifact vindex` to publish only the vindex. In dry-run mode the publisher
instead reports that the vision step is not configured and exits cleanly.

**No weights in the source repo:**

```
no .safetensors weights found in third-party/kimi-k2-5-vision
```

The `vision_source_repo` resolved but contains no `.safetensors` files to mirror.
Confirm the repo actually holds the vision-encoder weights and update
`vision_source_repo` accordingly.

## See Also

- [MTP Sidecar Guide](mtp-sidecar.md) — the sibling artifact for models with
  native multi-token prediction heads.
- [Add A Catalog Entry](add-catalog-entry.md) — where the optional sidecar
  fields fit into the full entry schema.
