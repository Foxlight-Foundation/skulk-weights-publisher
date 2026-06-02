---
title: Publishing Lifecycle
---

A vindex goes through a build-and-release lifecycle because it is the boundary
between expensive GPU inference nodes and cheaper CPU/high-memory weight
servers. The goal is a stable, published model representation that every Skulk
node can agree on before runtime placement begins.

## 1. Describe The Vindex

The vindex starts as a catalog source entry. The built-in Foxlight entries
are packaged with the CLI, and operator entries can be added through
`skulk-weights.yaml`. Each entry names the source model, quantization, slice
mode, local `.vindex` directory, and target Hugging Face repository. The slice
mode is part of the runtime contract: it tells operators whether they are
publishing a complete vindex or a specialized expert-server shape for weight
serving.

## 2. Validate The Catalog

```bash
skulk-weights catalog validate
```

Validation catches duplicate keys, unsupported slice names, bad repository
names, and output names that would not be safe to write.

## 3. Check The Runner

```bash
skulk-weights doctor --publish
```

The publishing runner needs Python, LARQL, writable scratch storage, network
access to Hugging Face, and `HF_TOKEN`. It does not have to be the eventual
runtime host; it is the machine that performs the expensive extraction and
upload.

## 4. Review The Plan

```bash
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

The dry-run prints the exact `larql extract` and `larql publish` commands. This
is the last cheap place to catch a wrong source model, path, slice mode, or
repository before disk-heavy extraction begins.

`publish` builds whichever artifacts you ask for through `--artifact`. The
values are:

- `vindex` — the LARQL retrieval index (the default subject of this page).
- `mtp` — the Multi-Token Prediction sidecar (step 7).
- `vision` — the vision-encoder sidecar mirror (step 8).
- `all` — every artifact configured on the catalog entry.

## 5. Extract The Vindex

Real publication starts by running `larql extract`. This can use substantial
scratch disk because it creates a local vindex directory before anything is
uploaded.

## 6. Publish The Vindex

After extraction, the publisher runs `larql publish` and uploads the vindex to
the Hugging Face repository in the catalog entry.

## 7. Upload The Self-Describing Model Card

Every real publish—vindex, MTP, or vision—also uploads a `README.md` model card
to the published repository. The card is self-describing so the artifact carries
its own provenance instead of relying on external records.

The frontmatter sets `base_model` to the source repo, tags the artifact
(`[artifact_type, skulk, foxlight, quant]`), and inherits the source model's
license unchanged (custom licenses also carry `license_name`/`license_link`). It
also embeds a `foxlight:` provenance block: artifact type, source repo, the
pinned `source_revision` commit SHA, target model, quant, catalog key, the tool
that extracted it, and a timestamp. The body explains what the artifact is, a
provenance table, usage, and a license note.

The source commit SHA and license are resolved best-effort from the Hub using
`HF_TOKEN`. Published artifacts inherit the source model's license unchanged and
are never re-licensed—everything published is for the community.

## 8. Extract And Publish The MTP Sidecar (optional)

For models that carry native Multi-Token Prediction heads—Qwen3, DeepSeek V3/R1,
and similar architectures—a second extraction pass pulls the `mtp.*` tensors from
the original BF16 checkpoint and uploads them at full precision (bf16,
unquantized) as `mtp.safetensors` to a separate sidecar repository — one sidecar
per base model, shared across every quantization of it.

This step is separate from vindex publication because the MTP weights must come from
the original PyTorch checkpoint. mlx-lm's `sanitize()` strips `mtp.*` keys during
conversion, so the mlx-converted source used for vindex extraction does not contain
them.

```bash
skulk-weights publish --model acme/qwen3-6b-full-q4-k --artifact mtp --dry-run
skulk-weights publish --model acme/qwen3-6b-full-q4-k --artifact mtp
```

The dry-run prints the source repo, sidecar repo, precision, and output path
before any download begins. Real execution downloads only the shards that contain
`mtp.*` keys (using the model's `model.safetensors.index.json` to identify them),
saves the tensors at full precision (bf16, unquantized) as a local
`mtp.safetensors`, and uploads it to the sidecar repo. One bf16 sidecar serves
every quantization of the base model.

If `mtp_source_repo` and `mtp_sidecar_repo` are not set on the catalog entry,
`--artifact mtp` raises an error with a clear message rather than silently
skipping.

See the [MTP Sidecar Guide](../guides/mtp-sidecar.md) for a complete walkthrough.

## 9. Mirror And Publish The Vision Sidecar (optional)

Some vision-language models ship an mlx-community checkpoint that omits the
vision encoder—Kimi K2.5, for example, keeps its vision weights in a third-party
repository. For those models SWP publishes a vision sidecar: a Foxlight-owned
mirror so the cluster does not depend on a third party.

Unlike the MTP step, the vision sidecar performs no quantization and no dtype
conversion. It copies the `vision_source_repo`'s weights and configs into
`vision_sidecar_repo` **byte-for-byte**. It needs `huggingface_hub` but not mlx.

```bash
skulk-weights publish --model acme/kimi-k2-5-full-q4-k --artifact vision --dry-run
skulk-weights publish --model acme/kimi-k2-5-full-q4-k --artifact vision
```

If `vision_source_repo` and `vision_sidecar_repo` are not set on the catalog
entry, `--artifact vision` raises an error with a clear message rather than
silently skipping.

## 10. File Into A Collection

Each successful publish is filed into the Hugging Face collection for its
artifact type: `Vindexes`, `MTP Sidecars`, or `Vision Sidecars`. The sidecar
collections are resolved by title—created if missing, reused if they already
exist—so a delete-and-republish stays in the right collection. The vindex is
filed into the configured slug exactly (the catalog `hf_collection` or
`SKULK_WEIGHTS_COLLECTION`).

Collection filing is disabled when no collection is configured, or when
`SKULK_WEIGHTS_COLLECTION` is set to one of `none`, `0`, `false`, `no`, `off`,
or `disabled`.

## 11. Use The Published Weights

Once published, the vindex and any sidecar have stable repository names. Skulk
operators can use those names when assigning GPU nodes to the latency-sensitive
inference path and CPU/high-memory LARQL servers to FFN or expert weight serving.
Skulk loads the MTP sidecar at inference time when MTP is enabled for a given
deployment.
