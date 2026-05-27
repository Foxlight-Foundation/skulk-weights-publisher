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

## 5. Extract The Vindex

Real publication starts by running `larql extract`. This can use substantial
scratch disk because it creates a local vindex directory before anything is
uploaded.

## 6. Publish The Vindex

After extraction, the publisher runs `larql publish` and uploads the vindex to
the Hugging Face repository in the catalog entry.

## 7. Extract And Publish The MTP Sidecar (optional)

For models that carry native Multi-Token Prediction heads—Qwen3, DeepSeek V3/R1,
and similar architectures—a second extraction pass pulls the `mtp.*` tensors from
the original BF16 checkpoint, quantizes them, and uploads them as `mtp.safetensors`
to a separate sidecar repository.

This step is separate from vindex publication because the MTP weights must come from
the original PyTorch checkpoint. mlx-lm's `sanitize()` strips `mtp.*` keys during
conversion, so the mlx-converted source used for vindex extraction does not contain
them.

```bash
skulk-weights publish --model acme/qwen3-6b-full-q4-k --artifact mtp --dry-run
skulk-weights publish --model acme/qwen3-6b-full-q4-k --artifact mtp
```

The dry-run prints the source repo, sidecar repo, quant, and output path before
any download begins. Real execution downloads only the shards that contain `mtp.*`
keys (using the model's `model.safetensors.index.json` to identify them), quantizes
the tensors, saves a local `mtp.safetensors`, and uploads it to the sidecar repo.

If `mtp_source_repo`, `mtp_sidecar_repo`, and `mtp_quant` are not set on the catalog
entry, `--artifact mtp` raises an error with a clear message rather than silently
skipping.

See the [MTP Sidecar Guide](../guides/mtp-sidecar.md) for a complete walkthrough.

## 8. Use The Published Weights

Once published, the vindex and any sidecar have stable repository names. Skulk
operators can use those names when assigning GPU nodes to the latency-sensitive
inference path and CPU/high-memory LARQL servers to FFN or expert weight serving.
Skulk loads the MTP sidecar at inference time when MTP is enabled for a given
deployment.
