---
title: First Publish
---

Your first real publish should prove the whole path with one small vindex:
catalog entry, runner, LARQL, scratch storage, Hugging Face token, upload,
and collection update.

Use a `smoke` entry first. It gives you the same workflow shape as larger
vindexes with less disk and time risk. Once the path works, larger full and
expert-server entries can be published for the real cost goal: keeping
weight-heavy model state off expensive GPU memory where CPU/high-memory LARQL
servers can host it.

## 1. Validate The Catalog

```bash
skulk-weights catalog validate
```

This proves the effective catalog is structurally safe before the runner
starts.

## 2. Check Publication Prerequisites

```bash
skulk-weights doctor --publish
```

This checks the pieces needed for a real publish: LARQL, `HF_TOKEN`, scratch
storage, Python dependencies, and the catalog.

## 3. Review The Dry-Run

```bash
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

Read the source model, output path, target repository, collection, and slice
mode. The dry-run should match the vindex you intend to publish and the runtime
role it is supposed to support.

## 4. Publish

```bash
export HF_TOKEN=...
export SKULK_WEIGHTS_SCRATCH=/fast/scratch/skulk-weights
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k
```

The command refuses to overwrite an existing output path. Use `--force` only
when you intentionally want to replace a local extraction output.

## 5. Record The Result

After publication, record the catalog key, source model, target repository,
collection, slice mode, and runner used. That gives Skulk operators a concrete
vindex to inspect when they start assigning GPU inference nodes and
CPU/high-memory LARQL servers.
