---
title: First Publish
---

Your first real publish should prove the whole path with one small vindex:
catalogue entry, runner, LARQL, scratch storage, Hugging Face token, and upload.

Use a `smoke` entry first. It gives you the same workflow shape as larger
vindexes with less disk and time risk.

## 1. Validate The Catalogue

```bash
skulk-vindex manifest validate
```

This proves `models.yaml` is structurally safe before the runner starts.

## 2. Check Publication Prerequisites

```bash
skulk-vindex doctor --publish
```

This checks the pieces needed for a real publish: LARQL, `HF_TOKEN`, scratch
storage, Python dependencies, and the manifest.

## 3. Review The Dry-Run

```bash
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

Read the source model, output path, target repository, and slice mode. The
dry-run should match the vindex you intend to publish.

## 4. Publish

```bash
export HF_TOKEN=...
export SKULK_VINDEX_SCRATCH=/fast/scratch/skulk-vindexes
skulk-vindex publish --model gemma-3-4b-full-q4-k
```

The command refuses to overwrite an existing output path. Use `--force` only
when you intentionally want to replace a local extraction output.

## 5. Record The Result

After publication, record the manifest key, source model, target repository, and
runner used. That gives Skulk operators a concrete vindex to inspect when they
start wiring the published vindex into runtime workflows.
