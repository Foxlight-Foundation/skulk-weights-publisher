---
title: Publishing Safety
---

Vindex publication can consume real resources. LARQL extraction writes large
weight directories, and publication uploads them to Hugging Face. The publisher
is designed so you can inspect the plan before LARQL uses disk, network, or
Hugging Face write access.

## Commands You Can Run Freely

These commands inspect local files and print plans:

- `skulk-weights catalog validate`
- `skulk-weights catalog list`
- `skulk-weights catalog show`
- `skulk-weights doctor`
- `skulk-weights publish --dry-run`

They are useful on a laptop, in pull-request validation, and on the publishing
runner before a real publish.

## Commands That Publish

A real publish runs LARQL and writes to Hugging Face. It needs:

- `larql` available on `PATH`
- `HF_TOKEN` with write access to the target repository and collection
- scratch storage for the extracted vindex directory
- network access to Hugging Face

## Dry-Run First

Every new catalog entry should pass a dry-run before a real publish:

```bash
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

The dry-run prints the exact `larql extract` and `larql publish` commands. It
is the normal review step before a runner starts doing expensive work, and it is
where you confirm the output will support the intended GPU/CPU placement split.

## Configuration Overrides

Two environment variables adjust where and how a publish runs:

- `SKULK_WEIGHTS_SCRATCH` sets the scratch directory used for extracted vindex
  output, so you can point heavy extraction at a disk with room for it.
- `SKULK_WEIGHTS_COLLECTION` overrides the collection a vindex is filed into.
  Setting it to `none`, `0`, `false`, `no`, `off`, or `disabled` turns
  collection filing off entirely.

## Overwrite Protection

Real publication refuses to replace an existing vindex output directory—the
local `.vindex` path written under scratch storage—unless you pass `--force`.

Use `--force` only when the previous local extraction output is disposable.

## Secret Handling

`HF_TOKEN` should be provided by the publishing runner environment or GitHub
Actions secrets. Do not store it in `skulk-weights.yaml`, manifest source files,
shell history, repository files, or Docusaurus docs.
