---
title: Safety Model
---

The publisher separates safe operations from operations that can consume large
amounts of disk, network, and Hugging Face quota.

Safe commands:

- `skulk-vindex manifest validate`
- `skulk-vindex manifest list`
- `skulk-vindex manifest get`
- `skulk-vindex doctor`
- `skulk-vindex publish --dry-run`

Publishing commands:

- require `larql` on `PATH`
- require `HF_TOKEN`
- create a scratch output directory
- upload the resulting vindex to the configured Hugging Face repository

## Dry-Run First

Every new catalogue entry should pass a dry-run before a real publish:

```bash
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

The dry-run prints the exact `larql extract` and `larql publish` commands. It
does not check whether the upstream model license allows access, because that is
resolved by LARQL and Hugging Face during real extraction.

## Overwrite Protection

Real publication refuses to replace an existing local output directory unless
you pass `--force`.

Use `--force` only when the previous local extraction output is disposable. It
does not delete or roll back a remote Hugging Face repository.

## Secret Handling

`HF_TOKEN` should be provided by the publishing runner environment or GitHub
Actions secrets. Do not store it in `models.yaml`, shell history, repository
files, or Docusaurus docs.
