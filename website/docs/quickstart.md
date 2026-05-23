---
title: Quickstart
---

This guide gets you from a clean checkout to your first publisher dry-run.

A dry-run is the best first command because it answers three practical
questions:

1. Which upstream model will be used?
2. Where will the local vindex artifact be written?
3. Which Hugging Face repository would receive the published artifact?

It prints the LARQL commands without extracting weights or uploading files.

## Requirements

- Python 3.11 or newer
- this repository checked out locally
- Node.js 20 or newer only if you are editing the documentation site
- LARQL and a Hugging Face token when you are ready to publish for real

## 1. Install The CLI

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

This installs the `skulk-vindex` command from the current checkout.

## 2. Validate The Catalogue

```bash
skulk-vindex manifest validate
skulk-vindex manifest list --tier smoke
```

The catalogue is `models.yaml`. It is the list of vindex artifacts this project
knows how to build. The `smoke` tier contains the smaller entries that are safest
for first publication tests.

## 3. Check Your Machine

```bash
skulk-vindex doctor
```

The doctor command checks the local Python environment, scratch directory, and
manifest. Use the stricter publishing checks on the machine that will actually
run LARQL:

```bash
skulk-vindex doctor --publish
```

## 4. Dry-Run One Artifact

```bash
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

You should see a summary like:

```text
model key: gemma-3-4b-full-q4-k
source model: google/gemma-3-4b-it
output path: .scratch/gemma-3-4b-it-full-q4-k.vindex
target repo: hf://skulk/gemma-3-4b-it-full-q4-k-vindex
extract command: larql extract ...
publish command: larql publish ...
```

That output is the contract. If the source model, output path, slice mode, or
target repository is wrong, fix `models.yaml` before publishing.

## 5. Learn The Pieces

Read [Skulk, LARQL, and vindexes](concepts/vindexes.md) next. It explains why
the publisher exists and what the dry-run output means.
