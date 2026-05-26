---
title: Quickstart
---

This guide gets you from a clean checkout to your first publisher dry-run.

If you are new to Skulk and LARQL, read [How Skulk Works](concepts/how-skulk-works.md)
first. It explains the cluster architecture, what a vindex is, and why the
publisher exists before you run any commands.

If you already have that context, keep the core model in mind:

- LARQL decompiles transformer weights into queryable vindexes.
- A vindex is a vector-index directory LARQL can query, run, and publish.
- Skulk uses published vindexes to place weight-serving work on the right
  machines: GPU nodes for the attention path, CPU/high-memory LARQL servers
  for FFN and expert weights.

A dry-run is the best first command because it answers four practical questions
before touching any disk or network:

1. Which upstream model will be used?
2. Where will the local vindex directory be written?
3. Which Hugging Face repository would receive the published vindex?
4. Which Hugging Face collection would list the published repo?

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

This installs the `skulk-weights` command from the current checkout.

## 2. Validate The Catalog

```bash
skulk-weights catalog validate
skulk-weights catalog sources
skulk-weights catalog list --tier smoke
```

The Foxlight catalog is included automatically. The `smoke` tier contains the
smaller entries that are safest for first publication tests. Keys are
namespaced, so Foxlight entries begin with `foxlight/`.

To add your own catalog later, create a starter config:

```bash
skulk-weights catalog init
```

Then add a source file under your own namespace and run commands with
`--config skulk-weights.yaml`. The built-in Foxlight entries are still included.

## 3. Check Your Machine

```bash
skulk-weights doctor
```

The doctor command checks the local Python environment, scratch directory, and
catalog. Use the stricter publishing checks on the machine that will actually
run LARQL:

```bash
skulk-weights doctor --publish
```

## 4. Dry-Run One Vindex

```bash
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

You should see a summary like:

```text
model key: foxlight/gemma-3-4b-full-q4-k
source model: google/gemma-3-4b-it
output path: .scratch/gemma-3-4b-it-full-q4-k.vindex
target repo: hf://FoxlightAI/gemma-3-4b-it-full-q4-k-vindex
collection: https://huggingface.co/collections/FoxlightAI/vindexes-6a124406dd5fb439c431b051
extract command: larql extract ...
publish command: larql publish ...
```

That output is the contract. If the source model, output path, slice mode,
target repository, or collection is wrong, fix the catalog source before
publishing.

One thing that looks surprising: entries with `slices: [full]` show
`--slices none` in the generated `larql publish` command. That is correct —
LARQL uses `none` to mean "publish the complete vindex." The catalog field
is `full`; the LARQL flag is `none`. They refer to the same thing.

## 5. Go Deeper

[How Skulk Works](concepts/how-skulk-works.md) explains the end-to-end cluster
architecture and why the vindex format exists. [Skulk, LARQL, and
Vindexes](concepts/vindexes.md) covers the vindex structure and extraction
levels in more detail.
