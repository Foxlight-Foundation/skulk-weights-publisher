---
title: Quickstart
---

This guide gets you from a clean checkout to your first publisher dry-runs —
one for a LARQL vindex, one for an MTP sidecar.

If you are new to Skulk and LARQL, read [How Skulk Works](concepts/how-skulk-works.md)
first. It explains the cluster architecture, what a vindex is, what MTP sidecars
are for, and why the publisher exists before you run any commands.

If you already have that context, keep the core model in mind:

- SWP publishes three artifact types: **LARQL vindexes**, **MTP sidecars**, and
  **vision sidecars**.
- A vindex is a vector-index directory LARQL can query, run, and publish to let
  Skulk split weight-serving work across GPU nodes and CPU/high-memory servers.
- An MTP sidecar is a full-precision (bf16, unquantized) file (`mtp.safetensors`)
  extracted from the BF16 checkpoint for models with native multi-token
  prediction heads. One sidecar per base model serves every quantization.
- A vision sidecar is a byte-for-byte mirror of a VLM's vision-encoder weights,
  for mlx-community quants that omit them.
- Some models (e.g. Gemma 4) skip embedded MTP heads entirely and ship a
  companion `{model}-assistant` drafter model; SWP records the pairing in the
  catalog instead of extracting tensors.
- Every real publish also uploads a self-describing `README.md` model card and
  files the artifact into its per-type Hugging Face collection.

A dry-run is the best first command because it prints the full publication plan
— source model, output path, target repo, commands — without touching disk or
network.

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- this repository checked out locally
- Node.js 20 or newer only if you are editing the documentation site
- LARQL and a Hugging Face token when you are ready to publish for real

## 1. Install The CLI

```bash
uv sync --extra dev
```

This installs the `skulk-weights` command from the current checkout. Run it via
`uv run skulk-weights ...`. Add `--extra mtp` for MTP/vision real-publish
support and `--extra ui` for the GUI.

## 2. Validate The Catalog

```bash
uv run skulk-weights catalog validate
uv run skulk-weights catalog sources
uv run skulk-weights catalog list --tier smoke
```

The Foxlight catalog is included automatically. The `smoke` tier contains the
smaller entries that are safest for first publication tests. Keys are
namespaced, so Foxlight entries begin with `foxlight/`.

To add your own catalog later, create a starter config:

```bash
uv run skulk-weights catalog init
```

Then add a source file under your own namespace and run commands with
`--config skulk-weights.yaml`. The built-in Foxlight entries are still included.

## 3. Check Your Machine

```bash
uv run skulk-weights doctor
```

The doctor command checks the local Python environment, scratch directory, and
catalog. Use the stricter publishing checks on the machine that will actually
run LARQL:

```bash
uv run skulk-weights doctor --publish
```

## 4. Dry-Run One Vindex

```bash
uv run skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --artifact vindex --dry-run
```

You should see a summary like:

```text
model key: foxlight/gemma-3-4b-full-q4-k
artifact: vindex
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

## 5. Dry-Run One MTP Sidecar

For catalog entries that have MTP fields configured, dry-run the sidecar step
separately to verify the source repo, sidecar repo, precision, and output
path before any download starts:

```bash
uv run skulk-weights publish --model my-org/my-model --artifact mtp --dry-run
```

`--artifact` also accepts `vision` (mirror a VLM's vision encoder) and `all`
(every configured artifact for the entry):

```bash
uv run skulk-weights publish --model my-org/my-vlm --artifact vision --dry-run
```

You should see something like:

```text
model key: my-org/my-model
artifact: mtp
mtp source repo:  hf://Qwen/Qwen3-6-7B
mtp sidecar repo: hf://my-org/qwen3-6-7b-mtp/mtp.safetensors
mtp precision:    bf16 (unquantized)
mtp output path:  .scratch/my-org--qwen3-6-7b-mtp-mtp.safetensors
```

If the built-in Foxlight entries do not have MTP configured yet, the output will
say `mtp step: not configured for this entry`. Refer to the
[MTP sidecar guide](guides/mtp-sidecar.md) to add an MTP-capable catalog entry,
or the [Vision sidecar guide](guides/vision-sidecar.md) for vision encoders.

## 6. Go Deeper

[How Skulk Works](concepts/how-skulk-works.md) explains the end-to-end cluster
architecture, why the vindex format exists, and what MTP sidecars enable.
[MTP Sidecar](guides/mtp-sidecar.md) covers the full extraction workflow,
catalog entry format, and troubleshooting for MTP publication.
[Skulk, LARQL, and Vindexes](concepts/vindexes.md) covers vindex structure and
extraction levels in more detail.
