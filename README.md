# SWP: Skulk Weights Publisher

Publish Skulk model weights: LARQL vindexes and MTP sidecars.

Documentation: <https://foxlight-foundation.github.io/skulk-weights-publisher/>

Skulk is a distributed LLM inference system. SWP publishes two kinds of model
weights that Skulk clusters consume:

**LARQL vindexes** — LARQL treats model weights as a database, decompiling
transformer weights into a queryable vindex and exposing LQL, the Lazarus Query
Language. A vindex is a vector-index directory published so Skulk does not have
to keep every weight resident in expensive GPU memory: CPU/high-memory LARQL
servers host feed-forward weights while GPU nodes handle the latency-sensitive
inference path.

**MTP sidecars** — Models with native multi-token prediction heads (`mtp.*`
tensor keys, such as Qwen3 and DeepSeek V3/R1) require those heads to be
published separately. Standard quantization pipelines (e.g. mlx-lm's
`sanitize()`) strip MTP tensors. SWP re-extracts them from the original BF16
checkpoint, quantizes, and publishes the result as `mtp.safetensors` to a
dedicated Hugging Face repo so Skulk can use speculative decoding.

This repository is the controlled publication workflow. It keeps the catalog
of publishable model weights, validates that catalog, prints the exact commands,
and runs publication from a configured runner.

The Foxlight catalog is included automatically and publishes to the
`FoxlightAI` Hugging Face organization and the public
[`Vindexes`](https://huggingface.co/collections/FoxlightAI/vindexes-6a124406dd5fb439c431b051)
collection. Operators can add their own catalog files with
`skulk-weights.yaml`; the merged catalog uses namespaced keys such as
`foxlight/gemma-3-4b-full-q4-k` and `my-org/my-model-full-q4-k` so shared
Foxlight entries and local operator entries can coexist safely.

## Why This Exists

Weight publication is expensive and easy to get wrong. A bad command can write
hundreds of gigabytes to the wrong scratch path, publish under the wrong Hugging
Face repository, or silently omit MTP heads that a model needs. This project
makes publication repeatable:

- packaged Foxlight catalog entries describe shared Skulk vindexes published
  under `FoxlightAI`; MTP sidecar entries can be added by operators
- `skulk-weights.yaml` can add operator-owned catalog source files
- `skulk-weights catalog validate` checks the merged catalog
- `skulk-weights publish --dry-run` prints the full publication plan
- GitHub Actions validates every catalog entry
- the self-hosted runner performs real extraction and publication

## Catalog

Vindex entries (all entries currently in the Foxlight catalog):

| Key | Source model | Quant | Slices |
|---|---|---|---|
| `foxlight/gemma-3-4b-full-q4-k` | `google/gemma-3-4b-it` | `q4k` | `full` |
| `foxlight/llama-3-2-3b-full-q4-k` | `meta-llama/Llama-3.2-3B-Instruct` | `q4k` | `full` |
| `foxlight/qwen-2-5-7b-full-q4-k` | `Qwen/Qwen2.5-7B-Instruct` | `q4k` | `full` |
| `foxlight/gemma-4-26b-a4b-full-q4-k` | `google/gemma-4-26b-a4b-it` | `q4k` | `full` |
| `foxlight/gemma-4-26b-a4b-expert-server-q4-k` | `google/gemma-4-26b-a4b-it` | `q4k` | `expert-server` |
| `foxlight/mixtral-8x7b-full-q4-k` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `q4k` | `full` |
| `foxlight/mixtral-8x7b-expert-server-q4-k` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `q4k` | `expert-server` |
| `foxlight/mixtral-8x22b-full-q4-k` | `mistralai/Mixtral-8x22B-Instruct-v0.1` | `q4k` | `full` |
| `foxlight/mixtral-8x22b-expert-server-q4-k` | `mistralai/Mixtral-8x22B-Instruct-v0.1` | `q4k` | `expert-server` |

Catalog entries can additionally declare MTP fields (`mtp_source_repo`,
`mtp_sidecar_repo`, `mtp_quant`) to enable sidecar extraction for models with
native prediction heads. See the [MTP sidecar guide](https://foxlight-foundation.github.io/skulk-weights-publisher/guides/mtp-sidecar)
for catalog entry format and prerequisites.

## Gemma 4 Assistant Models

Gemma 4 uses a different speculative-decoding pattern from Qwen3 and DeepSeek.
Instead of embedding `mtp.*` tensors in the BF16 checkpoint, Google publishes a
companion **assistant model** as a separate Hugging Face repository. The assistant
is already quantized and published by Google — SWP does not need to extract
anything.

**Pattern:** `{base_model}-assistant`

| Instruction-tuned model | Companion assistant |
|---|---|
| `google/gemma-4-27b-it` | `google/gemma-4-27b-it-assistant` |
| `google/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it-assistant` |

The catalog field for this pattern is `assistant_model_repo` (mutually exclusive
with `mtp_source_repo` / `mtp_sidecar_repo`):

```yaml
  - key: gemma-4-27b-it-full-q4-k
    source_model: mlx-community/gemma-4-27b-it-4bit
    quant: q4k
    tier: moe
    slices:
      - full
    output_name: gemma-4-27b-it-full-q4-k.vindex
    hf_repo: FoxlightAI/gemma-4-27b-it-full-q4-k-vindex
    hf_collection: FoxlightAI/vindexes-6a124406dd5fb439c431b051
    assistant_model_repo: google/gemma-4-27b-it-assistant
```

`skulk-weights catalog add` detects the assistant automatically:

```bash
skulk-weights catalog add mlx-community/gemma-4-27b-it-4bit
# Gemma 4 companion assistant detected: google/gemma-4-27b-it-assistant
# This model uses Google's companion-assistant pattern for speculative decoding.
# The assistant is already published — no tensor extraction needed.
```

See the [Gemma 4 assistant guide](https://foxlight-foundation.github.io/skulk-weights-publisher/guides/gemma4-assistant)
for step-by-step instructions and the [assistant-models concept page](https://foxlight-foundation.github.io/skulk-weights-publisher/concepts/assistant-models)
for background on both MTP patterns.

## Required Operator Setup

For vindex publication:

1. Install LARQL on the self-hosted runner and make `larql` available on `PATH`.
2. Configure `HF_TOKEN` as a GitHub Actions secret with write access to the
   target Hugging Face organization and collection.
3. Register a self-hosted GitHub Actions runner with the labels:
   `self-hosted`, `linux`, `larql`, `vindex`.
4. Provision at least 200 GB of fast scratch storage for the first smoke-tier
   publish. MoE entries require substantially more scratch capacity.

For MTP sidecar publication (additional):

5. The runner must be able to download the source BF16 checkpoint from Hugging
   Face. HF_TOKEN must have read access to the source repo and write access to
   the sidecar repo.
6. Provision additional scratch capacity for the BF16 checkpoint download before
   quantization. Typical BF16 checkpoints run 15–30 GB per model.
7. Install the `mtp` optional extras: `pip install -e ".[mtp]"`. This installs
   `safetensors` on all platforms; `mlx` is only installed on macOS (Apple
   Silicon) due to a platform marker. The standard Linux runner cannot perform
   real MTP extraction — a macOS Apple Silicon runner is required.

## Install The CLI

Use the package CLI for local development, CI validation, and runner operation.

With `uv` (recommended):

```bash
uv sync --extra dev
```

With pip:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

The legacy script names remain as compatibility wrappers, but `skulk-weights` is
the product interface:

```bash
skulk-weights doctor
skulk-weights catalog validate
skulk-weights catalog find google/gemma-3-4b-it
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --artifact vindex --dry-run
skulk-weights publish --model my-org/my-model --artifact mtp --dry-run
```

`skulk-weights catalog find <hf-url-or-owner/repo>` is the reverse of
`catalog show`: given the upstream HuggingFace source model you started from, it
prints every matching catalog entry as JSON, one object per line (or exits
non-zero with a clear message if none match). The mapping is one-to-many — a
single source model can have several entries (e.g. `full` and `expert-server`
slices of the same MoE model), and all are printed. It accepts both bare
`owner/repo` ids and full `huggingface.co/...` URLs, and respects operator
catalogs loaded via `--config`:

```bash
skulk-weights catalog find https://huggingface.co/google/gemma-3-4b-it
# {"hf_repo": "FoxlightAI/...", "key": "foxlight/gemma-3-4b-full-q4-k", "source_model": "google/gemma-3-4b-it", ...}
```

The `--artifact` flag selects which artifact to publish: `vindex` (LARQL vindex),
`mtp` (MTP sidecar), or `all` (both, when the entry has both configured). Omit
it to publish all declared artifacts for that entry.

The dry run prints the commands and paths that would execute without extracting
or uploading weight files.

## skulk-ui (Local GUI)

`skulk-ui` is a point-and-click interface for publishing MTP sidecars. Paste a
HuggingFace model URL, review the detected metadata, and click Publish — no
catalog knowledge required.

**Prerequisites**

- A **source checkout** of this repo. `skulk-ui` serves the React app from the
  in-repo `ui/` tree (built to `ui/dist/`), so it must run from a clone — it is
  not usable from a bare `pip install` of a published wheel. (Override the dist
  location with `SKULK_UI_DIST` if you build elsewhere.)
- The `[ui]` extras installed in the active environment (see below).
- Node.js 18+ and Yarn on `PATH` (only needed on first run — the React app is
  built automatically and cached in `ui/dist/`).
- A HuggingFace token with write access to `FoxlightAI`.

**Run** (from the repo root)

```bash
# With uv (recommended)
uv sync --extra ui
uv run skulk-ui

# With pip, from an editable/source install
pip install -e '.[ui]'
skulk-ui
```

On the first run `skulk-ui` detects that `ui/dist/` is missing, runs
`yarn install && yarn build` automatically (~30 s), then opens
`http://localhost:7842` in your browser. Subsequent starts are instant.

**Options**

```
--port PORT    Port to listen on (default: 7842)
--no-open      Do not open the browser automatically
```

**Configure your HF token**

Click the gear icon in the top-right corner of the UI. Enter your token and
click Save — it is stored in `~/.config/skulk-weights/.env` and read
automatically on every subsequent launch. You can also set `HF_TOKEN` in your
environment; that takes precedence over the saved value.

## Publication Preflight

Run this on the self-hosted runner before a real publish:

```bash
python -m pip install -e .
skulk-weights doctor --publish
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

Real publication refuses to overwrite an existing scratch output path. Remove
the output directory manually or rerun with `--force` when you intentionally
want to replace the local extraction output.

After `larql publish` succeeds, the publisher adds the model repo to the entry's
configured Hugging Face collection. The built-in Foxlight entries target:

```text
https://huggingface.co/collections/FoxlightAI/vindexes-6a124406dd5fb439c431b051
```

## Workflow

`.github/workflows/publish.yml` supports:

- pull request and main-branch validation on GitHub-hosted runners
- weekly cron for the smoke tier
- manual dispatch for one catalog key
- manual dispatch for all entries in the `smoke`, `moe`, or `all` tiers
- manual dry-run dispatch
- optional `catalog_config` dispatch input for operator catalog sources

The workflow validates catalog changes on hosted runners and reserves real
publication for the labelled self-hosted runner.

## Documentation Site

Published documentation is available at:

```text
https://foxlight-foundation.github.io/skulk-weights-publisher/
```

The novice-facing documentation lives in `website/docs/` and builds with
Docusaurus:

```bash
cd website
npm ci
npm run build
```

Pull requests build the site as a validation artifact. Branch pushes publish
preview docs under `/previews/<branch>/`. Pushes to `main` publish the root docs
site.
