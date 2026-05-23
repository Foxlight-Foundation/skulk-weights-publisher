# skulk-vindex-publisher

Build and publish LARQL vindexes for Skulk.

Skulk is a distributed LLM inference system. LARQL treats model weights as a
database: it decompiles transformer weights into a queryable vindex and exposes
LQL, the Lazarus Query Language, for browsing, editing, running inference
against, and recompiling model knowledge. A vindex is a vector-index directory
derived from an upstream Hugging Face model and published so CPU/high-memory
LARQL servers can host feed-forward weights while GPU nodes handle the
latency-sensitive inference path.

This repository is the controlled publication workflow. It keeps the catalogue
of publishable vindexes, validates that catalogue, prints the exact LARQL
commands, and runs publication from a configured runner.

## Why This Exists

Vindex publication is expensive and easy to get wrong. A bad command can write a
large vindex to the wrong scratch path or publish it under the wrong Hugging
Face repository. This project makes publication repeatable:

- `models.yaml` describes every vindex
- `skulk-vindex manifest validate` checks the catalogue
- `skulk-vindex publish --dry-run` prints the LARQL plan
- GitHub Actions validates every catalogue entry
- the self-hosted runner performs real LARQL publication

## Catalogue

| Key | Source model | Quant | Slices |
|---|---|---|---|
| `gemma-3-4b-full-q4-k` | `google/gemma-3-4b-it` | `q4k` | `full` |
| `llama-3-2-3b-full-q4-k` | `meta-llama/Llama-3.2-3B-Instruct` | `q4k` | `full` |
| `qwen-2-5-7b-full-q4-k` | `Qwen/Qwen2.5-7B-Instruct` | `q4k` | `full` |
| `gemma-4-26b-a4b-full-q4-k` | `google/gemma-4-26b-a4b-it` | `q4k` | `full` |
| `gemma-4-26b-a4b-expert-server-q4-k` | `google/gemma-4-26b-a4b-it` | `q4k` | `expert-server` |
| `mixtral-8x7b-full-q4-k` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `q4k` | `full` |
| `mixtral-8x7b-expert-server-q4-k` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `q4k` | `expert-server` |
| `mixtral-8x22b-full-q4-k` | `mistralai/Mixtral-8x22B-Instruct-v0.1` | `q4k` | `full` |
| `mixtral-8x22b-expert-server-q4-k` | `mistralai/Mixtral-8x22B-Instruct-v0.1` | `q4k` | `expert-server` |

## Required Operator Setup

1. Install LARQL on the self-hosted runner and make `larql` available on `PATH`.
2. Configure `HF_TOKEN` as a GitHub Actions secret with write access to the
   target Hugging Face organization.
3. Register a self-hosted GitHub Actions runner with the labels:
   `self-hosted`, `linux`, `larql`, `vindex`.
4. Provision at least 200 GB of fast scratch storage for the first smoke-tier
   publish. MoE entries require substantially more scratch capacity.

## Install The CLI

Use the package CLI for local development, CI validation, and runner operation:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The legacy script names remain as compatibility wrappers, but `skulk-vindex` is
the product interface:

```bash
skulk-vindex doctor
skulk-vindex manifest validate
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

The dry run prints the LARQL commands that would execute without extracting or
publishing vindexes.

## Publication Preflight

Run this on the self-hosted runner before a real publish:

```bash
python -m pip install -e .
skulk-vindex doctor --publish
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

Real publication refuses to overwrite an existing scratch output path. Remove
the output directory manually or rerun with `--force` when you intentionally
want to replace the local extraction output.

## Workflow

`.github/workflows/publish.yml` supports:

- pull request and main-branch validation on GitHub-hosted runners
- weekly cron for the smoke tier
- manual dispatch for one manifest key
- manual dispatch for all entries in the `smoke`, `moe`, or `all` tiers
- manual dry-run dispatch

The workflow validates catalogue changes on hosted runners and reserves real
publication for the labelled self-hosted runner.

## Documentation Site

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
