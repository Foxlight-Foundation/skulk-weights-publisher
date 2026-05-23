# skulk-vindex-publisher

Publisher for Skulk LARQL vindex artifacts.

This repository owns extraction, publication, and catalogue metadata for
vindexes consumed by Skulk. Skulk itself is a vindex consumer: it downloads
`hf://...` vindex directories at runtime, but it does not extract them.

## Publisher Scope

Phase 1 scaffolds the publishing pipeline and smoke-tier manifest. CI validates
the manifest and dry-runs every entry without requiring LARQL, HuggingFace
credentials, or a registered self-hosted runner.

Smoke-tier models:

| Key | Source model | Quant | Slices |
|---|---|---|---|
| `gemma-3-4b-full-q4-k` | `google/gemma-3-4b-it` | `q4k` | `full` |
| `llama-3-2-3b-full-q4-k` | `meta-llama/Llama-3.2-3B-Instruct` | `q4k` | `full` |
| `qwen-2-5-7b-full-q4-k` | `Qwen/Qwen2.5-7B-Instruct` | `q4k` | `full` |

Phase 2 adds catalogue entries for the MoE sweet spot. These are the targets
Skulk uses to validate `LarqlRunner` isolation and future FFN delegation work;
actual HuggingFace publication remains operator-gated because it depends on
self-hosted runner capacity and HF credentials.

| Key | Source model | Quant | Slices |
|---|---|---|---|
| `gemma-4-26b-a4b-full-q4-k` | `google/gemma-4-26b-a4b-it` | `q4k` | `full` |
| `gemma-4-26b-a4b-expert-server-q4-k` | `google/gemma-4-26b-a4b-it` | `q4k` | `expert-server` |
| `mixtral-8x7b-full-q4-k` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `q4k` | `full` |
| `mixtral-8x7b-expert-server-q4-k` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `q4k` | `expert-server` |
| `mixtral-8x22b-full-q4-k` | `mistralai/Mixtral-8x22B-Instruct-v0.1` | `q4k` | `full` |
| `mixtral-8x22b-expert-server-q4-k` | `mistralai/Mixtral-8x22B-Instruct-v0.1` | `q4k` | `expert-server` |

## Required Operator Setup

1. Install LARQL on the self-hosted runner and make `larql` available on
   `PATH`.
2. Install a HuggingFace CLI/tooling path compatible with LARQL publication.
3. Configure `HF_TOKEN` as a GitHub Actions secret with write access to the
   target HuggingFace organization.
4. Register a self-hosted GitHub Actions runner with the labels:
   `self-hosted`, `linux`, `larql`, `vindex`.
5. Provision at least 200 GB of fast scratch storage for the Phase 1 smoke tier.
   Phase 2 MoE entries require substantially more scratch capacity before
   non-dry-run publication.

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
publishing artifacts.

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

The workflow is intentionally conservative. It documents the required runner
and credentials, but actual credential registration and runner capacity are
operator-managed.

## Documentation Site

The novice-facing documentation lives in `website/docs/` and builds with
Docusaurus:

```bash
cd website
npm ci
npm run build
```

Pull requests build the site as a validation artifact. Merges to `main` deploy
the same build output to GitHub Pages through `.github/workflows/docs.yml`.
