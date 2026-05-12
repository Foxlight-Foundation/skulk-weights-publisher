# skulk-vindex-publisher

Publisher for Skulk LARQL vindex artifacts.

This repository owns extraction, publication, and catalogue metadata for
vindexes consumed by Skulk. Skulk itself is a vindex consumer: it downloads
`hf://...` vindex directories at runtime, but it does not extract them.

## Phase 1 Scope

Phase 1 scaffolds the publishing pipeline and smoke-tier manifest. It does not
assume HuggingFace credentials or a registered self-hosted runner are already
available.

Smoke-tier models:

| Key | Source model | Quant | Slices |
|---|---|---|---|
| `gemma-3-4b-full-q4-k` | `google/gemma-3-4b-it` | `q4k` | `full` |
| `llama-3-2-3b-full-q4-k` | `meta-llama/Llama-3.2-3B-Instruct` | `q4k` | `full` |
| `qwen-2-5-7b-full-q4-k` | `Qwen/Qwen2.5-7B-Instruct` | `q4k` | `full` |

## Required Operator Setup

1. Install LARQL on the self-hosted runner and make `larql` available on
   `PATH`.
2. Install a HuggingFace CLI/tooling path compatible with LARQL publication.
3. Configure `HF_TOKEN` as a GitHub Actions secret with write access to the
   target HuggingFace organization.
4. Register a self-hosted GitHub Actions runner with the labels:
   `self-hosted`, `linux`, `larql`, `vindex`.
5. Provision at least 200 GB of fast scratch storage for the Phase 1 smoke tier.

## Local Dry Run

```bash
python3 -m pip install -r requirements.txt
scripts/doctor.sh
scripts/publish-vindex.sh --model gemma-3-4b-full-q4-k --dry-run
```

The dry run prints the LARQL commands that would execute without extracting or
publishing artifacts.

## Workflow

`.github/workflows/publish.yml` supports:

- weekly cron
- manual dispatch for all smoke-tier models
- manual dispatch for one manifest key

The workflow is intentionally conservative. It documents the required runner
and credentials, but actual credential registration and runner capacity are
operator-managed.
