---
title: Gemma 4 Assistant Models
---

This guide covers adding a Gemma 4 model to the Foxlight catalog via both the
CLI and the GUI. Gemma 4 uses a companion-assistant pattern for speculative
decoding: Google publishes a pre-built assistant model alongside each
instruction-tuned release. SWP detects this automatically — no tensor
extraction is needed.

## Background

For an overview of both MTP patterns (embedded heads vs companion assistant)
see [Assistant Models](../concepts/assistant-models.md).

## Prerequisites

- SWP installed: `uv sync --extra dev`
- An HF token with read access (no write required — the assistant is already published)
- The URL or `owner/repo` for the Gemma 4 model you want to add (the
  instruction-tuned release, or an MLX community quantization of it)

## CLI

The examples below use `google/gemma-4-12B-it`, which is **not** in the
built-in catalog — `catalog add` refuses models whose key, repo, or output
name already exists, even with `--dry-run`. (The larger `gemma-4-31B-it` and
`gemma-4-26B-A4B-it` already ship as catalog entries; to look at one of those,
jump straight to [step 3](#3-verify).)

### 1. Dry run

```bash
skulk-weights catalog add google/gemma-4-12B-it --dry-run
```

Expected output (abbreviated):

```
fetching metadata for google/gemma-4-12B-it...
checking google/gemma-4-12B-it for MTP keys...
checking google/gemma-4-12B-it-assistant on HuggingFace...

--- entry preview ---

  - key: gemma-4-12b-full-q4-k
    source_model: google/gemma-4-12B-it
    quant: q4k
    tier: smoke
    slices:
      - full
    output_name: gemma-4-12b-it-full-q4-k.vindex
    hf_repo: FoxlightAI/gemma-4-12b-it-full-q4-k-vindex
    hf_collection: FoxlightAI/vindexes-6a124406dd5fb439c431b051
    assistant_model_repo: google/gemma-4-12B-it-assistant

Gemma 4 companion assistant detected: google/gemma-4-12B-it-assistant
This model uses Google's companion-assistant pattern for speculative decoding.
The assistant is already published — no tensor extraction needed.
--- dry run: nothing written ---
```

The `assistant_model_repo` field is written instead of any `mtp_*` fields.

### 2. Write to catalog

```bash
skulk-weights catalog add google/gemma-4-12B-it
# or skip the confirmation prompt:
skulk-weights catalog add google/gemma-4-12B-it --yes
```

The entry is appended to the built-in `foxlight.yaml`. Run
`skulk-weights catalog validate` to confirm the new entry passes schema
validation.

### 3. Verify

Inspect any catalogued assistant-pattern entry — your newly added one, or the
`gemma-4-31b` entry that ships with SWP:

```bash
skulk-weights catalog show foxlight/gemma-4-31b-full-q4-k
```

The JSON output should include:

```json
{
  "assistant_model_repo": "google/gemma-4-31B-it-assistant",
  "mtp_source_repo": null,
  "mtp_sidecar_repo": null
}
```

## GUI (skulk-ui)

1. Start the GUI: `uv run skulk-ui`
2. Paste the model URL (e.g. `https://huggingface.co/google/gemma-4-12B-it`)
   into the Detect field and click **Detect**.
3. The detection result panel shows:
   - **Assistant model**: `google/gemma-4-12B-it-assistant`
   - An info banner: "This model uses Gemma 4's companion assistant pattern.
     The assistant model is pre-published by Google — no tensor extraction needed."
4. Click **Register in Catalog**. The log panel shows confirmation that the
   assistant model location was verified and the job completes immediately.
   (As with the CLI, registration fails if the model is already in the
   catalog — the shipped `gemma-4-31B`/`gemma-4-26B-A4B` entries can be
   inspected with **Find in Catalog** instead.)

## Catalog entry format

```yaml
  - key: gemma-4-31b-full-q4-k
    source_model: google/gemma-4-31B-it
    quant: q4k
    tier: smoke
    slices:
      - full
    output_name: gemma-4-31b-it-full-q4-k.vindex
    hf_repo: FoxlightAI/gemma-4-31b-it-full-q4-k-vindex
    hf_collection: FoxlightAI/vindexes-6a124406dd5fb439c431b051
    assistant_model_repo: google/gemma-4-31B-it-assistant
```

`assistant_model_repo` must:
- Look like `owner/repo` (the `HF_REPO_PATTERN` regex)
- Not be set at the same time as any `mtp_*` field

## Known Gemma 4 assistants

| Instruction-tuned model | Companion assistant |
|---|---|
| `google/gemma-4-31B-it` | `google/gemma-4-31B-it-assistant` |
| `google/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it-assistant` |
| `google/gemma-4-12B-it` | `google/gemma-4-12B-it-assistant` |
| `google/gemma-4-E4B-it` | `google/gemma-4-E4B-it-assistant` |
| `google/gemma-4-E2B-it` | `google/gemma-4-E2B-it-assistant` |

Additional variants follow the same `{model}-assistant` naming convention.
SWP auto-detects them, so this table only needs updating when you want to
document confirmed mappings.
