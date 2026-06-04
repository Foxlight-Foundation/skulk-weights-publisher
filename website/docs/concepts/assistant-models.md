---
title: Assistant Models
---

Skulk uses speculative decoding to accelerate inference. For speculative decoding
to work, the serving node must have access to a smaller **draft model** that
proposes token candidates for the target model to verify. Different model families
ship their draft-model weights in different ways — SWP handles both.

## Pattern 1: Embedded MTP heads (Qwen3, DeepSeek)

Qwen3 and DeepSeek V3/R1 embed multi-token prediction (MTP) heads directly as
`mtp.*` tensor keys inside the BF16 checkpoint. Standard quantization pipelines
(mlx-lm's `sanitize()`) strip these keys when converting the model. SWP
re-extracts them from the original BF16 checkpoint and publishes the result at
full precision (bf16, unquantized) as `mtp.safetensors` to a separate Hugging
Face repository. One bf16 sidecar serves every quantization of the base model,
so there is one sidecar per base model.

Catalog fields for this pattern:

```yaml
mtp_source_repo: Qwen/Qwen3.5-9B        # BF16 source to extract from
mtp_sidecar_repo: FoxlightAI/qwen3-5-9b-mtp  # where mtp.safetensors lands
```

Detection: `skulk-weights catalog add` fetches `model.safetensors.index.json`
from the resolved BF16 base and counts keys that start with `mtp.` or contain
`.mtp.`. If any are found, MTP fields are written to the catalog entry.

See the [MTP sidecar guide](../guides/mtp-sidecar.md) for full instructions.

## Pattern 2: Companion assistant (Gemma 4)

Gemma 4 does not embed MTP tensors in the base checkpoint. Instead, Google
publishes a separate **assistant model** alongside each instruction-tuned release.
The assistant is already quantized and published on Hugging Face — no tensor
extraction is needed.

The naming convention is `{base_model}-assistant`:

| Instruction-tuned model | Companion assistant |
|---|---|
| `google/gemma-4-31B-it` | `google/gemma-4-31B-it-assistant` |
| `google/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it-assistant` |

Catalog field for this pattern:

```yaml
assistant_model_repo: google/gemma-4-31B-it-assistant
```

`assistant_model_repo` is mutually exclusive with `mtp_source_repo` and
`mtp_sidecar_repo`. Setting both on the same entry is a validation error.

Detection: `find_assistant_model` checks three candidates in order via a HEAD
request against the HuggingFace API—`{model}-assistant` for the pasted model
itself, then `{immediate_base}-assistant`, then `{resolved_base}-assistant` for
the resolved BF16 root. The first candidate that exists (HTTP 200) is written to
the catalog entry. No download occurs.

When an assistant is found, the written entry uses `assistant_model_repo` and
omits the `mtp_*` fields—assistant presence takes priority at write time.

## Choosing between the two patterns

You do not choose — SWP detects the correct pattern automatically. The decision
tree is:

1. Resolve the BF16 base model from the quantized model's `base_model:quantized:`
   tag.
2. Fetch `model.safetensors.index.json` from the BF16 base. Count `mtp.*` keys.
3. If `mtp.*` keys are found → write MTP fields.
4. If no `mtp.*` keys are found → check HuggingFace for a companion assistant
   (`{base}-assistant`). If found → write `assistant_model_repo`.
5. If neither → no speculative-decoding fields are written.

## Catalog schema summary

```yaml
# Embedded MTP heads (Qwen3, DeepSeek) — sidecar ships bf16, one per base model
mtp_source_repo: owner/model-bf16
mtp_sidecar_repo: FoxlightAI/model-mtp

# Companion assistant (Gemma 4)
assistant_model_repo: google/gemma-4-31B-it-assistant
```

Both groups are optional and mutually exclusive. An entry with neither is a
valid vindex-only entry with no speculative-decoding support.
