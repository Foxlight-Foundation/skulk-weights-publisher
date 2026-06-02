---
slug: /
title: "SWP: Skulk Weights Publisher"
---

SWP: Skulk Weights Publisher publishes model weights for Skulk clusters. It
handles three distinct artifact types:

**LARQL vindexes** — LARQL decompiles transformer weights into a queryable
vindex format so Skulk does not have to keep every weight resident in expensive
GPU memory. CPU/high-memory LARQL servers host the feed-forward and expert
weights while GPU nodes handle the latency-sensitive attention path.

**MTP sidecars** — Models with native multi-token prediction heads (Qwen3,
DeepSeek V3/R1, and others that expose `mtp.*` tensor keys) need those heads
published separately. Standard quantization pipelines strip MTP tensors. SWP
re-extracts them from the original BF16 checkpoint and uploads them at full
precision (bf16, unquantized) as `mtp.safetensors` to a dedicated repo — one
sidecar per base model, shared across every quantization of it — so Skulk can
use speculative decoding.

**Vision sidecars** — mlx-community VLM quants frequently omit the vision
encoder, leaving the vision weights to live in a third-party repository. SWP
mirrors those encoder weights byte-for-byte into a Foxlight-owned repo so the
multimodal path has no external dependency. See the
[Vision sidecar guide](guides/vision-sidecar.md).

Not every model embeds MTP heads. Gemma 4, for example, ships a separate
companion `{model}-assistant` drafter model for speculative decoding rather
than `mtp.*` tensors. SWP detects this pattern and records the pairing in the
catalog — no tensor extraction is needed. See the
[Gemma 4 assistant guide](guides/gemma4-assistant.md).

Every real publish also uploads a self-describing `README.md` model card: the
source model's license is inherited unchanged, with a Foxlight provenance block
pinning the exact source SHA, plus usage notes. Each artifact is filed into its
per-artifact-type Hugging Face collection (Vindexes, MTP Sidecars, or Vision
Sidecars).

SWP keeps the list of models Skulk wants, validates how each one should be
extracted and sliced or quantized, shows the exact commands that will run, and
executes the publish workflow from a configured runner.

The Foxlight catalog is built in. You can use it immediately, or add your own
operator catalog with `skulk-weights.yaml`. The merged catalog uses namespaced
keys such as `foxlight/gemma-3-4b-full-q4-k` and `my-org/my-model-full-q4-k`
so shared Foxlight entries and local operator entries remain distinct.

**New to Skulk and LARQL?** Read [How Skulk Works](concepts/how-skulk-works.md)
before the quickstart. It explains the cluster architecture, what vindexes are,
what MTP sidecars are for, and why the publisher exists.

## What is LARQL?

LARQL treats a model as a database. It decompiles transformer weights into a
queryable format and exposes LQL, the Lazarus Query Language, for browsing,
editing, running inference against, and recompiling model knowledge. In this
project, we use LARQL through `larql extract` to create vindexes and
`larql publish` to upload the full and expert-server forms that Skulk can place
on the right machines.

## What is a vindex?

A vindex, short for vector index, is LARQL's on-disk representation of a model.
It is a directory of memory-mapped files where transformer weights have been
reorganized for queryability: gate vectors become a nearest-neighbor index,
embeddings become token lookups, and down projections become edge labels.
Because the weights are in this structured form, LARQL can serve FFN and expert
weights from CPU/high-memory nodes instead of requiring every weight-serving
machine to be a GPU inference node.

## What is an MTP sidecar?

Multi-token prediction is a training technique where a model learns to predict
several future tokens simultaneously rather than one at a time. Models trained
this way expose native MTP heads as tensors with `mtp.*` keys in their
checkpoint. At inference time Skulk can use these heads for speculative
decoding — drafting multiple tokens in one forward pass and verifying them —
which substantially increases throughput with no accuracy loss.

Standard quantization pipelines strip MTP tensors to reduce download size.
SWP's MTP pipeline re-extracts those tensors from the original BF16 checkpoint
and publishes them at full precision (bf16, unquantized) as `mtp.safetensors` to
a separate Hugging Face repo — one sidecar per base model, shared across every
quantization of it. The vindex and the MTP sidecar are versioned
independently so each can be updated or replaced without disturbing the other.

## Why do this?

Weight publication is expensive and easy to get wrong. Bad commands can write
hundreds of gigabytes to the wrong path, publish under the wrong Hugging Face
repository name, or silently skip MTP heads that a model needs for speculative
decoding. SWP makes publication repeatable: the catalog records the exact source,
quantization, slice mode, and target for each artifact; `--dry-run` shows the
full plan before anything is extracted or uploaded; and the GitHub Actions
workflow validates every catalog entry on every pull request.
