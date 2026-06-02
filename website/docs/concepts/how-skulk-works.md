---
title: How Skulk Works
---

This page gives you a working picture of Skulk before you touch anything in
this repository. It assumes you understand LLM inference at a conceptual level
but have not encountered LARQL, vindexes, or MTP sidecars before.

## The problem: GPU memory is the bottleneck

Running a large language model means keeping its weights in memory fast enough
to serve inference. For most inference setups, that means GPU memory — which is
fast, bandwidth-rich, and expensive. A 70B-parameter model in float16 needs
around 140 GB of memory just to hold the weights, before any batching or
activations. Most GPU nodes top out well below that.

The common responses are quantization (smaller weights), model parallelism
(sharding across multiple GPUs), or just buying more GPUs. All three work, but
all three treat the problem as "not enough GPU memory" rather than asking
whether every weight actually needs to be on a GPU at all.

It doesn't. That is Skulk's starting point.

## What Skulk does differently

A transformer model has two structurally different parts:

- **Attention layers**: small, compute-intensive, latency-sensitive. These
  operations happen sequentially, have tight interdependencies, and benefit
  directly from GPU hardware.
- **Feed-forward network (FFN) layers**: much larger, memory-intensive, more
  tolerant of latency on the weight-access side. In standard dense models,
  FFN weights account for roughly two-thirds of total parameters. In
  mixture-of-experts models, expert weights dwarf everything else.

The key insight is that FFN and expert weights require a lot of memory capacity
but the critical resource is *bandwidth to the weights*, not GPU compute. A
CPU server with a large pool of high-bandwidth memory, or a server with
substantial DRAM, can serve those weights at the speeds LARQL needs — without
paying GPU prices.

Skulk splits the model across hardware to match resource needs:

- **GPU inference nodes** handle the attention path — the latency-sensitive
  part that benefits from GPU parallelism.
- **CPU/high-memory LARQL servers** host the FFN and expert weights — the
  memory-heavy part where weight access, not compute, is the bottleneck.

The cluster coordinates across these roles. Each node agrees on exactly which
published vindex it is using so there is no ambiguity about which version of a
model is running where.

## What LARQL does

LARQL is the software that makes distributed weight serving possible. Its core
idea: treat a model like a database. Instead of loading a model as opaque
binary blobs into a runtime, LARQL *decompiles* the weights into a structured,
queryable format — the vindex. Once in that format, weights can be queried
individually, browsed, and served from remote nodes using LQL, the Lazarus
Query Language.

LQL is a SQL-like interface to model knowledge. A LARQL server hosting FFN
weights can answer queries like "give me the output of expert 14 for this
activation vector" the same way a database answers a SELECT. The GPU inference
node does not need the full weight blob locally; it issues queries to the LARQL
server and receives what it needs.

LARQL comes with two commands that this publisher drives:

```bash
larql extract <source-model> -o <output> --quant <quant>
larql publish <output> --repo <hf-repo> --slices <slice-mode>
```

`extract` reads a Hugging Face model and decompiles it into a local vindex
directory. `publish` uploads that vindex — or a slice of it — to a Hugging
Face repository so every node in the cluster can refer to a stable, versioned
copy.

## What a vindex is

A vindex is the on-disk representation LARQL builds from a model. It is a
directory of memory-mapped files where the weights have been reorganized for
structured access rather than sequential loading.

In practical terms, the reorganization means:

- **Gate vectors** (which expert or FFN path to activate) become a
  nearest-neighbor index. LARQL can look up the right expert without scanning
  all of them.
- **Embeddings** become token lookups. Input tokens resolve to their embedding
  vectors directly.
- **Down projections** (the output side of each FFN block) become edge labels
  in the structure.

The result is a model you can query — not just load and run as a monolith.
That queryability is what lets different nodes in a Skulk cluster access
different parts of the same model without each one holding the whole thing.

Because a vindex is a directory with a stable structure, it can be versioned,
published to a Hugging Face repository, and downloaded by any node that needs
it. That stable reference is how the cluster agrees on exactly which model it
is running.

## A request through the cluster

Here is what inference looks like when a Skulk cluster has a vindex loaded:

1. A request arrives. The GPU inference node begins the forward pass.
2. During the attention layers, the GPU handles computation locally.
3. When execution reaches the FFN or expert layers, the inference node queries
   the LARQL server for the relevant weight vectors.
4. The LARQL server looks up the result in the vindex — nearest-neighbor lookup
   for gate routing, direct retrieval for the selected expert or FFN weights —
   and returns the output.
5. The inference node continues the forward pass with the retrieved values.
6. The response is returned to the caller.

The GPU node never needs the full FFN weight set in its own memory. The LARQL
server never needs to run a full GPU inference pass. Each machine does the part
it is well-suited for.

## Mixture-of-experts and expert-server slices

Mixture-of-experts models (Mixtral, Gemma 4) make this split even more
valuable. A MoE model routes each token through a small subset of its experts,
but the total expert parameter count is large — a 8x7B MoE model like Mixtral
has 46 billion total parameters even though only about 13 billion are active at
once.

Those expert weights are the right candidates for LARQL serving: large,
memory-heavy, and accessed in a structured, lookupable pattern (the router
selects specific experts per token).

Skulk captures this with a dedicated vindex shape called **expert-server**. An
expert-server vindex contains the expert weights in a form optimized for
CPU/high-memory LARQL servers. It is published as a separate Hugging Face
repository from the full vindex so each piece has a stable, unambiguous
identity and can be placed on the correct hardware independently.

That is why the catalog has separate entries for `gemma-4-26b-a4b-full-q4-k`
and `gemma-4-26b-a4b-expert-server-q4-k` — they describe different published
shapes of the same upstream model, intended for different runtime roles.

## Multi-token prediction sidecars

Some models are trained with native multi-token prediction (MTP) heads. Instead
of predicting one next token at a time, a model with MTP learns to predict
several future tokens simultaneously. At inference time, Skulk can use these
heads for speculative decoding: the model drafts multiple tokens in a single
forward pass, then verifies them cheaply, substantially increasing throughput
with no accuracy loss.

MTP heads appear as tensors with `mtp.*` keys in the checkpoint. Models that
expose them include Qwen3 and DeepSeek V3/R1. The problem is that standard
quantization pipelines — including mlx-lm's `sanitize()` function used by most
GGUF and MLX workflows — strip MTP tensors to reduce download size. The
quantized checkpoint you download from Hugging Face typically does not have them.

SWP solves this with a separate extraction pipeline. Given a catalog entry with
MTP fields configured, SWP:

1. Downloads the original BF16 checkpoint (not the quantized one).
2. Identifies the `mtp.*` tensor keys.
3. Quantizes only those tensors at the specified precision.
4. Uploads the result as `mtp.safetensors` to a dedicated sidecar repo.

The sidecar repo is separate from the vindex repo so each artifact has a stable,
unambiguous identity and can be updated independently. The Skulk cluster loads
the sidecar alongside the vindex when speculative decoding is enabled for that
model.

### The Gemma 4 assistant pattern

Not every model embeds `mtp.*` heads. Gemma 4 ships its draft model as a
separate companion model—a `{model}-assistant` drafter that Google publishes on
Hugging Face. SWP does not extract any tensors for it and does not write a model
card for it: it simply records the companion as `assistant_model_repo`, a
pointer the cluster resolves at speculative-decoding time. The MTP sidecar and
the assistant pattern are two ways to supply the same thing—a draft model—and a
catalog entry uses one or the other, never both.

## Vision sidecars

Some vision-language models are published as an mlx-community checkpoint that
omits the vision encoder, leaving those weights in a third-party repository
(Kimi K2.5 is the motivating case). To avoid depending on a third party, SWP can
publish a **vision sidecar**: a Foxlight-owned mirror of that repository's vision
weights and configs.

The vision sidecar is a pure mirror—no quantization, no dtype conversion. SWP
copies the `vision_source_repo` into `vision_sidecar_repo` byte-for-byte using
`huggingface_hub` (no mlx required). As with every other artifact, the cluster
gets a stable Foxlight-owned repository name it can pin.

## Where SWP Fits

Weight publication is expensive and easy to get wrong. A bad `larql extract`
command can write hundreds of gigabytes to the wrong path. A bad `larql publish`
command can upload a vindex under the wrong Hugging Face repository name,
leaving the cluster unable to agree on which object to use. An MTP step that
runs against the wrong source checkpoint silently omits the prediction heads.

SWP: Skulk Weights Publisher exists to make both kinds of publication repeatable:

- the catalog records the exact source model, quantization, slice mode, target
  repository, and — where applicable — MTP source repo, sidecar repo, and
  quantization for each entry
- every real publish uploads a self-describing model card to the published repo,
  so each artifact carries its own provenance (pinned source commit SHA,
  inherited license) instead of relying on memory
- `skulk-weights publish --dry-run` prints the full plan before anything is
  extracted or uploaded
- the GitHub Actions workflow validates every catalog entry on every pull
  request
- the self-hosted runner performs real extraction and upload only when the plan
  has been reviewed

Once a vindex and MTP sidecar are published, Skulk operators have stable
repository names to assign to GPU inference nodes, LARQL servers, and the
speculative decoding pipeline. Those assignments are what connect the published
artifacts to the cluster's runtime placement.

Read [The Catalog](catalog.md) next to see how entries are structured and
how operator-owned entries can be added alongside the built-in Foxlight ones.
