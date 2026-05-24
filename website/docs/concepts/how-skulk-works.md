---
title: How Skulk Works
---

This page gives you a working picture of Skulk before you touch anything in
this repository. It assumes you understand LLM inference at a conceptual level
but have not encountered LARQL or vindexes before.

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

## Where SVP Fits

Publishing a vindex is expensive and easy to get wrong. A bad `larql extract`
command can write hundreds of gigabytes to the wrong path. A bad `larql publish`
command can upload a vindex under the wrong Hugging Face repository name,
leaving the cluster unable to agree on which object to use.

SVP: Skulk Vindex Publisher exists to make publication repeatable:

- the catalog records the exact source model, quantization, slice mode, and
  target repository for each vindex
- `skulk-vindex publish --dry-run` prints the exact LARQL commands before
  anything is extracted or uploaded
- the GitHub Actions workflow validates every catalog entry on every pull
  request
- the self-hosted runner performs real extraction and upload only when the plan
  has been reviewed

Once a vindex is published, Skulk operators have a stable repository name to
assign to GPU inference nodes and LARQL servers. That assignment is what
connects the published vindex to the cluster's runtime placement.

Read [The Catalog](catalog.md) next to see how entries are structured and
how operator-owned vindexes can be added alongside the built-in Foxlight
entries.
