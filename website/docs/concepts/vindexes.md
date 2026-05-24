---
title: Skulk, LARQL, and Vindexes
---

This page explains the vocabulary behind the publisher. It assumes you know the
basics of LLM inference and focuses on the terms that are specific to this
project.

## Skulk

Skulk is a distributed LLM inference system. Instead of assuming one machine
does all of the work, Skulk is designed around a cluster of machines that can
coordinate model loading and inference.

Distributed inference makes model identity, model placement, and hardware cost
more important. Every machine in a cluster needs to agree on the exact vindex,
not just a friendly model name. The cluster also needs to know which machines
should run the inference hot path and which machines can serve large
weight-heavy parts of the model.

## LARQL

LARQL is the project that makes transformer weights queryable. Its core idea is
that the model is the database: it decompiles model weights into a vindex and
then exposes LQL, the Lazarus Query Language, as the SQL-like surface for
browsing, editing, running inference against, and recompiling model knowledge.

In this repository, LARQL appears as two command shapes:

```bash
larql extract <source-model> -o <local-output> --quant <quant>
larql publish <local-output> --repo <target-repo> --slices <slice-mode>
```

`extract` creates the local vindex. `publish` uploads the vindex, or a sliced
form of it, to the Hugging Face repository listed in the catalog.

## Vindex

A vindex is the vector index LARQL builds from a model.

Concretely, a vindex is a directory of memory-mapped files where the model's
weights have been reorganized for queryability. Gate vectors become a
nearest-neighbor index. Embeddings become token lookups. Down projections become
edge labels. That is what lets LARQL query model knowledge with LQL instead of
treating the weights as opaque runtime files.

That matters because high-bandwidth GPU memory is expensive, and large models
consume a lot of it just by keeping weights resident. Feed-forward network
weights, including MoE expert weights, are large and memory-heavy. Some of that
work still involves computation, but the expensive resource pressure is often
weight access and memory capacity, not a requirement that every weight-serving
machine be a GPU node. LARQL can host those weights from a CPU/high-memory
server while another node runs the latency-sensitive inference path on GPU. For
Skulk, that is the practical reason to separate model weights from one
monolithic inference process.

It also matters operationally because "use model X" is not specific enough for
production inference. Operators need to know which vindex of model X they are
using and where each slice is supposed to run.

LARQL extraction levels decide which operations the vindex can support:

| Level | Enables | Why it matters |
|---|---|---|
| `browse` | `DESCRIBE`, `WALK`, `SELECT` | Query and inspect model knowledge without a full forward pass. |
| `inference` | browse operations plus `INFER` | Run inference. This is LARQL's default extraction level. |
| `all` | inference operations plus `COMPILE` | Bake patch overlays into a new standalone vindex. |

The catalog records the production contract around that vindex:

- source model: which upstream Hugging Face model LARQL reads from
- quantization: how LARQL stores the extracted weights
- slice mode: whether the published vindex is complete or an expert-server
  slice
- output name: the local `.vindex` directory produced under scratch storage
- target repository: where the vindex is published

## The Publisher

Skulk Vindex Publisher is the automation layer around LARQL publication. It
reads the catalog, validates that entries are well formed, prints the exact
command plan, and runs publication from the machine configured for that job.

The publisher matters because this split only works when every machine agrees
on the same extracted and sliced weights. A bad command can waste hours of
compute, fill scratch storage, or publish a vindex under the wrong name. The
publisher makes the process repeatable before Skulk depends on the result.
