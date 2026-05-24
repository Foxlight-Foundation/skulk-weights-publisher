---
slug: /
title: Skulk Vindex Publisher
---

Skulk Vindex Publisher publishes LARQL vindexes for Skulk so model weights do
not have to live entirely inside expensive GPU memory. It keeps the list of
upstream Hugging Face models Skulk wants, validates how each one should be
extracted and sliced, shows the exact LARQL commands that will run, and runs the
publish workflow so CPU/high-memory LARQL servers can host feed-forward weights
while GPU nodes handle the latency-sensitive inference work.

The Foxlight catalog is built in. You can use it immediately, or add your own
operator catalog with `skulk-vindex.yaml`. The merged catalog uses
namespaced keys such as `foxlight/gemma-3-4b-full-q4-k` and
`my-org/my-model-full-q4-k` so shared Foxlight vindexes and local operator
vindexes remain distinct.

**New to Skulk and LARQL?** Read [How Skulk Works](concepts/how-skulk-works.md)
before the quickstart. It explains the cluster architecture, what vindexes are,
and why the publisher exists.

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

## Why do this?

Large GPUs and systems with high-bandwidth unified memory are expensive. If the
whole model has to stay resident there, operators pay GPU prices for work that
is often dominated by weight access and memory capacity rather than GPU-only
compute. A published vindex gives Skulk a stable way to split that cost: GPU
nodes handle the latency-sensitive inference path, LARQL servers on
CPU/high-memory machines host the weight-heavy FFN and expert pieces, and every
node refers to the same extracted and sliced model representation.
