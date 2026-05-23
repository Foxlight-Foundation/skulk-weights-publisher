---
slug: /
title: Skulk Vindex Publisher
---

Skulk Vindex Publisher publishes LARQL vindexes for Skulk so model weights can
be served separately from the GPU inference path. It keeps the list of upstream
Hugging Face models Skulk wants, validates how each one should be extracted and
sliced, shows the exact LARQL commands that will run, and runs the publish
workflow so CPU/high-memory LARQL servers can host feed-forward weights while
GPU nodes handle the latency-sensitive inference work.

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

Large models are constrained by both compute and memory. The latency-sensitive
inference path benefits from GPU, but FFN and expert weights are large,
memory-heavy, and can be served by CPU/high-memory machines. A published vindex
gives Skulk a stable way to split those jobs: GPU nodes handle inference,
LARQL servers host the weight-heavy pieces, and every node refers to the same
extracted and sliced model representation.
