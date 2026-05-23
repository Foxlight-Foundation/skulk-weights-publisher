---
title: Skulk, LARQL, and Vindexes
---

This project has three names you need to understand before the commands make
sense.

## Skulk

Skulk is a distributed LLM inference system. Instead of assuming one machine
does all of the work, Skulk is designed around a cluster of machines that can
coordinate model loading and inference.

That creates an artifact problem. A cluster needs to agree on exactly what model
artifact it is using. The artifact must be built in a repeatable way, published
somewhere machines can fetch it, and named clearly enough that operators know
what they are starting.

## LARQL

LARQL is the toolchain used here to prepare and publish Skulk's LARQL-backed
model artifacts. In this repository, LARQL appears as two commands:

```bash
larql extract <source-model> -o <local-output> --quant <quant>
larql publish <local-output> --repo <target-repo> --slices <slice-mode>
```

`extract` turns an upstream Hugging Face model into a local artifact directory.
`publish` uploads that artifact to the Hugging Face repository listed in the
catalogue.

## Vindex

A vindex is the artifact LARQL builds. It is a directory-shaped package derived
from an upstream model. Skulk operators publish vindexes so Skulk can later
download and use the prepared artifact instead of rebuilding it ad hoc.

The important part is not the file extension. The important part is the contract:

- source model: which upstream model the artifact came from
- quantization: how the artifact was prepared
- slice mode: whether the artifact is complete or a specialized slice
- output name: the local directory produced by extraction
- target repository: where the artifact is published

## The Publisher

Skulk Vindex Publisher is the automation layer around LARQL. It reads the
catalogue, validates that entries are well formed, prints the exact LARQL command
plan, and runs publication only from a machine configured for that job.

The publisher matters because vindex publication is expensive and stateful. A
bad command can waste hours of compute, fill scratch storage, or publish an
artifact under the wrong name. The publisher makes the process repeatable before
Skulk depends on the result.
