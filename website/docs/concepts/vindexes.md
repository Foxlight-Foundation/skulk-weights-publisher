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

Distributed inference makes artifact identity more important. Every machine in a
cluster needs to agree on the exact prepared artifact, not just a friendly model
name.

## LARQL

LARQL is the toolchain that prepares model artifacts for this Skulk path.

In ordinary inference work, you might think about downloading model weights and
starting a runtime. In this workflow, there is an explicit artifact preparation
step first. LARQL performs that step. It reads an upstream model, applies the
chosen preparation settings, writes a local vindex directory, and publishes that
directory.

In this repository, LARQL appears as two command shapes:

```bash
larql extract <source-model> -o <local-output> --quant <quant>
larql publish <local-output> --repo <target-repo> --slices <slice-mode>
```

`extract` creates the local artifact. `publish` uploads the artifact to the
Hugging Face repository listed in the catalogue.

## Vindex

A vindex is the prepared model artifact LARQL builds.

It is useful to think of a vindex as a release artifact for inference. The
upstream Hugging Face model is the source. The vindex is the prepared output
with a known quantization, slice mode, output name, and publication target.

That matters because "use model X" is not specific enough for production
inference. Operators also need to know which prepared version of model X they
are using.

The important part is not the file extension. The important part is the contract:

- source model: which upstream model the artifact came from
- quantization: how the artifact was prepared
- slice mode: whether the artifact is complete or a specialized slice
- output name: the local directory produced by extraction
- target repository: where the artifact is published

## The Publisher

Skulk Vindex Publisher is the automation layer around LARQL and vindexes. It
does not ask users to remember long LARQL commands or invent artifact names by
hand. It reads the catalogue, validates that entries are well formed, prints the
exact command plan, and runs publication from the machine configured for that
job.

The publisher matters because vindex publication is expensive and stateful. A
bad command can waste hours of compute, fill scratch storage, or publish an
artifact under the wrong name. The publisher makes the process repeatable before
Skulk depends on the result.
