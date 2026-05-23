---
slug: /
title: Skulk Vindex Publisher
---

Skulk Vindex Publisher turns upstream Hugging Face models into publishable
vindex artifacts for Skulk. It keeps a catalogue of artifacts, validates each
publish plan, shows the exact LARQL commands that will run, and gives operators
a repeatable way to publish model artifacts instead of rebuilding them by hand.

## What is LARQL?

LARQL is the artifact preparation toolchain used by this part of Skulk. It reads
an upstream model, prepares it with the requested quantization and slice shape,
writes the local vindex artifact, and publishes that artifact to the configured
Hugging Face repository.

In this project, LARQL is the engine behind the publish plan:

```bash
larql extract <source-model> -o <local-output> --quant <quant>
larql publish <local-output> --repo <target-repo> --slices <slice-mode>
```

## What is a vindex?

A vindex is the prepared model artifact that LARQL builds and publishes. Think
of it as a release artifact for inference: it records which upstream model was
used, how it was prepared, which slice shape it has, where it was written
locally, and where it was published. Skulk operators use the published vindex
instead of guessing how to recreate the same artifact later.
