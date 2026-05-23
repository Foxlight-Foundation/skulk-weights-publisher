---
slug: /
title: Skulk Vindex Publisher
---

Skulk Vindex Publisher is the tool that turns upstream Hugging Face model
weights into packaged artifacts that Skulk operators can publish, inspect, and
reuse.

If you know how LLM inference works but have not worked with Skulk, LARQL, or a
vindex before, the short version is:

- **Skulk** runs LLM inference across one or more machines.
- **LARQL** prepares model artifacts for Skulk's LARQL-backed runtime path.
- A **vindex** is the prepared artifact that gets published and later downloaded.
- This project is the publishing workstation for those artifacts.

Think of a vindex like a release artifact for model inference. The upstream
model on Hugging Face is the source material. LARQL builds the vindex from that
source. Skulk uses the published vindex when an operator wants that model path.

## Why This Exists

Large-model artifacts are expensive to build and easy to publish incorrectly.
The same source model can produce different outputs depending on quantization,
slice type, scratch storage, and publication target. A hand-written shell command
is not enough for a production catalogue.

This repository gives Skulk a controlled artifact factory:

- a catalogue of model artifacts Skulk expects to exist
- a CLI that explains and validates each publish plan before it runs
- a dry-run path that shows the exact LARQL commands
- GitHub Actions validation so catalogue changes are reviewed before publishing
- a publishing workflow for the controlled runner that has disk, LARQL, and
  Hugging Face credentials

Start with [Skulk, LARQL, and vindexes](concepts/vindexes.md) if the nouns are
new. Start with the [quickstart](quickstart.md) if you want to run the tool.
