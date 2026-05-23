---
slug: /
title: Skulk Vindex Publisher
---

You probably already know what an LLM is and what inference means. The new terms
here are **Skulk**, **LARQL**, and **vindex**.

**Skulk** is a distributed inference system. It is built for running model
workloads across one or more machines instead of treating inference as a
single-process, single-machine problem.

**LARQL** is the artifact preparation toolchain used by this part of Skulk. It
takes an upstream Hugging Face model and prepares a Skulk-ready model artifact
with a specific quantization and slice shape.

A **vindex** is that prepared artifact. It is the thing LARQL builds and
publishes. Skulk operators can then refer to the published vindex instead of
rebuilding or guessing how the artifact was produced.

Skulk Vindex Publisher is the tool that makes those vindexes repeatable. It
keeps the catalogue, validates each entry, prints the exact LARQL commands, and
runs publication from a configured runner.

## Why This Exists

Inference systems care deeply about artifact identity. A model name alone is not
enough. The runtime also needs to know how the artifact was prepared, which
quantization was used, whether the artifact is complete or sliced, and where the
published output lives.

That is what a vindex captures for Skulk.

LARQL can build the artifact, but production use needs more than a one-off shell
command. Operators need a catalogue, dry-runs, validation, consistent names, and
a safe publishing workflow.

This repository gives Skulk a controlled artifact factory:

- a catalogue of vindexes Skulk operators can publish
- a CLI that explains and validates each vindex plan before it runs
- a dry-run path that shows the exact `larql extract` and `larql publish`
  commands
- GitHub Actions validation for catalogue and docs changes
- a publishing workflow for the controlled runner with disk, LARQL, and
  Hugging Face credentials

Start with [Skulk, LARQL, and vindexes](concepts/vindexes.md) for the mental
model. Start with the [quickstart](quickstart.md) when you are ready to run a
dry-run.
