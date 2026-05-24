---
title: Publishing Lifecycle
---

A vindex goes through a build-and-release lifecycle because it is the boundary
between expensive GPU inference nodes and cheaper CPU/high-memory weight
servers. The goal is a stable, published model representation that every Skulk
node can agree on before runtime placement begins.

## 1. Describe The Vindex

The vindex starts as a catalogue source entry. The built-in Foxlight entries
are packaged with the CLI, and operator entries can be added through
`skulk-vindex.yaml`. Each entry names the source model, quantization, slice
mode, local `.vindex` directory, and target Hugging Face repository. The slice
mode is part of the runtime contract: it tells operators whether they are
publishing a complete vindex or a specialized expert-server shape for weight
serving.

## 2. Validate The Catalogue

```bash
skulk-vindex catalogue validate
```

Validation catches duplicate keys, unsupported slice names, bad repository
names, and output names that would not be safe to write.

## 3. Check The Runner

```bash
skulk-vindex doctor --publish
```

The publishing runner needs Python, LARQL, writable scratch storage, network
access to Hugging Face, and `HF_TOKEN`. It does not have to be the eventual
runtime host; it is the machine that performs the expensive extraction and
upload.

## 4. Review The Plan

```bash
skulk-vindex publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

The dry-run prints the exact `larql extract` and `larql publish` commands. This
is the last cheap place to catch a wrong source model, path, slice mode, or
repository before disk-heavy extraction begins.

## 5. Extract The Vindex

Real publication starts by running `larql extract`. This can use substantial
scratch disk because it creates a local vindex directory before anything is
uploaded.

## 6. Publish The Vindex

After extraction, the publisher runs `larql publish` and uploads the vindex to
the Hugging Face repository in the catalogue entry.

## 7. Use The Published Vindex

Once published, the vindex has a stable repository name. Skulk operators can
use that name when assigning GPU nodes to the latency-sensitive inference path
and CPU/high-memory LARQL servers to FFN or expert weight serving.
