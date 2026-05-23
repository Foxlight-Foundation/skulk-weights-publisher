---
title: Publishing Lifecycle
---

A vindex goes through a build-and-release lifecycle, just like any other
artifact you expect production systems to consume.

## 1. Describe The Artifact

The artifact starts as a `models.yaml` entry. That entry names the source model,
quantization, slice mode, local output directory, and target Hugging Face
repository.

## 2. Validate The Catalogue

```bash
skulk-vindex manifest validate
```

Validation catches duplicate keys, unsupported slice names, bad repository
names, and output names that would not be safe to write.

## 3. Check The Runner

```bash
skulk-vindex doctor --publish
```

The publishing runner needs Python, LARQL, writable scratch storage, network
access to Hugging Face, and `HF_TOKEN`.

## 4. Review The Plan

```bash
skulk-vindex publish --model gemma-3-4b-full-q4-k --dry-run
```

The dry-run prints the exact `larql extract` and `larql publish` commands. This
is the last cheap place to catch a wrong source model, path, slice mode, or
repository.

## 5. Extract The Vindex

Real publication starts by running `larql extract`. This can use substantial
scratch disk because it creates a local vindex directory before anything is
uploaded.

## 6. Publish The Vindex

After extraction, the publisher runs `larql publish` and uploads the artifact to
the Hugging Face repository in the catalogue entry.

## 7. Use The Published Artifact

Once published, the vindex has a stable repository name. Skulk operators can
refer to that repository when they need the prepared artifact for runtime work.
