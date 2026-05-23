---
slug: /
title: Skulk Vindex Publisher
---

Skulk Vindex Publisher publishes LARQL vindexes for Skulk. It keeps the list of
upstream Hugging Face models Skulk wants, validates how each one should be
extracted, shows the exact LARQL commands that will run, and runs the publish
workflow so operators get repeatable Hugging Face repositories instead of
one-off local builds.

## What is LARQL?

LARQL treats a model as a database. It decompiles transformer weights into a
queryable format and exposes LQL, the Lazarus Query Language, for browsing,
editing, running inference against, and recompiling model knowledge. In this
project, we use LARQL through `larql extract` to create vindexes and
`larql publish` to upload them.

## What is a vindex?

A vindex, short for vector index, is LARQL's on-disk representation of a model.
It is a directory of memory-mapped files where transformer weights have been
reorganized for queryability: gate vectors become a nearest-neighbor index,
embeddings become token lookups, and down projections become edge labels.
Published vindexes are what Skulk can refer to later by a stable Hugging Face
repository name.
