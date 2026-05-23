---
title: Publishing Lifecycle
---

The production lifecycle should be explicit:

1. validate the catalogue
2. check runner prerequisites
3. plan the commands
4. extract the local vindex
5. publish to Hugging Face
6. verify the published artifact
7. keep a report of what happened

The current implementation covers validation, prerequisite checks, command
planning, extraction, and publication. Pull-back verification, checksums,
provenance reports, and generated Hugging Face model cards are planned
production hardening work.
