---
title: Vindexes
---

A vindex is a directory-shaped LARQL artifact derived from an upstream model.
Skulk treats vindexes as downloadable runtime artifacts. It does not extract
them inside the Skulk repository.

The publisher owns the artifact creation path:

1. choose an upstream Hugging Face model
2. run LARQL extraction with a known quantization
3. publish the resulting vindex directory
4. record enough metadata for Skulk operators to understand what was built

The current catalogue uses `q4k` entries because those are the first practical
targets for small smoke tests and later MoE expert-server artifacts.
