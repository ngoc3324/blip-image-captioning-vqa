---
title: "Rebuild Image Captioning VQA Project from Dataset"
description: "Restore source images from the referenced Hugging Face archive and rebuild the four-phase BLIP project."
status: in-progress
priority: P1
effort: 24h
branch: develop-iam
tags: [feature, recovery, experimental]
blockedBy: []
blocks: []
created: 2026-11-22
---

# Rebuild Image Captioning VQA Project from Dataset

## Overview

Restore the missing 300 image files from the annotation's public source archive, preserve an explicit source-to-local mapping, and recreate the dataset loader, BLIP inference, evaluation, and Gradio phases.

## Recovery Assumption

`source_index` is mapped to the source archive's image-entry order. The original selection order cannot be proven from the annotation file alone, so `dataset/restoration_manifest.json` records the mapping and the result must be visually spot-checked before final metric claims.

## Phases

| Phase | Status |
|---|---|
| Dataset/image restoration | In progress |
| BLIP inference and decoding | Pending |
| Evaluation and error analysis | Pending |
| Gradio UI and report | Pending |

## Acceptance

- 300 local images match all annotation paths and are readable.
- Project code can validate data, run BLIP caption/VQA, evaluate predictions, and launch Gradio.
- Mapping provenance and recovery uncertainty are documented.
