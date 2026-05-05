# Dataset curation with CLIP embeddings

This document explains how `compute_embeddings.py` and `launch_visualizer.py` are used to inspect and improve the training dataset before fine-tuning ResNet18.

## Why CLIP, not ResNet, for dataset analysis

ResNet18 fine-tuned on your 5 classes only knows what your 5 classes look like. CLIP (ViT-L/14, OpenAI weights) was trained on 400M image-text pairs and has a much broader visual vocabulary. For *finding mislabeled samples* and *detecting edge cases*, CLIP is the better diagnostic tool — its embeddings reflect visual similarity at a level of generality the fine-tuned classifier doesn't have.

## Workflow

1. Run `compute_embeddings.py` after any change to the dataset. It diffs the disk against the cache and only embeds new images.
2. Run `launch_visualizer.py` to open FiftyOne. The UMAP plot shows your dataset clustered by visual similarity, colored by ground-truth label.
3. Look for points where the color doesn't match the cluster. Those are either mislabeled samples, ambiguous edge cases, or genuinely hard examples.
4. Click into individual samples in FiftyOne, decide whether to relabel, remove, or keep.
5. Re-run `compute_embeddings.py` (incremental, fast) and re-launch the visualizer to confirm.

## What good clusters look like

- **Tight, well-separated** — class concept is coherent, dataset is homogeneous
- **Tight but uniform** — class concept is coherent, dataset may be too narrow (lacks variety; will fail on out-of-distribution textures)
- **Diffuse, overlapping** — class definitions don't carve reality at useful joints; consider splitting or merging classes

## Reproducibility

UMAP is stochastic. `launch_visualizer.py` pins `seed=42` so layouts are stable across runs. The CLIP cache stores the model name (`ViT-L-14`) and refuses to mix vectors from different models — switching CLIP variants invalidates the cache and triggers a full recompute.
