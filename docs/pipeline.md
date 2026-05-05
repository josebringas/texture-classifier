# Pipeline architecture

## Decoupled stages

The pipeline is intentionally split into separate scripts rather than one monolithic train-and-serve binary. This follows the three-pipeline pattern (feature → training → inference) common to mature ML systems, scaled down to a single-developer workflow.

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Dataset Prep    │ ──▶ │  Training        │ ──▶ │  Inference       │
│                  │     │                  │     │                  │
│  - normalize     │     │  - train         │     │  - CLI predict   │
│  - embed (CLIP)  │     │  - eval (val)    │     │  - UI batch sort │
│  - visualize     │     │  - register      │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │                        │                         │
        ▼                        ▼                         ▼
   embeddings_cache.npz    model/model_vNNN/        sorted UE folders
```

## Why separate the embedding step

`compute_embeddings.py` runs CLIP once and caches vectors to disk. `launch_visualizer.py` reads the cache and opens FiftyOne. They're separate scripts because:

- **Computing embeddings is slow and occasional.** Run when the dataset changes.
- **Launching the visualizer is fast and frequent.** Run every time you want to inspect the dataset.

Fused together, you'd pay full embedding cost on every visualization. Split, the visualizer is a "open the window" operation, not a "wait for computation" operation.

## Why versioned model bundles

Every training run writes to a fresh `model_vNNN/` folder containing the weights, class names, val indices, and metadata together. This pattern serves three needs:

1. **Reproducibility.** The val indices and split seed are recorded alongside the weights. Evaluation can grade the model against the exact same held-out samples it was scored on during training, with a seed cross-check that fails loudly if anything got out of sync.
2. **Comparison.** Run the confusion matrix script against `model_v001`, `model_v002`, `model_v003` and compare per-class accuracies side by side. Real model-improvement experiments require this.
3. **Rollback.** If `v003` regressed for some reason, point your inference scripts at `v002` by changing one constant. No re-training needed.

The registry is a pure filesystem convention — no database, no service. `model_registry.py` provides a single helper (`resolve_model_dir`) that scans the directory and returns the latest (or a specified) version.

## Confidence threshold and the Review folder

The classifier is intentionally not a fully autonomous sorter. The `TextureClassifier_v001.py` UI exposes a confidence threshold slider. Predictions above the threshold auto-sort into their predicted category folder; predictions below are routed to a `Review/` folder for human triage.

This matters because:

- The trained model only knows the 5 (or N) classes it was trained on. Out-of-distribution inputs (a metal texture, a water shader) will get classified as *something* — the threshold catches them as low-confidence and escalates rather than misfiles them.
- Real-world inboxes have edge cases that even a good model gets wrong. Routing those to a human reviewer is faster and safer than auto-sorting and hoping.

The threshold is exposed at runtime because different production scenarios want different cutoffs. A strict pipeline might want 0.95+; a high-throughput sort job might accept 0.70.

## Honest evaluation

`visualizer_DataBleeding.py` is named for the failure mode it prevents: evaluating the model against the full dataset (including training samples) inflates the apparent accuracy because the model has effectively memorized parts of the training set. The script:

1. Loads the held-out validation indices from `val_indices.json` (written by `model_training.py` at split time)
2. Cross-checks the seed in that file against the seed in `model_metadata.json` — refuses to run if they disagree
3. Wraps the dataset in a `Subset` containing only the val indices
4. Computes the confusion matrix on that subset

The script also supports `EVAL_SCOPE = "all"` for a deliberately-dishonest full-dataset evaluation, useful as a diagnostic: a large gap between "all" accuracy and "val" accuracy is a clear overfitting signal.
