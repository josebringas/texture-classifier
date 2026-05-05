# model/

Trained model registry. Each training run creates a versioned subfolder
(`model_v001/`, `model_v002/`, ...) containing the model weights and metadata
needed to use or evaluate it.

Each version folder contains:

```
model_vNNN/
├── best_model.pth          # ResNet18 weights (best val accuracy snapshot)
├── class_names.json        # ordered class list (matches dataset folder order)
├── val_indices.json        # held-out validation indices + reproducibility seed
└── model_metadata.json     # training config, accuracy, CLIP provenance, dates
```

`val_indices.json` makes evaluation honest: `visualizer_DataBleeding.py` reads
it to grade the model on the same held-out subset it was validated against
during training, with a seed cross-check that fails loudly if the indices and
the model don't belong together.

`model_registry.py` provides `resolve_model_dir()`, which inference and
evaluation scripts use to locate the latest version (or pin to a specific
one). All version folders are gitignored — train your own with
`model_training.py`.
