"""
visualizer_DataBleeding.py

Evaluates a trained model on its held-out validation set and plots a confusion
matrix. Uses val_indices.json (written by model_training.py) so the evaluation
is honest — no training images bleed into the "accuracy" number.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model_registry import resolve_model_dir

# ── Config ───────────────────────────────────────────────
DATASET_PATH   = Path(__file__).parent / "textures"
REGISTRY_ROOT  = Path(__file__).parent / "model"
IMG_SIZE       = 224
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pick model version:
#   MODEL_VERSION = None              → latest
#   MODEL_VERSION = "model_v002"      → pin to a specific run
MODEL_VERSION = None

# Evaluation scope:
#   "val"  → honest eval on held-out val indices (recommended)
#   "all"  → evaluate on entire dataset (includes training bleed — diagnostic only)
EVAL_SCOPE = "val"

# ── Resolve paths ────────────────────────────────────────
model_dir    = resolve_model_dir(REGISTRY_ROOT, MODEL_VERSION)
MODEL_PATH   = model_dir / "best_model.pth"
NAMES_PATH   = model_dir / "class_names.json"
META_PATH    = model_dir / "model_metadata.json"
VAL_IDX_PATH = model_dir / "val_indices.json"

print(f"Evaluating model: {model_dir.name}")

# ── Load class names + metadata ──────────────────────────
with open(NAMES_PATH) as f:
    class_names = json.load(f)

metadata = None
if META_PATH.exists():
    with open(META_PATH) as f:
        metadata = json.load(f)
    print(f"  Trained on:      {metadata.get('trained_on', '?')}")
    print(f"  Reported val acc: {metadata.get('val_accuracy', '?')}")

# ── Transforms ───────────────────────────────────────────
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Dataset ──────────────────────────────────────────────
full_dataset = datasets.ImageFolder(str(DATASET_PATH), transform=val_transforms)
print(f"  Full dataset: {len(full_dataset)} images, {len(full_dataset.classes)} classes")

# Defend against class mismatch (someone added/removed a folder since training)
if full_dataset.classes != class_names:
    print("  WARNING: current dataset classes don't match trained model's classes")
    print(f"    trained: {class_names}")
    print(f"    current: {full_dataset.classes}")

# ── Build evaluation subset ──────────────────────────────
if EVAL_SCOPE == "val":
    if not VAL_IDX_PATH.exists():
        raise FileNotFoundError(
            f"No val_indices.json in {model_dir}.\n"
            f"Retrain with the updated training script to generate it, "
            f"or set EVAL_SCOPE = 'all' for a (dishonest) full-dataset eval."
        )

    with open(VAL_IDX_PATH) as f:
        val_payload = json.load(f)

    # Sanity check: the split seed recorded in val_indices must match
    # the one the model was trained with. If not, the indices point at
    # the wrong images and the evaluation is meaningless.
    if metadata is not None:
        idx_seed   = val_payload.get("seed")
        model_seed = metadata.get("hyperparameters", {}).get("train_val_split_seed")
        if idx_seed is not None and model_seed is not None and idx_seed != model_seed:
            raise ValueError(
                f"Seed mismatch between val_indices.json ({idx_seed}) "
                f"and model metadata ({model_seed}).\n"
                f"These files don't belong together — evaluation would be wrong."
            )

    val_indices = val_payload["indices"]
    eval_dataset = Subset(full_dataset, val_indices)
    print(f"  Evaluating on val subset: {len(eval_dataset)} images (honest)")

elif EVAL_SCOPE == "all":
    eval_dataset = full_dataset
    print(f"  Evaluating on full dataset: {len(eval_dataset)} images")
    print(f"  WARNING: this includes training images — not a real accuracy measure")

else:
    raise ValueError(f"EVAL_SCOPE must be 'val' or 'all', got {EVAL_SCOPE!r}")

loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

# ── Model ────────────────────────────────────────────────
model    = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model    = model.to(DEVICE)
model.eval()

# ── Inference ────────────────────────────────────────────
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        preds  = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Metrics ──────────────────────────────────────────────
overall_acc = (all_preds == all_labels).mean()
print(f"\nOverall accuracy ({EVAL_SCOPE}): {overall_acc:.2%}")

# Per-class accuracy — often more revealing than the overall number
print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    mask = all_labels == i
    if mask.sum() == 0:
        print(f"  {name:<20} (no samples in this scope)")
        continue
    class_acc = (all_preds[mask] == all_labels[mask]).mean()
    print(f"  {name:<20} {class_acc:.2%}  ({mask.sum()} samples)")

# ── Confusion matrix ─────────────────────────────────────
cm   = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
plt.title(f"{model_dir.name} — Confusion Matrix ({EVAL_SCOPE} set, acc {overall_acc:.1%})")
plt.tight_layout()

out_name = f"confusion_matrix_{model_dir.name}_{EVAL_SCOPE}.png"
plt.savefig(out_name, dpi=150)
plt.show()
print(f"\nSaved: {out_name}")
