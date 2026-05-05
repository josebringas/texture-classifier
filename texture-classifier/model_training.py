import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path
from datetime import date
from collections import Counter
import json

# ── Config ──────────────────────────────────────────────
DATASET_PATH   = Path(__file__).parent / "textures"
REGISTRY_ROOT  = Path(__file__).parent / "model"
REGISTRY_ROOT.mkdir(exist_ok=True)

EPOCHS       = 30        # more epochs since we're training deeper
BATCH_SIZE   = 16
LR           = 1e-5      # lower LR to protect pretrained weights
VAL_SPLIT    = 0.2
IMG_SIZE     = 224
SEED         = 42        # fixes the train/val split so it's reproducible
ARCHITECTURE = "resnet18"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset-analysis provenance — record which CLIP model was used for
# embedding/UMAP analysis of this dataset. Informational only; this script
# doesn't use CLIP directly, but analysis decisions (class splits, removed
# samples, etc.) were driven by embeddings from this model.
CLIP_MODEL_USED      = "ViT-L-14"
CLIP_PRETRAINED_TAG  = "openai"

# ── Versioned output folder ─────────────────────────────
# Scan existing model_vNNN folders, pick the next number.
def next_version_dir(root: Path) -> Path:
    existing = [p for p in root.iterdir()
                if p.is_dir() and p.name.startswith("model_v")]
    numbers = []
    for p in existing:
        suffix = p.name.replace("model_v", "")
        if suffix.isdigit():
            numbers.append(int(suffix))
    next_num = max(numbers, default=0) + 1
    return root / f"model_v{next_num:03d}"

OUTPUT_PATH = next_version_dir(REGISTRY_ROOT)
OUTPUT_PATH.mkdir()
VERSION = OUTPUT_PATH.name   # e.g. "model_v001" — stored in metadata

print(f"Training on: {DEVICE}")
print(f"Output folder: {OUTPUT_PATH}")

# ── Transforms ──────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Dataset ─────────────────────────────────────────────
full_dataset = datasets.ImageFolder(str(DATASET_PATH))
class_names  = full_dataset.classes
print(f"Classes found: {class_names}")
print(f"Total samples: {len(full_dataset)}")

with open(OUTPUT_PATH / "class_names.json", "w") as f:
    json.dump(class_names, f)

val_size   = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

# Seeded generator → split is reproducible across runs
generator = torch.Generator().manual_seed(SEED)
train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                  generator=generator)

# Save val indices so eval scripts (e.g. confusion matrix) can grade
# the model honestly on the held-out set instead of the full dataset.
val_indices_payload = {
    "seed": SEED,
    "val_split": VAL_SPLIT,
    "indices": val_set.indices,
    "file_paths": [full_dataset.samples[i][0] for i in val_set.indices],
}
with open(OUTPUT_PATH / "val_indices.json", "w") as f:
    json.dump(val_indices_payload, f, indent=2)
print(f"Saved val indices: {len(val_set.indices)} samples held out")

train_set.dataset.transform = train_transforms
val_set.dataset.transform   = val_transforms

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

# ── Model ───────────────────────────────────────────────
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 and final classifier
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model     = model.to(DEVICE)

# Only pass trainable params to optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# LR scheduler - reduces LR when val loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=4, factor=0.5
)

criterion = nn.CrossEntropyLoss()

# ── Training loop ───────────────────────────────────────
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            val_loss    += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / train_size
    val_acc   = val_correct   / val_size
    avg_val_loss = val_loss / len(val_loader)

    # Step scheduler based on val loss
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2%} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.2%}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_PATH / "best_model.pth")
        print(f"  ✓ Best model saved ({val_acc:.2%})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2%}")
print(f"Model saved to: {OUTPUT_PATH / 'best_model.pth'}")

# ── Model registry entry ────────────────────────────────
# Single source of truth for what this trained model actually is.
# Pair this with best_model.pth — never let them get separated.
class_distribution = dict(Counter(class_names[label]
                                  for _, label in full_dataset.samples))

metadata = {
    "version": VERSION,
    "trained_on": date.today().isoformat(),
    "model_filename": "best_model.pth",
    "architecture": ARCHITECTURE,
    "num_classes": len(class_names),
    "categories": class_names,
    "class_distribution": class_distribution,
    "training_samples": train_size,
    "val_samples": val_size,
    "val_accuracy": round(best_val_acc, 4),
    "hyperparameters": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "image_size": IMG_SIZE,
        "val_split": VAL_SPLIT,
        "train_val_split_seed": SEED,
    },
    "dataset_analysis": {
        "clip_model":      CLIP_MODEL_USED,
        "clip_pretrained": CLIP_PRETRAINED_TAG,
        "note": "CLIP model used for embedding/UMAP analysis of this dataset. "
                "Not used during ResNet training.",
    },
    "device_trained_on": str(DEVICE),
}

with open(OUTPUT_PATH / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Registry entry written: {OUTPUT_PATH / 'model_metadata.json'}")


'''
**What this script does step by step**

- Loads your folder structure as labeled data automatically via `ImageFolder`
- Splits 80% training / 20% validation
- Applies augmentation to training only — flips, rotations, color jitter — your free sample multiplier
- Loads ResNet18 pretrained on ImageNet, freezes all layers except the final classifier
- Only trains the last layer — fast, avoids overfitting at your dataset size
- Saves the best performing checkpoint automatically

**Expected output per epoch looks like:**

Epoch 01/20 | Train Loss: 1.2847 | Train Acc: 52.38% | Val Loss: 0.8921 | Val Acc: 71.43%
'''