import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import json
import sys

from model_registry import resolve_model_dir

# ── Config ───────────────────────────────────────────────
REGISTRY_ROOT        = Path(__file__).parent / "model"
CONFIDENCE_THRESHOLD = 0.85
IMG_SIZE             = 224
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pick model version:
#   MODEL_VERSION = None              → latest
#   MODEL_VERSION = "model_v002"      → pin to a specific run
MODEL_VERSION = None

# ── Resolve paths ────────────────────────────────────────
model_dir    = resolve_model_dir(REGISTRY_ROOT, MODEL_VERSION)
MODEL_PATH   = model_dir / "best_model.pth"
NAMES_PATH   = model_dir / "class_names.json"
META_PATH    = model_dir / "model_metadata.json"

print(f"Using model: {model_dir.name}")

# ── Load ─────────────────────────────────────────────────
with open(NAMES_PATH) as f:
    class_names = json.load(f)

# Read metadata for context (non-fatal if missing — older runs may not have it)
if META_PATH.exists():
    with open(META_PATH) as f:
        metadata = json.load(f)
    print(f"  Trained on:   {metadata.get('trained_on', '?')}")
    print(f"  Val accuracy: {metadata.get('val_accuracy', '?')}")
    print(f"  Classes:      {metadata.get('categories', class_names)}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

model    = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model    = model.to(DEVICE)
model.eval()

# ── Inference ────────────────────────────────────────────
def classify_texture(image_path: str):
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs    = model(tensor)
        probs      = torch.softmax(outputs, dim=1)
        confidence = probs.max().item()
        pred_idx   = probs.argmax().item()
        pred_label = class_names[pred_idx]

    all_probs = {class_names[i]: f"{probs[0][i].item():.2%}"
                 for i in range(len(class_names))}

    status = "✓ AUTO-SORT" if confidence >= CONFIDENCE_THRESHOLD else "⚠ REVIEW"

    print(f"\nImage     : {Path(image_path).name}")
    print(f"Predicted : {pred_label}")
    print(f"Confidence: {confidence:.2%} — {status}")
    print(f"All probs : {all_probs}")

    return pred_label, confidence

# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_texture.jpg"
    classify_texture(image_path)
