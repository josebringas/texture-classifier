# textures/

Place your training dataset here, organized as one subfolder per class.
Folder names become class labels — `ImageFolder` reads them automatically.

Current taxonomy (5 classes):

```
textures/
├── brick/
├── concrete-plaster/
├── fabric/
├── nature/
└── wood/
```

To use a different taxonomy, replace these folders with your own. No code
changes are needed — `model_training.py` picks up the new class names from
the folder structure.

**Image requirements:**
- JPEG, PNG, or TIFF (TIFFs are auto-converted to PNG by `image_normalization.py`)
- Square aspect ratio (the normalizer center-crops non-square images)
- Roughly balanced class sizes (within 2–3× of each other)

**Recommended minimums** for fine-tuning ResNet18 from ImageNet weights:
- 30+ samples per class (more is better)
- Visual variety within each class (different lighting, scale, color, angle)

The actual image files are gitignored — only the folder structure ships with
the repo.
