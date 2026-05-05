"""
launch_visualizer.py

Loads cached CLIP embeddings and opens the FiftyOne app with a UMAP plot.
Fast — does no model inference. Just reads the cache, computes UMAP, launches.

Run compute_embeddings.py first if the cache is missing or stale.
"""

import numpy as np
import fiftyone as fo
import fiftyone.brain as fob
from pathlib import Path

# ── Config ───────────────────────────────────────────────
TEXTURE_ROOT = Path(__file__).parent / "textures"
CACHE_PATH   = Path(__file__).parent / "embeddings_cache.npz"

# ── Sanity check ─────────────────────────────────────────
if not CACHE_PATH.exists():
    raise FileNotFoundError(
        f"No embeddings cache at {CACHE_PATH}\n"
        f"Run compute_embeddings.py first."
    )

# ── Load cache ───────────────────────────────────────────
cache = np.load(CACHE_PATH, allow_pickle=True)
cached_paths   = cache["paths"].tolist()
cached_vectors = cache["vectors"]
print(f"Loaded {len(cached_paths)} cached embeddings")

# ── Build FiftyOne dataset from folder structure ─────────
# Folder names become labels, exactly like ImageClassificationDirectoryTree.
dataset = fo.Dataset.from_dir(
    dataset_dir=str(TEXTURE_ROOT),
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    name="texture_dataset",
    overwrite=True,
)
print(dataset)

# ── Attach cached embeddings to samples ──────────────────
# FiftyOne stores embeddings per-sample. We need to map cache rows → samples
# by file path, since sample order isn't guaranteed to match cache order.
path_to_vector = {p: v for p, v in zip(cached_paths, cached_vectors)}

missing = 0
for sample in dataset:
    vec = path_to_vector.get(sample.filepath)
    if vec is None:
        missing += 1
        continue
    sample["clip_embedding"] = vec.tolist()
    sample.save()

if missing:
    print(f"  Warning: {missing} samples had no cached embedding "
          f"(run compute_embeddings.py to refresh)")

# ── UMAP visualization from the cached vectors ───────────
fob.compute_visualization(
    dataset,
    embeddings="clip_embedding",
    method="umap",
    brain_key="umap_viz",
)

# ── Launch ───────────────────────────────────────────────
session = fo.launch_app(dataset)
session.wait()
