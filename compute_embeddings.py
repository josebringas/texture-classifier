"""
compute_embeddings.py

Computes CLIP embeddings for every image in textures/ and caches them to disk.
Runs incrementally — only processes images that aren't already cached.

Run this whenever the textures/ folder changes (new images added, old ones removed).
"""

import numpy as np
import torch
import open_clip
from pathlib import Path
from PIL import Image

# ── Config ───────────────────────────────────────────────
TEXTURE_ROOT    = Path(__file__).parent / "textures"
CACHE_PATH      = Path(__file__).parent / "embeddings_cache.npz"
IMG_EXTS        = {".jpg", ".jpeg", ".png"}
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP model config — change these together.
# ViT-B-32-quickgelu → 512-dim, fast, default choice
# ViT-L-14           → 768-dim, ~4x slower, better at subtle distinctions
MODEL_NAME      = "ViT-L-14"
PRETRAINED_TAG  = "openai"
EMBEDDING_DIM   = 768

# ── Step 1: scan disk ────────────────────────────────────
print(f"Scanning {TEXTURE_ROOT}...")
disk_paths = sorted(
    str(p) for p in TEXTURE_ROOT.rglob("*.*")
    if p.suffix.lower() in IMG_EXTS
)
print(f"  Found {len(disk_paths)} images on disk")

# ── Step 2: load existing cache (if any) ─────────────────
if CACHE_PATH.exists():
    cache = np.load(CACHE_PATH, allow_pickle=True)
    cache_model = str(cache["model_name"]) if "model_name" in cache.files else "unknown"

    # If the cache was built with a different model, it's unusable —
    # vectors from different models aren't comparable (different dims, different space).
    if cache_model != MODEL_NAME:
        print(f"  Cache was built with '{cache_model}', current model is '{MODEL_NAME}'")
        print(f"  Discarding incompatible cache — will recompute from scratch")
        cached_paths   = []
        cached_vectors = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    else:
        cached_paths   = cache["paths"].tolist()
        cached_vectors = cache["vectors"]
        print(f"  Loaded cache with {len(cached_paths)} entries (model: {cache_model})")
else:
    cached_paths   = []
    cached_vectors = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    print("  No existing cache — starting fresh")

# ── Step 3: diff disk vs cache ───────────────────────────
disk_set   = set(disk_paths)
cached_set = set(cached_paths)

new_paths     = [p for p in disk_paths   if p not in cached_set]  # need compute
kept_indices  = [i for i, p in enumerate(cached_paths) if p in disk_set]  # keep
removed_count = len(cached_paths) - len(kept_indices)

print(f"  New images to embed: {len(new_paths)}")
print(f"  Cached entries kept: {len(kept_indices)}")
print(f"  Cached entries dropped (no longer on disk): {removed_count}")

# ── Step 4: compute embeddings for new images ────────────
if new_paths:
    print(f"\nLoading CLIP model {MODEL_NAME} ({PRETRAINED_TAG}) on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED_TAG, device=DEVICE
    )
    model.eval()

    new_vectors = np.zeros((len(new_paths), EMBEDDING_DIM), dtype=np.float32)

    with torch.no_grad():
        for i, path in enumerate(new_paths):
            img    = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            vec    = model.encode_image(tensor).cpu().numpy()[0]
            new_vectors[i] = vec

            if (i + 1) % 25 == 0 or (i + 1) == len(new_paths):
                print(f"  Embedded {i+1}/{len(new_paths)}")
else:
    new_vectors = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    print("\nNothing to compute — cache is already up to date.")

# ── Step 5: merge kept + new into updated cache ──────────
kept_paths   = [cached_paths[i]   for i in kept_indices]
kept_vectors = cached_vectors[kept_indices] if kept_indices else \
               np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

final_paths   = kept_paths + new_paths
final_vectors = np.vstack([kept_vectors, new_vectors])

# ── Step 6: save ─────────────────────────────────────────
np.savez_compressed(
    CACHE_PATH,
    paths=np.array(final_paths),
    vectors=final_vectors,
    model_name=np.array(MODEL_NAME),
)
print(f"\nCache updated: {CACHE_PATH}")
print(f"  Model:         {MODEL_NAME}")
print(f"  Total entries: {len(final_paths)}")
print(f"  Vector shape:  {final_vectors.shape}")
