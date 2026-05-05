from PIL import Image
from pathlib import Path

TEXTURE_ROOT = Path("./textures")
VALID_EXTS   = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TIFF_EXTS    = {".tif", ".tiff"}

for img_path in TEXTURE_ROOT.rglob("*.*"):
    if img_path.suffix.lower() not in VALID_EXTS:
        continue

    img = Image.open(img_path)

    # Normalize anything non-standard to 8-bit RGB.
    # Catches: RGBA (alpha), LA/L (grayscale), P (palette),
    # I;16 / I;16B (16-bit TIFF), CMYK, etc.
    if img.mode != "RGB":
        original_mode = img.mode
        img = img.convert("RGB")
        print(f"Converted to RGB: {img_path.name} (was {original_mode})")

    # Center-crop to square
    w, h = img.size
    if w != h:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top  = (h - min_dim) // 2
        img  = img.crop((left, top, left + min_dim, top + min_dim))
        print(f"Cropped: {img_path.name} {w}x{h} → {min_dim}x{min_dim}")

    # TIFFs get converted to PNG on disk (lossless, much smaller at 8-bit).
    # Everything else saves back in its original format.
    if img_path.suffix.lower() in TIFF_EXTS:
        new_path = img_path.with_suffix(".png")
        img.save(new_path, "PNG")
        img_path.unlink()  # remove original .tif/.tiff
        print(f"Converted to PNG: {img_path.name} → {new_path.name}")
    else:
        img.save(img_path)

print("Done.")
