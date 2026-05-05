"""
model_registry.py

Small helper for resolving which model_vNNN/ folder to load from.
Shared by model_inference.py, visualizer_DataBleeding.py, and the
TextureClassifier UI.
"""

from pathlib import Path


def resolve_model_dir(registry_root: Path, version: str | None = None) -> Path:
    """
    Return the path to a specific model version folder inside registry_root.

    If `version` is None: picks the highest-numbered model_vNNN folder (latest).
    If `version` is given: looks for exactly that folder.

    Examples:
        resolve_model_dir(Path("model"))                 # latest
        resolve_model_dir(Path("model"), "model_v002")   # specific
    """
    if not registry_root.exists():
        raise FileNotFoundError(
            f"Registry root does not exist: {registry_root}\n"
            f"Run model_training.py first to create it."
        )

    if version is not None:
        path = registry_root / version
        if not path.exists():
            available = sorted(p.name for p in registry_root.iterdir()
                               if p.is_dir() and p.name.startswith("model_v"))
            raise FileNotFoundError(
                f"Requested version not found: {path}\n"
                f"Available versions: {available or '(none)'}"
            )
        return path

    # No version specified — find the latest
    candidates = []
    for p in registry_root.iterdir():
        if p.is_dir() and p.name.startswith("model_v"):
            suffix = p.name.replace("model_v", "")
            if suffix.isdigit():
                candidates.append((int(suffix), p))

    if not candidates:
        raise FileNotFoundError(
            f"No model_vNNN folders found in {registry_root}.\n"
            f"Run model_training.py first."
        )

    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]
