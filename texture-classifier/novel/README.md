# novel/

Drop unsorted texture maps here for the classifier UI to ingest.

`TextureClassifier_v001.py` reads from this folder (or any folder you pick
via the BROWSE button), classifies each image, and routes it to the
appropriate category folder in your Unreal Engine project's `Content/Textures`
directory.

Predictions above the configurable confidence threshold (default 85%) are
auto-sorted. Anything below the threshold is moved to a `Review/` subfolder
inside the UE destination for human evaluation.

Files in this folder are gitignored — it's a working directory, not source.
