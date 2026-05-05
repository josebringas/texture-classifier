import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import json
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, font
import time

from model_registry import resolve_model_dir

# ── Config ───────────────────────────────────────────────
REGISTRY_ROOT = Path(__file__).parent / "model"
UE_TEXTURES   = Path("C:/Users/joseb/Documents/Unreal Projects/PushByVoice 5.4/Content/Textures")

# Pick model version:
#   MODEL_VERSION = None              → latest
#   MODEL_VERSION = "model_v002"      → pin to a specific run
MODEL_VERSION = None

CONFIDENCE_THRESHOLD = 0.85
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png"]

# ── Colors ───────────────────────────────────────────────
C = {
    "bg":          "#0D0D0D",
    "panel":       "#141414",
    "border":      "#2A2A2A",
    "accent":      "#00FF88",
    "accent_dim":  "#00994D",
    "warn":        "#FFB800",
    "error":       "#FF4444",
    "text":        "#E8E8E8",
    "text_dim":    "#666666",
    "text_mid":    "#999999",
    "row_a":       "#161616",
    "row_b":       "#121212",
}

# ── Model loader ─────────────────────────────────────────
def load_model():
    model_dir   = resolve_model_dir(REGISTRY_ROOT, MODEL_VERSION)
    model_path  = model_dir / "best_model.pth"
    names_path  = model_dir / "class_names.json"
    meta_path   = model_dir / "model_metadata.json"

    with open(names_path) as f:
        class_names = json.load(f)

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    metadata["_version_dir"] = model_dir.name   # always include which folder

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(
        model_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model, transform, class_names, metadata

# ── Main App ─────────────────────────────────────────────
class TextureClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Texture Classifier — UE Pipeline")
        self.root.configure(bg=C["bg"])
        self.root.geometry("920x680")
        self.root.minsize(820, 580)
        self.root.resizable(True, True)

        self.inbox_path   = tk.StringVar(value="")
        self.ue_path      = tk.StringVar(value=str(UE_TEXTURES))
        self.status_text  = tk.StringVar(value="Ready.")
        self.model        = None
        self.transform    = None
        self.class_names  = None
        self.metadata     = None
        self.running      = False

        self._build_fonts()
        self._build_ui()
        self._load_model_async()

    def _build_fonts(self):
        self.f_mono_lg  = font.Font(family="Courier New", size=13, weight="bold")
        self.f_mono_sm  = font.Font(family="Courier New", size=9)
        self.f_mono_md  = font.Font(family="Courier New", size=10)
        self.f_label    = font.Font(family="Courier New", size=8)
        self.f_title    = font.Font(family="Courier New", size=15, weight="bold")
        self.f_sub      = font.Font(family="Courier New", size=8)

    def _build_ui(self):
        # ── Header ───────────────────────────────────────
        header = tk.Frame(self.root, bg=C["bg"], pady=0)
        header.pack(fill="x", padx=20, pady=(18, 0))

        title_row = tk.Frame(header, bg=C["bg"])
        title_row.pack(fill="x")

        tk.Label(title_row, text="TEXTURE", font=self.f_title,
                 bg=C["bg"], fg=C["accent"]).pack(side="left")
        tk.Label(title_row, text=" CLASSIFIER", font=self.f_title,
                 bg=C["bg"], fg=C["text"]).pack(side="left")

        self.subtitle_text = tk.StringVar(
            value="ResNet18 · CLIP Embeddings · UE Pipeline  //  loading model...")
        tk.Label(header,
                 textvariable=self.subtitle_text,
                 font=self.f_sub, bg=C["bg"], fg=C["text_dim"]).pack(anchor="w")

        # ── Divider ──────────────────────────────────────
        tk.Frame(self.root, bg=C["border"], height=1).pack(
            fill="x", padx=20, pady=(10, 0))

        # ── Path config panel ────────────────────────────
        config = tk.Frame(self.root, bg=C["panel"],
                          highlightbackground=C["border"],
                          highlightthickness=1)
        config.pack(fill="x", padx=20, pady=(12, 0))

        self._path_row(config,
                       label="INBOX",
                       desc="Unsorted textures folder",
                       var=self.inbox_path,
                       cmd=self._pick_inbox,
                       row=0)

        tk.Frame(config, bg=C["border"], height=1).grid(
            row=1, column=0, columnspan=4, sticky="ew", padx=0)

        self._path_row(config,
                       label="UE OUTPUT",
                       desc="UE Content/Textures root",
                       var=self.ue_path,
                       cmd=self._pick_ue,
                       row=2)

        config.columnconfigure(2, weight=1)

        # ── Confidence slider ────────────────────────────
        thresh_row = tk.Frame(self.root, bg=C["bg"])
        thresh_row.pack(fill="x", padx=20, pady=(10, 0))

        tk.Label(thresh_row, text="CONFIDENCE THRESHOLD",
                 font=self.f_label, bg=C["bg"],
                 fg=C["text_dim"]).pack(side="left")

        self.thresh_val = tk.DoubleVar(value=CONFIDENCE_THRESHOLD)
        self.thresh_label = tk.Label(thresh_row,
                                     text=f"{CONFIDENCE_THRESHOLD:.0%}",
                                     font=self.f_mono_md,
                                     bg=C["bg"], fg=C["accent"], width=5)
        self.thresh_label.pack(side="right")

        slider = tk.Scale(thresh_row,
                          from_=0.50, to=0.99,
                          resolution=0.01,
                          orient="horizontal",
                          variable=self.thresh_val,
                          command=self._update_thresh,
                          bg=C["bg"], fg=C["text_mid"],
                          troughcolor=C["border"],
                          activebackground=C["accent"],
                          highlightthickness=0,
                          sliderrelief="flat",
                          bd=0, showvalue=False,
                          length=200)
        slider.pack(side="right", padx=(8, 6))

        # ── Run button ───────────────────────────────────
        btn_row = tk.Frame(self.root, bg=C["bg"])
        btn_row.pack(fill="x", padx=20, pady=(14, 0))

        self.run_btn = tk.Button(
            btn_row,
            text="▶  RUN BATCH SORT",
            font=self.f_mono_lg,
            bg=C["accent"], fg=C["bg"],
            activebackground=C["accent_dim"],
            activeforeground=C["bg"],
            relief="flat", bd=0,
            padx=24, pady=10,
            cursor="hand2",
            command=self._run_batch
        )
        self.run_btn.pack(side="left")

        self.model_status = tk.Label(
            btn_row, text="⟳  Loading model...",
            font=self.f_mono_sm,
            bg=C["bg"], fg=C["warn"])
        self.model_status.pack(side="left", padx=(16, 0))

        # ── Stats bar ────────────────────────────────────
        stats = tk.Frame(self.root, bg=C["bg"])
        stats.pack(fill="x", padx=20, pady=(10, 0))

        self.stat_total   = self._stat_chip(stats, "TOTAL",    "—")
        self.stat_sorted  = self._stat_chip(stats, "SORTED",   "—", C["accent"])
        self.stat_review  = self._stat_chip(stats, "REVIEW",   "—", C["warn"])

        # ── Log panel ────────────────────────────────────
        tk.Frame(self.root, bg=C["border"], height=1).pack(
            fill="x", padx=20, pady=(10, 0))

        log_header = tk.Frame(self.root, bg=C["bg"])
        log_header.pack(fill="x", padx=20, pady=(6, 0))
        tk.Label(log_header, text="OUTPUT LOG",
                 font=self.f_label, bg=C["bg"],
                 fg=C["text_dim"]).pack(side="left")

        log_frame = tk.Frame(self.root, bg=C["border"],
                             highlightbackground=C["border"],
                             highlightthickness=1)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(4, 0))

        self.log = tk.Text(
            log_frame,
            bg=C["panel"], fg=C["text"],
            font=self.f_mono_sm,
            relief="flat", bd=0,
            padx=12, pady=10,
            state="disabled",
            wrap="none",
            insertbackground=C["accent"],
            selectbackground=C["accent_dim"],
            cursor="arrow"
        )
        self.log.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log.yview,
                                  bg=C["panel"], troughcolor=C["panel"],
                                  activebackground=C["border"],
                                  relief="flat", bd=0, width=8)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)

        # Tag colors for log
        self.log.tag_config("ok",    foreground=C["accent"])
        self.log.tag_config("warn",  foreground=C["warn"])
        self.log.tag_config("err",   foreground=C["error"])
        self.log.tag_config("dim",   foreground=C["text_dim"])
        self.log.tag_config("head",  foreground=C["text_mid"])

        # ── Status bar ───────────────────────────────────
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=0)
        status_bar = tk.Frame(self.root, bg=C["bg"], pady=5)
        status_bar.pack(fill="x", padx=20)
        tk.Label(status_bar, textvariable=self.status_text,
                 font=self.f_sub, bg=C["bg"],
                 fg=C["text_dim"]).pack(side="left")
        tk.Label(status_bar, text=f"device: {DEVICE}",
                 font=self.f_sub, bg=C["bg"],
                 fg=C["text_dim"]).pack(side="right")

    def _path_row(self, parent, label, desc, var, cmd, row):
        tk.Label(parent, text=label, font=self.f_label,
                 bg=C["panel"], fg=C["text_dim"],
                 width=10, anchor="w",
                 padx=12, pady=10).grid(row=row, column=0, sticky="w")

        tk.Label(parent, text=desc, font=self.f_sub,
                 bg=C["panel"], fg=C["text_dim"]).grid(
                     row=row, column=1, sticky="w", padx=(0, 8))

        entry = tk.Entry(parent, textvariable=var,
                         font=self.f_mono_sm,
                         bg=C["bg"], fg=C["text"],
                         insertbackground=C["accent"],
                         relief="flat", bd=0,
                         highlightthickness=1,
                         highlightbackground=C["border"],
                         highlightcolor=C["accent"])
        entry.grid(row=row, column=2, sticky="ew", padx=(0, 8), pady=8)

        tk.Button(parent, text="BROWSE",
                  font=self.f_label,
                  bg=C["border"], fg=C["text_mid"],
                  activebackground=C["accent"],
                  activeforeground=C["bg"],
                  relief="flat", bd=0,
                  padx=10, pady=6,
                  cursor="hand2",
                  command=cmd).grid(row=row, column=3, padx=(0, 10))

    def _stat_chip(self, parent, label, value, color=None):
        color = color or C["text_mid"]
        frame = tk.Frame(parent, bg=C["panel"],
                         highlightbackground=C["border"],
                         highlightthickness=1)
        frame.pack(side="left", padx=(0, 8), ipadx=12, ipady=6)
        tk.Label(frame, text=label, font=self.f_label,
                 bg=C["panel"], fg=C["text_dim"]).pack()
        val_label = tk.Label(frame, text=value,
                             font=self.f_mono_lg,
                             bg=C["panel"], fg=color)
        val_label.pack()
        return val_label

    def _update_thresh(self, val):
        self.thresh_label.config(text=f"{float(val):.0%}")

    def _pick_inbox(self):
        path = filedialog.askdirectory(title="Select Inbox (Unsorted Textures) Folder")
        if path:
            self.inbox_path.set(path)

    def _pick_ue(self):
        path = filedialog.askdirectory(title="Select UE Content/Textures Root Folder")
        if path:
            self.ue_path.set(path)

    def _log(self, msg, tag=""):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _load_model_async(self):
        def _load():
            try:
                (self.model, self.transform,
                 self.class_names, self.metadata) = load_model()

                # Build a compact metadata line for the header subtitle
                meta = self.metadata or {}
                version  = meta.get("_version_dir", "?")
                trained  = meta.get("trained_on", "?")
                val_acc  = meta.get("val_accuracy")
                acc_str  = f"{val_acc:.1%}" if isinstance(val_acc, (int, float)) else "?"
                clip_used = meta.get("dataset_analysis", {}).get("clip_model", "?")

                subtitle = (f"ResNet18 · {clip_used} embeddings · "
                            f"{version}  //  trained {trained}  //  val acc {acc_str}")

                self.root.after(0, lambda: self.subtitle_text.set(subtitle))
                self.root.after(0, lambda: self.model_status.config(
                    text=f"✓  Model ready — {len(self.class_names)} classes",
                    fg=C["accent"]))
                self.root.after(0, lambda: self.status_text.set(
                    f"Loaded {version}. Classes: {', '.join(self.class_names)}"))
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda: self.subtitle_text.set(
                    "ResNet18 · CLIP Embeddings · UE Pipeline  //  model error"))
                self.root.after(0, lambda: self.model_status.config(
                    text=f"✗  Model error: {err_msg}", fg=C["error"]))
        threading.Thread(target=_load, daemon=True).start()

    def _run_batch(self):
        if self.running:
            return
        if not self.model:
            self._log("✗  Model not loaded yet.", "err")
            return

        inbox = Path(self.inbox_path.get())
        ue    = Path(self.ue_path.get())

        if not inbox.exists():
            self._log("✗  Inbox folder not found. Please select a valid folder.", "err")
            return
        if not ue.exists():
            self._log("✗  UE Textures folder not found. Please select a valid folder.", "err")
            return

        self.running = True
        self.run_btn.config(state="disabled",
                            text="⟳  PROCESSING...",
                            bg=C["border"], fg=C["text_dim"])

        def _batch():
            threshold = self.thresh_val.get()
            textures  = [f for f in inbox.iterdir()
                         if f.suffix.lower() in SUPPORTED_FORMATS]

            if not textures:
                self.root.after(0, lambda: self._log(
                    "  No textures found in inbox folder.", "warn"))
                self._reset_btn()
                return

            # Create subfolders
            for cat in self.class_names:
                (ue / cat).mkdir(parents=True, exist_ok=True)
            (ue / "Review").mkdir(parents=True, exist_ok=True)

            total     = len(textures)
            sorted_n  = 0
            flagged_n = 0
            t_start   = time.time()

            self.root.after(0, lambda: self._log(
                f"\n{'─'*70}", "dim"))
            self.root.after(0, lambda: self._log(
                f"  BATCH START — {total} texture(s) found", "head"))
            model_version = (self.metadata or {}).get("_version_dir", "?")
            self.root.after(0, lambda: self._log(
                f"  Model: {model_version}  |  Threshold: {threshold:.0%}  |  Device: {DEVICE}", "dim"))
            self.root.after(0, lambda: self._log(
                f"{'─'*70}", "dim"))

            flagged_list = []

            for i, tex in enumerate(textures):
                try:
                    img    = Image.open(tex).convert("RGB")
                    tensor = self.transform(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        outputs    = self.model(tensor)
                        probs      = torch.softmax(outputs, dim=1)
                        confidence = probs.max().item()
                        pred_idx   = probs.argmax().item()
                        pred_label = self.class_names[pred_idx]

                    if confidence >= threshold:
                        dest   = ue / pred_label / tex.name
                        tag    = "ok"
                        marker = "✓"
                        line   = (f"  {marker}  {tex.name[:42]:42s}"
                                  f"→ {pred_label:20s} {confidence:.1%}")
                        sorted_n += 1
                    else:
                        dest   = ue / "Review" / tex.name
                        tag    = "warn"
                        marker = "⚠"
                        line   = (f"  {marker}  {tex.name[:42]:42s}"
                                  f"→ REVIEW  ({confidence:.1%} / {pred_label})")
                        flagged_n += 1
                        flagged_list.append((tex.name, pred_label, confidence))

                    shutil.move(str(tex), str(dest))

                    # Capture for lambda closure
                    _line, _tag = line, tag
                    self.root.after(0, lambda l=_line, t=_tag: self._log(l, t))

                except Exception as e:
                    err = f"  ✗  {tex.name} — ERROR: {e}"
                    self.root.after(0, lambda m=err: self._log(m, "err"))

            elapsed = time.time() - t_start

            self.root.after(0, lambda: self._log(f"{'─'*70}", "dim"))
            self.root.after(0, lambda: self._log(
                f"  COMPLETE — {elapsed:.1f}s", "head"))
            self.root.after(0, lambda: self._log(
                f"  Auto-sorted : {sorted_n}", "ok"))
            self.root.after(0, lambda: self._log(
                f"  Flagged     : {flagged_n}", "warn" if flagged_n else "ok"))
            self.root.after(0, lambda: self._log(f"{'─'*70}\n", "dim"))

            # Update stat chips
            self.root.after(0, lambda: self.stat_total.config(text=str(total)))
            self.root.after(0, lambda: self.stat_sorted.config(text=str(sorted_n)))
            self.root.after(0, lambda: self.stat_review.config(text=str(flagged_n)))
            self.root.after(0, lambda: self.status_text.set(
                f"Batch complete — {sorted_n} sorted, {flagged_n} flagged for review."))
            self._reset_btn()

        threading.Thread(target=_batch, daemon=True).start()

    def _reset_btn(self):
        self.running = False
        self.root.after(0, lambda: self.run_btn.config(
            state="normal",
            text="▶  RUN BATCH SORT",
            bg=C["accent"], fg=C["bg"]))


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = TextureClassifierApp(root)
    root.mainloop()