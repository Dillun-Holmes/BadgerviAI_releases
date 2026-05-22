#!/usr/bin/env python3
"""Badger_vision — Production-Ready Linux Training Script

Supports every dataset layout produced by Badger Factory, plus plain
YOLO and COCO archives.  Point it at a folder or archive, pick a task,
and training starts automatically.

Supported dataset formats
-------------------------
* **Badger Factory – Keypoint Detection**
    ``keypoint_detection/yolo/`` with ``images/{train,val}.7z`` and
    ``labels/{train,val}.7z`` containing flat image/label files.
* **Badger Factory – Object Detection**
    Same YOLO layout as keypoints.
* **Badger Factory – Image Classification**
    ``image_classification/classifier/evolving_ds_<name>/`` with
    ``{train,val}.7z`` each containing ``<class_name>/image.png`` folders.
* **COCO export**
    ``coco/{train,val}.7z`` each containing ``images/``, ``masks/``,
    and ``coco_instances.json``.
* **Plain YOLO / COCO folder** already extracted on disk.

Features
--------
* Auto-creates a virtual environment and installs dependencies if nothing
  is set up — just run the script.
* Extracts ``.zip``, ``.7z``, ``.tar.gz``, ``.tar.bz2``, ``.tar.xz``,
  ``.rar`` archives (including nested per-split archives).
* **Always uses GPU** — aborts with a clear message if no CUDA device.
* Rich **tqdm progress bars** per batch with live loss / lr / GPU mem.
* **Training snapshot** printed every epoch: current epoch, total epochs,
  ETA, best loss, learning rate, GPU memory.
* Full checkpoint save / resume, EMA, AMP, early-stopping.

Usage::

    python notebooks/linux_train.py /path/to/dataset --task detection
    python notebooks/linux_train.py /path/to/dataset.7z --task keypoints
    python notebooks/linux_train.py /path/to/dataset --task classification
    python notebooks/linux_train.py                    # synthetic demo
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("linux_train")

VALID_TASKS = {"detection", "keypoints", "classification"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_ARCHIVES = {
    ".zip",
    ".7z",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".rar",
}

# ===================================================================
# 1. Auto-setup: venv + dependencies
# ===================================================================


def _in_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix


WHEEL_URL = (
    "https://github.com/Dillun-Holmes/BadgerviAI_releases/releases/download/v4.3.3/badger_vision-4.3.3-py3-none-any.whl"
)


def auto_setup(repo_root: Path) -> None:
    """Create a venv and install Badger_vision if not already set up."""
    if _in_virtualenv():
        log.info("Virtual-env active: %s", sys.prefix)
        _ensure_packages(["tqdm", "py7zr", "rarfile"])
        return

    # Check if badger_vision is already importable in the current Python
    try:
        import badger_vision  # noqa: F401

        log.info("badger_vision already installed in: %s", sys.executable)
        _ensure_packages(["tqdm", "py7zr", "rarfile"])
        return
    except ImportError:
        pass

    venv_dir = repo_root / ".venv"
    venv_python = venv_dir / "bin" / "python"

    if venv_dir.exists() and venv_python.exists():
        log.info(
            "A virtual environment already exists at %s\n"
            "  Activate it and re-run:\n"
            "    source %s/bin/activate && python %s",
            venv_dir,
            venv_dir,
            " ".join(sys.argv),
        )
        sys.exit(1)

    if venv_dir.exists() and not venv_python.exists():
        log.warning("Incomplete .venv at %s — removing and recreating", venv_dir)
        shutil.rmtree(venv_dir)

    log.info("No virtual environment detected — creating at %s ...", venv_dir)
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    pip = str(venv_dir / "bin" / "pip")
    python = str(venv_python)

    log.info("Installing Badger_vision + training extras ...")
    subprocess.check_call(
        [pip, "install", "--upgrade", "pip", "setuptools", "wheel"],
        stdout=subprocess.DEVNULL,
    )

    # Try editable install first (works in the source repo).
    # Fall back to the release wheel (works in BadgerviAI_releases).
    setup_py = repo_root / "setup.py"
    pyproject = repo_root / "pyproject.toml"
    if setup_py.exists() or pyproject.exists():
        subprocess.check_call(
            [pip, "install", "-e", str(repo_root)],
            stdout=subprocess.DEVNULL,
        )
    else:
        log.info("No setup.py/pyproject.toml — installing release wheel ...")
        subprocess.check_call(
            [pip, "install", WHEEL_URL],
            stdout=subprocess.DEVNULL,
        )

    subprocess.check_call(
        [pip, "install", "tqdm", "py7zr", "rarfile"],
        stdout=subprocess.DEVNULL,
    )

    log.info("Setup complete — re-launching inside the new venv ...")
    os.execv(python, [python] + sys.argv)


def _ensure_packages(packages: list[str]) -> None:
    """Install missing pip packages into the current env."""
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.info("Installing missing packages: %s", ", ".join(missing))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing,
            stdout=subprocess.DEVNULL,
        )


# ===================================================================
# 2. GPU enforcement
# ===================================================================


def enforce_gpu():
    """Abort with a helpful message if no CUDA GPU is found."""
    import torch

    if not torch.cuda.is_available():
        log.error(
            "No CUDA GPU detected!  This script requires an NVIDIA GPU.\n"
            "  - Check drivers:    nvidia-smi\n"
            '  - Check PyTorch:    python -c "import torch; print(torch.version.cuda)"\n'
            "  - Install CUDA build: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log.info("GPU : %s  (%.1f GB VRAM)", name, mem_gb)
    log.info("CUDA: %s  |  PyTorch: %s", torch.version.cuda, torch.__version__)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    return torch.device("cuda:0")


# ===================================================================
# 2b. Training config loader
# ===================================================================


_DEFAULT_TRAIN_CONFIG: dict = {
    "dataset_path": "",
    "task": "detection",
    "model": "resnext",
    "model_variant": "resnext_nano",
    "resnext_config": {
        "neck_channels": 64,
        "head_depth": 2,
        "use_ghost": True,
    },
    "convnext_config": {
        "channels": 64,
        "transformer_depth": 2,
        "token_count": 100,
    },
    "epochs": 100,
    "batch_size": 0,
    "img_size": 640,
    "lr": 0.01,
    "warmup_epochs": 3,
    "accumulation_steps": 1,
    "early_stopping_patience": 50,
    "device": "auto",
    "augmentation": True,
    "augmentation_config": {
        "hsv": True,
        "hsv_h_gain": 0.015,
        "hsv_s_gain": 0.7,
        "hsv_v_gain": 0.4,
        "flip": True,
        "flip_p": 0.5,
        "scale": True,
        "scale_min": 0.5,
        "scale_max": 1.5,
        "mosaic": True,
        "mixup": True,
        "mixup_alpha": 1.5,
        "copy_paste": False,
        "copy_paste_p": 0.5,
        "color_jitter": True,
    },
    "detection": {"num_classes": 80, "nms_threshold": 0.45, "conf_threshold": 0.25},
    "keypoints": {
        "num_keypoints": 17,
        "kpt_loss_weight": 1.0,
        "sigmas": None,
        "visible_only": True,
    },
    "classification": {"dropout": 0.0, "label_smoothing": 0.0},
    "resume": "",
    "workspace": "badger_vision_workspace",
}


def load_train_config(config_path: Path | str | None = None) -> dict:
    """Load training configuration from a YAML file.

    Searches for ``train_config.yaml`` next to this script when
    *config_path* is ``None``.  Missing keys fall back to built-in
    defaults so the file can be as sparse as the user likes.

    Returns:
        Merged configuration dictionary.
    """
    import copy

    cfg = copy.deepcopy(_DEFAULT_TRAIN_CONFIG)

    if config_path is None:
        config_path = Path(__file__).resolve().parent / "train_config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        log.info("No train_config.yaml found — using built-in defaults")
        return cfg

    try:
        import yaml
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "pyyaml"],
                stdout=subprocess.DEVNULL,
            )
            import yaml
        except Exception:
            log.warning("Cannot load YAML config (pyyaml unavailable) — using defaults")
            return cfg

    with open(config_path) as f:
        user_cfg = yaml.safe_load(f) or {}

    log.info("Loaded training config from %s", config_path)

    # Merge top-level scalars
    for key in cfg:
        if key in user_cfg:
            if isinstance(cfg[key], dict) and isinstance(user_cfg[key], dict):
                cfg[key].update(user_cfg[key])
            else:
                cfg[key] = user_cfg[key]

    # Validate task
    if cfg["task"] not in VALID_TASKS:
        log.error("Invalid task %r in config — must be one of %s", cfg["task"], VALID_TASKS)
        sys.exit(1)

    return cfg


def resolve_device(device_str: str | None):  # -> torch.device
    """Resolve device string from config into a torch.device.

    ``"auto"`` picks CUDA when available, else CPU.
    ``"cpu"`` forces CPU.
    A digit string like ``"0"`` selects that CUDA device.
    """
    import torch

    if device_str is None or device_str.lower() == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log.info("GPU : %s  (%.1f GB VRAM)", name, mem_gb)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            return dev
        log.warning("No CUDA GPU detected — falling back to CPU (training will be slow)")
        return torch.device("cpu")

    if device_str.lower() == "cpu":
        return torch.device("cpu")

    # Numeric GPU id(s) — pick first for single-GPU
    ids = [int(x.strip()) for x in device_str.split(",") if x.strip().isdigit()]
    if ids:
        dev = torch.device(f"cuda:{ids[0]}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return dev

    return torch.device("cpu")


# ===================================================================
# 3. Archive helpers
# ===================================================================


def _archive_suffix(path: Path) -> str:
    """Return the compound suffix, e.g. '.tar.gz'."""
    joined = "".join(path.suffixes).lower()
    for ext in sorted(SUPPORTED_ARCHIVES, key=len, reverse=True):
        if joined.endswith(ext):
            return ext
    return path.suffix.lower()


def is_archive(path: Path) -> bool:
    return path.is_file() and _archive_suffix(path) in SUPPORTED_ARCHIVES


def extract_archive(archive_path: Path, dest: Path) -> Path:
    """Extract any supported archive to *dest*; return extraction root."""
    dest.mkdir(parents=True, exist_ok=True)
    ext = _archive_suffix(archive_path)
    log.info("Extracting %s (%s) -> %s", archive_path.name, ext, dest)

    if ext == ".zip":
        import zipfile

        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)

    elif ext == ".7z":
        import py7zr

        with py7zr.SevenZipFile(archive_path) as sz:
            sz.extractall(dest)

    elif ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"):
        import tarfile

        with tarfile.open(archive_path) as tf:
            tf.extractall(dest)

    elif ext == ".rar":
        import rarfile

        with rarfile.RarFile(archive_path) as rf:
            rf.extractall(dest)

    else:
        log.error("Unsupported archive format: %s", ext)
        sys.exit(1)

    # If the archive contained a single top-level folder, descend into it
    children = [c for c in dest.iterdir() if not c.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return dest


def extract_if_archive(path: Path, dest: Path) -> Path:
    """If *path* is an archive, extract it; otherwise return as-is."""
    if path.is_file() and is_archive(path):
        return extract_archive(path, dest)
    return path


# ===================================================================
# 4. Dataset format detection + preparation
# ===================================================================


def detect_format(root: Path) -> str:
    """Auto-detect the dataset format under *root*.

    Returns one of:
        'badger_yolo'        – Badger Factory YOLO layout (images/*.7z, labels/*.7z)
        'badger_classifier'  – Badger Factory classification (evolving_ds_*/...)
        'coco_archive'        – COCO export with train.7z / val.7z containing coco_instances.json
        'yolo_flat'           – Standard YOLO (images/ + labels/ already extracted)
        'coco_flat'           – COCO JSON already on disk with images/
        'unknown'
    """
    # Badger Factory YOLO: images/ dir containing train.7z
    images_dir = root / "images"
    labels_dir = root / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        has_archive = any(is_archive(f) for f in images_dir.iterdir() if f.is_file())
        if has_archive:
            return "badger_yolo"
        # Already extracted flat yolo
        return "yolo_flat"

    # Badger Factory classifier: evolving_ds_* sub-folder
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("evolving_ds_"):
            return "badger_classifier"

    # COCO archive: train.7z at root level
    for f in root.iterdir():
        if f.is_file() and f.stem.lower() in ("train", "val", "valid") and is_archive(f):
            return "coco_archive"

    # COCO flat: coco_instances.json / annotations.json present
    for name in ("coco_instances.json", "annotations.json", "_annotations.coco.json"):
        if (root / name).exists():
            return "coco_flat"

    # Check for train/ sub-folder
    train_sub = root / "train"
    if train_sub.is_dir():
        return detect_format(train_sub)

    # yolo/ subfolder (Badger Factory top-level)
    yolo_sub = root / "yolo"
    if yolo_sub.is_dir():
        return detect_format(yolo_sub)

    # coco/ subfolder
    coco_sub = root / "coco"
    if coco_sub.is_dir():
        return detect_format(coco_sub)

    return "unknown"


# ------------------------------------------------------------------
# 4a. Badger Factory YOLO (keypoints / object detection)
# ------------------------------------------------------------------


def _extract_split_archives(parent: Path, split: str, workspace: Path) -> Path:
    """Extract ``parent/<split>.7z`` (or .zip etc.) into workspace."""
    for f in parent.iterdir():
        if f.is_file() and f.stem.lower() == split and is_archive(f):
            dest = workspace / parent.name / split
            return extract_archive(f, dest)
    # Already a folder?
    d = parent / split
    if d.is_dir():
        return d
    log.error("Cannot find %s split in %s", split, parent)
    sys.exit(1)


def prepare_badger_yolo(root: Path, workspace: Path, classes_txt: Path | None = None) -> dict:
    """Handle Badger Factory YOLO layout.

    Extracts images/{train,val}.7z and labels/{train,val}.7z, then
    converts YOLO labels to COCO JSON.
    """
    images_dir = root / "images"
    labels_dir = root / "labels"

    # Try to find classes.txt at root or one level up
    if classes_txt is None:
        for candidate in [root / "classes.txt", root.parent / "classes.txt"]:
            if candidate.exists():
                classes_txt = candidate
                break

    splits: dict = {}
    for split_name in ("train", "val"):
        img_path = _extract_split_archives(images_dir, split_name, workspace)
        lbl_path = _extract_split_archives(labels_dir, split_name, workspace)

        ann_out = workspace / f"{split_name}_annotations.json"
        _yolo_to_coco(img_path, lbl_path, ann_out, classes_txt)

        splits[f"{split_name}_img_dir"] = str(img_path)
        splits[f"{split_name}_ann_file"] = str(ann_out)

    return splits


# ------------------------------------------------------------------
# 4b. Badger Factory classifier
# ------------------------------------------------------------------


def prepare_badger_classifier(root: Path, workspace: Path) -> dict:
    """Handle Badger Factory classification layout.

    Each split archive contains ``<class_name>/<image>`` folders.
    We create a COCO-style JSON mapping class folders to category IDs.
    """
    ds_dir = None
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("evolving_ds_"):
            ds_dir = child
            break
    if ds_dir is None:
        ds_dir = root

    splits: dict = {}
    for split_name in ("train", "val"):
        # Find and extract split archive
        split_root = None
        for f in ds_dir.iterdir():
            if f.is_file() and f.stem.lower() == split_name and is_archive(f):
                split_root = extract_archive(f, workspace / f"cls_{split_name}")
                break
        if split_root is None:
            d = ds_dir / split_name
            if d.is_dir():
                split_root = d
        if split_root is None:
            if split_name == "train":
                log.error("Cannot find %s split for classification in %s", split_name, ds_dir)
                sys.exit(1)
            continue

        # Build COCO JSON from class folders
        ann_out = workspace / f"{split_name}_annotations.json"
        img_dir = _classify_to_coco(split_root, ann_out)
        splits[f"{split_name}_img_dir"] = str(img_dir)
        splits[f"{split_name}_ann_file"] = str(ann_out)

    splits.setdefault("val_img_dir", None)
    splits.setdefault("val_ann_file", None)
    return splits


def _classify_to_coco(split_root: Path, output: Path) -> Path:
    """Convert classification folder layout to COCO JSON.

    Returns the image directory (we create a flat symlink dir so the
    COCO dataloader can find images by file_name).
    """
    from PIL import Image as PILImage

    flat_img_dir = output.parent / (output.stem + "_images")
    flat_img_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(d for d in split_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    categories = [{"id": i + 1, "name": d.name} for i, d in enumerate(class_dirs)]
    cat_map = {d.name: i + 1 for i, d in enumerate(class_dirs)}

    images_list: list[dict] = []
    annotations: list[dict] = []
    img_id = 0
    ann_id = 0

    for class_dir in class_dirs:
        cat_id = cat_map[class_dir.name]
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with PILImage.open(img_file) as im:
                    w, h = im.size
            except Exception:
                continue

            # Unique filename: classname_origname
            unique_name = f"{class_dir.name}_{img_file.name}"
            link = flat_img_dir / unique_name
            if not link.exists():
                shutil.copy2(img_file, link)

            images_list.append({"id": img_id, "file_name": unique_name, "width": w, "height": h})
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [0, 0, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            img_id += 1
            ann_id += 1

    coco_json = {"images": images_list, "annotations": annotations, "categories": categories}
    output.write_text(json.dumps(coco_json))
    log.info("Classification COCO JSON: %s  (%d images, %d classes)", output, len(images_list), len(categories))
    return flat_img_dir


# ------------------------------------------------------------------
# 4c. COCO archive (train.7z / val.7z each with coco_instances.json)
# ------------------------------------------------------------------


def prepare_coco_archive(root: Path, workspace: Path) -> dict:
    """Handle COCO export layout with per-split archives."""
    splits: dict = {}
    for split_name in ("train", "val"):
        split_root = None
        for f in root.iterdir():
            if f.is_file() and f.stem.lower() == split_name and is_archive(f):
                split_root = extract_archive(f, workspace / f"coco_{split_name}")
                break
        if split_root is None:
            d = root / split_name
            if d.is_dir():
                split_root = d
        if split_root is None:
            if split_name == "train":
                log.error("Cannot find %s split in COCO layout: %s", split_name, root)
                sys.exit(1)
            continue

        img_dir, ann_file = _find_coco_split(split_root)
        splits[f"{split_name}_img_dir"] = str(img_dir)
        splits[f"{split_name}_ann_file"] = str(ann_file)

    splits.setdefault("val_img_dir", None)
    splits.setdefault("val_ann_file", None)
    return splits


def _find_coco_split(split_root: Path) -> tuple[Path, Path]:
    """Find images dir and COCO JSON inside an extracted split."""
    # coco_instances.json
    for name in ("coco_instances.json", "annotations.json", "instances.json", "_annotations.coco.json"):
        p = split_root / name
        if p.exists():
            img_dir = split_root / "images"
            if not img_dir.is_dir():
                img_dir = split_root
            return img_dir, p

    # Search one level deeper
    for child in split_root.iterdir():
        if child.is_dir():
            for name in ("coco_instances.json", "annotations.json", "instances.json"):
                p = child / name
                if p.exists():
                    img_dir = child / "images"
                    if not img_dir.is_dir():
                        img_dir = child
                    return img_dir, p

    log.error("Cannot find COCO JSON in %s", split_root)
    sys.exit(1)


# ------------------------------------------------------------------
# 4d. Flat YOLO (already extracted images/ + labels/)
# ------------------------------------------------------------------


def prepare_yolo_flat(root: Path, workspace: Path) -> dict:
    """Handle standard YOLO layout with images/ and labels/ dirs."""
    splits: dict = {}

    # Check if root itself has images/ + labels/ (single split)
    # or train/ val/ sub-dirs
    for split_name in ("train", "val"):
        split_dir = root / split_name
        if not split_dir.is_dir():
            if split_name == "val":
                split_dir = root / "valid"
            if not split_dir.is_dir() and split_name == "train":
                # Flat: root IS the single split
                split_dir = root

        if not split_dir.is_dir():
            if split_name == "train":
                log.error("No train split found in %s", root)
                sys.exit(1)
            continue

        img_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
        lbl_dir = split_dir / "labels"
        if not lbl_dir.is_dir():
            if split_name == "train":
                log.error("No labels/ directory in %s", split_dir)
                sys.exit(1)
            continue

        classes_txt = root / "classes.txt"
        if not classes_txt.exists():
            classes_txt = None

        ann_out = workspace / f"{split_name}_annotations.json"
        _yolo_to_coco(img_dir, lbl_dir, ann_out, classes_txt)
        splits[f"{split_name}_img_dir"] = str(img_dir)
        splits[f"{split_name}_ann_file"] = str(ann_out)

    splits.setdefault("val_img_dir", None)
    splits.setdefault("val_ann_file", None)
    return splits


# ------------------------------------------------------------------
# 4e. Flat COCO (annotations JSON already present)
# ------------------------------------------------------------------


def prepare_coco_flat(root: Path, workspace: Path) -> dict:
    """Handle an already-extracted COCO folder."""
    img_dir, ann_file = _find_coco_split(root)
    return {
        "train_img_dir": str(img_dir),
        "train_ann_file": str(ann_file),
        "val_img_dir": None,
        "val_ann_file": None,
    }


# ------------------------------------------------------------------
# YOLO -> COCO converter
# ------------------------------------------------------------------


def _yolo_to_coco(img_dir: Path, label_dir: Path, output: Path, classes_txt: Path | None = None) -> None:
    """Convert YOLO .txt labels to COCO JSON."""
    from PIL import Image as PILImage

    images_list: list[dict] = []
    annotations: list[dict] = []
    category_set: set[int] = set()
    ann_id = 1

    img_files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    log.info("Converting %d images YOLO -> COCO ...", len(img_files))

    for img_id, img_path in enumerate(img_files, start=1):
        try:
            with PILImage.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue

        images_list.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_idx = int(parts[0])
                cx, cy, bw, bh = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
                abs_x = (cx - bw / 2) * w
                abs_y = (cy - bh / 2) * h
                abs_w = bw * w
                abs_h = bh * h
                if abs_w <= 0 or abs_h <= 0:
                    continue
                cat_id = cls_idx + 1
                category_set.add(cat_id)
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)],
                        "area": round(abs_w * abs_h, 2),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    # Categories
    categories = [{"id": cid, "name": f"class_{cid}"} for cid in sorted(category_set)]
    if classes_txt and classes_txt.exists():
        names = classes_txt.read_text().strip().splitlines()
        categories = [{"id": i + 1, "name": n.strip()} for i, n in enumerate(names)]

    coco_json = {"images": images_list, "annotations": annotations, "categories": categories}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco_json))
    log.info("COCO JSON: %s  (%d imgs, %d anns, %d cats)", output, len(images_list), len(annotations), len(categories))


# ===================================================================
# 5. Navigate Badger Factory top-level to the right sub-folder
# ===================================================================


def resolve_dataset_root(root: Path, task: str) -> Path:
    """Walk into the right Badger Factory sub-folder for the task."""
    task_dirs = {
        "detection": ["object_detection", "yolo"],
        "keypoints": ["keypoint_detection", "yolo"],
        "classification": ["image_classification", "classifier"],
    }
    candidates = task_dirs.get(task, [])

    cur = root
    for dirname in candidates:
        child = cur / dirname
        if child.is_dir():
            cur = child

    # Also descend into coco/ if present and task isn't classification
    if task != "classification":
        coco_sub = cur / "coco"
        yolo_sub = cur / "yolo"
        if yolo_sub.is_dir():
            cur = yolo_sub
        elif coco_sub.is_dir():
            cur = coco_sub

    return cur


# ===================================================================
# 6. Config generation
# ===================================================================


def detect_num_classes(ann_file: str) -> int:
    with open(ann_file) as f:
        data = json.load(f)
    return len(data.get("categories", [])) or 80


def count_images(ann_file: str) -> int:
    with open(ann_file) as f:
        data = json.load(f)
    return len(data.get("images", []))


def auto_batch_size(gpu_mem_gb: float, img_size: int) -> int:
    mem_per_img = (img_size / 640) ** 2 * 0.06
    max_batch = max(1, int(gpu_mem_gb * 0.7 / mem_per_img))

    # Also cap based on system RAM — training needs RAM for workers, data, etc.
    try:
        sys_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        ram_batch_cap = max(1, int(sys_ram_gb * 0.4 / mem_per_img))
        max_batch = min(max_batch, ram_batch_cap)
    except Exception:
        pass

    for bs in [512, 256, 128, 64, 32, 24, 16, 12, 8, 4, 2, 1]:
        if bs <= max_batch:
            return bs
    return 1


def write_configs(
    workspace: Path,
    data_info: dict,
    num_classes: int,
    epochs: int,
    batch_size: int,
    img_size: int,
    lr: float,
    model_type: str,
    train_config: dict | None = None,
) -> tuple[Path, Path]:
    """Write model + data YAML configs.

    When *train_config* is provided the model variant, neck channels,
    head depth, ghost mode, warmup epochs, accumulation steps, and
    early-stopping patience are read from it.
    """
    if train_config is None:
        train_config = _DEFAULT_TRAIN_CONFIG

    configs_dir = workspace / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    warmup = train_config.get("warmup_epochs", 3)
    accum = train_config.get("accumulation_steps", 1)
    patience = train_config.get("early_stopping_patience", 50)

    if model_type == "resnext":
        rx = train_config.get("resnext_config", {})
        variant = train_config.get("model_variant", "resnext_nano")
        neck_ch = rx.get("neck_channels", 64)
        head_d = rx.get("head_depth", 2)
        ghost = str(rx.get("use_ghost", True)).lower()
        model_yaml = textwrap.dedent(f"""\
            model:
              name: "BadgerResNeXt-Train"
              type: "resnext"
              backbone: "{variant}"
              neck_channels: {neck_ch}
              head_depth: {head_d}
              use_ghost: {ghost}
              num_classes: {num_classes}
              image_size: {img_size}
            training:
              epochs: {epochs}
              base_lr: {lr}
              warmup_epochs: {warmup}
              accumulation_steps: {accum}
              early_stopping_patience: {patience}
              batch_size: {batch_size}
        """)
    else:
        cx = train_config.get("convnext_config", {})
        variant = train_config.get("model_variant", "nano")
        # Map variant name to timm backbone
        _variant_to_backbone = {
            "nano": "convnext_tiny",
            "small": "convnext_tiny",
            "medium": "convnext_small",
            "large": "convnext_base",
            "xlarge": "convnext_large",
        }
        backbone = _variant_to_backbone.get(variant, "convnext_tiny")
        channels = cx.get("channels", 64)
        t_depth = cx.get("transformer_depth", 2)
        tokens = cx.get("token_count", 100)
        model_yaml = textwrap.dedent(f"""\
            model:
              name: "Badger_vision-Train"
              backbone: "{backbone}"
              channels: {channels}
              transformer_depth: {t_depth}
              token_count: {tokens}
              num_classes: {num_classes}
              image_size: {img_size}
            training:
              epochs: {epochs}
              base_lr: {lr}
              warmup_epochs: {warmup}
              accumulation_steps: {accum}
              early_stopping_patience: {patience}
              batch_size: {batch_size}
        """)

    model_cfg = configs_dir / "model.yaml"
    model_cfg.write_text(model_yaml)

    data_yaml = textwrap.dedent(f"""\
        img_dir: "{data_info["train_img_dir"]}"
        ann_file: "{data_info["train_ann_file"]}"
    """)
    data_cfg = configs_dir / "data.yaml"
    data_cfg.write_text(data_yaml)

    if data_info.get("val_img_dir"):
        val_yaml = textwrap.dedent(f"""\
            img_dir: "{data_info["val_img_dir"]}"
            ann_file: "{data_info["val_ann_file"]}"
        """)
        (configs_dir / "val_data.yaml").write_text(val_yaml)

    log.info("Configs written to %s", configs_dir)
    return model_cfg, data_cfg


# ===================================================================
# 7. Training snapshot helper
# ===================================================================


def _format_eta(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    return str(datetime.timedelta(seconds=int(seconds)))


def print_snapshot(
    epoch: int,
    total_epochs: int,
    avg_loss: float,
    best_loss: float,
    lr: float,
    gpu_mem_gb: float,
    epoch_time: float,
    elapsed: float,
) -> None:
    """Print a training snapshot summary."""
    remaining_epochs = total_epochs - (epoch + 1)
    eta_seconds = remaining_epochs * epoch_time if epoch_time > 0 else -1

    pct = (epoch + 1) / total_epochs * 100
    bar_len = 30
    filled = int(bar_len * (epoch + 1) / total_epochs)
    bar = "█" * filled + "░" * (bar_len - filled)

    log.info("─" * 64)
    log.info("  TRAINING SNAPSHOT")
    log.info("─" * 64)
    log.info("  Epoch      : %d / %d  (%.1f%%)", epoch + 1, total_epochs, pct)
    log.info("  Progress   : [%s]", bar)
    log.info("  Loss       : %.4f  (best: %.4f)", avg_loss, best_loss)
    log.info("  LR         : %.2e", lr)
    log.info("  GPU Memory : %.1f GB", gpu_mem_gb)
    log.info("  Epoch Time : %.1fs", epoch_time)
    log.info("  Elapsed    : %s", _format_eta(elapsed))
    log.info("  ETA        : %s", _format_eta(eta_seconds))
    log.info("─" * 64)


# ===================================================================
# 8. Training loop with progress bars
# ===================================================================


def run_training(
    model_cfg_path: Path,
    data_cfg_path: Path,
    data_info: dict,
    device,
    epochs: int,
    batch_size: int,
    resume: str | None = None,
    task: str = "detection",
    train_config: dict | None = None,
) -> None:
    """Full training loop with tqdm progress bars and epoch snapshots.

    Args:
        task: ``"detection"``, ``"keypoints"``, or ``"classification"``.
        train_config: Merged config dict from ``load_train_config()``.
            When provided, augmentation and task-specific settings are read
            from it.
    """
    import torch
    from tqdm import tqdm

    from badger_vision.core.api import Badger_vision, validate_classification, validate_detection
    from badger_vision.data import COCODataset, create_dataloader
    from badger_vision.training.smart_trainer import SmartTrainer
    from badger_vision.utils.env import get_optimal_env_config
    from badger_vision.utils.profiler import model_summary
    from badger_vision.utils.yaml_utils import load_yaml

    if train_config is None:
        train_config = _DEFAULT_TRAIN_CONFIG

    config = load_yaml(str(model_cfg_path))
    data_config = load_yaml(str(data_cfg_path))
    img_size = config.get("model", {}).get("image_size", 640)

    # Augmentation toggle from config
    augment_enabled = bool(train_config.get("augmentation", True))

    # Move model to the requested device explicitly
    pv = Badger_vision(str(model_cfg_path))
    model = pv.model.to(device)
    summary = model_summary(model, input_size=(1, 3, img_size, img_size))

    # Dataset — pass augmentation flag from config
    train_dataset = COCODataset(
        img_dir=data_config["img_dir"],
        ann_file=data_config["ann_file"],
        img_size=img_size,
        augment=augment_enabled,
    )

    env_config = get_optimal_env_config()
    num_workers = min(env_config["num_workers"], multiprocessing.cpu_count())
    use_pin_memory = env_config.get("pin_memory", torch.cuda.is_available())

    # Allow train_config overrides for num_workers and pin_memory
    cfg_workers = train_config.get("num_workers", 0)
    if cfg_workers > 0:
        num_workers = cfg_workers
    cfg_pin = train_config.get("pin_memory", "auto")
    if cfg_pin is not None and str(cfg_pin).lower() != "auto":
        use_pin_memory = bool(cfg_pin)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    # Validation
    val_loader = None
    if data_info.get("val_img_dir"):
        val_dataset = COCODataset(
            img_dir=data_info["val_img_dir"],
            ann_file=data_info["val_ann_file"],
            img_size=img_size,
            augment=False,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )

    from badger_vision.core.api import detection_loss_fn

    trainer = SmartTrainer(model, config, device, detection_loss_fn, epochs=epochs)

    # Resume
    start_epoch = 0
    if resume and os.path.exists(resume):
        log.info("Resuming from %s", resume)
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info("Resumed at epoch %d", start_epoch)

    run_dir = Path("runs") / f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    # ── Pre-training summary ──
    num_train = len(train_dataset)
    num_val = len(val_loader.dataset) if val_loader else 0
    log.info("=" * 64)
    log.info("  BADGER_VISION TRAINING")
    log.info("=" * 64)
    log.info("  Model      : %s", config.get("model", {}).get("name", "Badger_vision"))
    log.info(
        "  Params     : %sM  |  FLOPs: %s G  |  Size: %s MB",
        summary["params_M"],
        summary["flops_G"],
        summary["size_mb"],
    )
    log.info("  Task       : %s", task)
    log.info("  Epochs     : %d (starting from %d)", epochs, start_epoch)
    log.info("  Batch Size : %d", batch_size)
    log.info("  LR         : %.6f", config["training"]["base_lr"])
    log.info("  Image Size : %d", img_size)
    log.info("  Train Imgs : %d", num_train)
    log.info("  Val Imgs   : %d", num_val)
    log.info("  Workers    : %d", num_workers)
    log.info("  Pin Memory : %s", use_pin_memory)
    try:
        _sys_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        log.info("  System RAM : %.1f GB", _sys_ram)
    except Exception:
        pass
    log.info("  Augment    : %s", augment_enabled)
    log.info("  AMP        : %s", device.type == "cuda")
    log.info("  Device     : %s", device)
    log.info("  Output     : %s", run_dir)
    log.info("=" * 64)

    training_start = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        trainer.corrector.reset_epoch()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        raw_model = model.module if hasattr(model, "module") else model
        if epoch < 3:
            for param in raw_model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in raw_model.backbone.parameters():
                param.requires_grad = True

        pbar = tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=120,
        )

        trainer.optimizer.zero_grad()
        accumulation_steps = config.get("training", {}).get("accumulation_steps", 1)

        for batch_idx, batch in pbar:
            scene_stats = {"object_count": 10, "avg_object_size": 50, "avg_confidence": 0.8}
            error_signals = {"small_object_misses": trainer.corrector.small_object_misses}
            trainer.error_router.evaluate_scene(scene_stats, error_signals)
            routing_config = trainer.error_router.get_routing_config()
            raw_model.set_routing_config(routing_config)

            adjustments = trainer.corrector.get_adjustments()
            trainer.adaptive_loss.update_weights(adjustments)

            images = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
            if isinstance(batch, (tuple, list)):
                targets = [t.to(device) for t in batch[1]] if isinstance(batch[1], list) else batch[1].to(device)
            else:
                targets = None

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = model(images)
                loss = trainer.adaptive_loss(preds, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    trainer.optimizer.zero_grad()
                    continue

                trainer.corrector.record_error(preds, targets, ious=None)

            scaled_loss = loss / accumulation_steps
            trainer.scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
                trainer.optimizer.zero_grad()
                trainer.ema.update(model)

            batch_loss = loss.item()
            epoch_loss += batch_loss

            avg_loss = epoch_loss / (batch_idx + 1)
            lr_current = trainer.optimizer.param_groups[0]["lr"]
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
            pbar.set_postfix(
                {
                    "loss": f"{batch_loss:.4f}",
                    "avg": f"{avg_loss:.4f}",
                    "lr": f"{lr_current:.2e}",
                    "gpu": f"{gpu_mem:.1f}G",
                }
            )

        pbar.close()
        trainer.step_scheduler(epoch)

        # Epoch metrics
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        lr_now = trainer.optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - training_start
        gpu_mem_now = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0

        # Checkpoint
        is_best = avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "ema_state_dict": trainer.ema.ema.state_dict(),
            "ema_updates": trainer.ema.updates,
            "best_loss": best_loss,
            "training_args": config,
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")
        if is_best:
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        # Print snapshot
        print_snapshot(
            epoch,
            epochs,
            avg_epoch_loss,
            best_loss,
            lr_now,
            gpu_mem_now,
            epoch_time,
            elapsed,
        )

        if is_best:
            log.info("  >> New best loss: %.4f — saved checkpoint_best.pt", best_loss)

        # Validation metrics
        if val_loader is not None:
            num_classes = config.get("model", {}).get("num_classes", 80)
            if task == "classification":
                val_results = validate_classification(model, val_loader, device, num_classes=num_classes)
                top1 = val_results.get("top1", 0.0)
                top5 = val_results.get("top5", 0.0)
                log.info("  Val Top-1=%.4f  Top-5=%.4f", top1, top5)
            else:
                val_results = validate_detection(model, val_loader, device, num_classes=num_classes, img_size=img_size)
                mAP_50 = val_results.get("mAP_50", 0.0)
                mAP_50_95 = val_results.get("mAP_50_95", 0.0)
                log.info("  Val mAP@50=%.4f  mAP@50:95=%.4f", mAP_50, mAP_50_95)

        # Early stopping on loss (lower is better)
        if trainer.early_stopping.step(avg_epoch_loss):
            log.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    total_time = time.time() - training_start
    log.info("=" * 64)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 64)
    log.info("  Total Time : %s", _format_eta(total_time))
    log.info("  Best Loss  : %.4f", best_loss)
    log.info("  Checkpoints: %s", run_dir)
    log.info("=" * 64)


# ===================================================================
# 9. Synthetic demo dataset
# ===================================================================


def generate_synthetic_dataset(workspace: Path) -> dict:
    """Create a tiny COCO dataset with random images for demo purposes."""
    import numpy as np
    from PIL import Image

    data_dir = workspace / "synthetic_demo"
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    ann_id = 0
    for i in range(8):
        fname = f"img_{i:04d}.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img.save(str(img_dir / fname))
        coco["images"].append({"id": i, "file_name": fname, "width": 640, "height": 640})
        for _ in range(2):
            x = int(np.random.randint(0, 400))
            y = int(np.random.randint(0, 400))
            w = int(np.random.randint(30, 200))
            h = int(np.random.randint(30, 200))
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    ann_file = data_dir / "annotations.json"
    ann_file.write_text(json.dumps(coco))

    # Split: first 6 train, last 2 val
    train_ann = {
        "images": coco["images"][:6],
        "annotations": [a for a in coco["annotations"] if a["image_id"] < 6],
        "categories": coco["categories"],
    }
    val_ann = {
        "images": coco["images"][6:],
        "annotations": [a for a in coco["annotations"] if a["image_id"] >= 6],
        "categories": coco["categories"],
    }
    train_file = data_dir / "train_annotations.json"
    val_file = data_dir / "val_annotations.json"
    train_file.write_text(json.dumps(train_ann))
    val_file.write_text(json.dumps(val_ann))

    log.info("Synthetic demo dataset created at %s (8 images, 1 class)", data_dir)
    return {
        "train_img_dir": str(img_dir),
        "train_ann_file": str(train_file),
        "val_img_dir": str(img_dir),
        "val_ann_file": str(val_file),
    }


# ===================================================================
# 10. CLI entry point
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Badger_vision — Production-Ready Linux Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            All settings are loaded from notebooks/train_config.yaml by
            default.  CLI arguments override the config file.

            Dataset formats (auto-detected):
              Badger Factory - Keypoint Detection   yolo/images/*.7z + labels/*.7z
              Badger Factory - Object Detection      (same layout)
              Badger Factory - Image Classification  classifier/evolving_ds_*/{train,val}.7z
              COCO export                             coco/{train,val}.7z with coco_instances.json
              Plain YOLO / COCO                       already extracted on disk

            Examples:
              %(prog)s                                          # uses train_config.yaml
              %(prog)s /data/badger_factory_export/ --task detection
              %(prog)s /data/my_dataset.7z --task keypoints --epochs 200
              %(prog)s /data/classifier_export/ --task classification
              %(prog)s /data/coco_export/ --task detection --model convnext
              %(prog)s /data/dataset.zip --task detection --resume runs/train_*/checkpoint_last.pt
        """),
    )
    parser.add_argument(
        "dataset",
        type=str,
        nargs="?",
        default=None,
        help="Path to dataset folder or archive. Omit to use train_config.yaml dataset_path (or synthetic demo).",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to train_config.yaml (auto-detected if omitted)")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["detection", "keypoints", "classification"],
        help="Training task (uses config value if omitted)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (uses config value if omitted)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (0 = auto from GPU VRAM)")
    parser.add_argument("--img-size", type=int, default=None, help="Image size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--model", type=str, default=None, choices=["resnext", "convnext"], help="Architecture")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--workspace", type=str, default=None, help="Working directory")
    parser.add_argument("--no-setup", action="store_true", help="Skip auto-setup")

    args = parser.parse_args()

    # ── Load config (train_config.yaml) ──
    tcfg = load_train_config(args.config)

    # CLI args override config values
    dataset_arg = args.dataset
    if dataset_arg is None and tcfg.get("dataset_path"):
        dataset_arg = tcfg["dataset_path"]

    task = args.task or tcfg.get("task")
    epochs = args.epochs if args.epochs is not None else tcfg.get("epochs", 100)
    batch_size = args.batch_size if args.batch_size is not None else tcfg.get("batch_size", 0)
    img_size = args.img_size if args.img_size is not None else tcfg.get("img_size", 640)
    lr = args.lr if args.lr is not None else tcfg.get("lr", 0.01)
    model_type = args.model or tcfg.get("model", "resnext")
    resume = args.resume or tcfg.get("resume") or None
    workspace_str = args.workspace or tcfg.get("workspace", "badger_vision_workspace")

    # ── Auto-setup ──
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if not args.no_setup:
        auto_setup(repo_root)

    # ── Device (from config) ──
    device = resolve_device(tcfg.get("device", "auto"))
    import torch

    if device.type != "cuda":
        log.warning(
            'No GPU will be used — training will be slow!\n  Set device: "auto" in train_config.yaml to enable GPU.'
        )

    # ── Task picker ──
    if task is None:
        print("\n  What would you like to train?\n")
        print("    1) Object Detection")
        print("    2) Keypoint Detection")
        print("    3) Image Classification")
        print()
        choice = input("  Enter choice [1/2/3]: ").strip()
        task_map = {"1": "detection", "2": "keypoints", "3": "classification"}
        task = task_map.get(choice)
        if task is None:
            log.error("Invalid choice: %s", choice)
            sys.exit(1)
    log.info("Task: %s", task)

    # ── Workspace ──
    workspace = Path(workspace_str).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──
    use_synthetic = not dataset_arg
    if use_synthetic:
        log.info("No dataset supplied — generating synthetic demo data")
        if task is None:
            task = "detection"
            log.info("Defaulting task to: detection")
        data_info = generate_synthetic_dataset(workspace)
    else:
        dataset_path = Path(dataset_arg).resolve()
        if not dataset_path.exists():
            log.error("Dataset path does not exist: %s", dataset_path)
            sys.exit(1)

        # Extract top-level archive if needed
        if dataset_path.is_file() and is_archive(dataset_path):
            extract_dest = workspace / "dataset"
            if extract_dest.exists():
                shutil.rmtree(extract_dest)
            dataset_root = extract_archive(dataset_path, extract_dest)
        else:
            dataset_root = dataset_path

        # Navigate into the right sub-folder for the task
        dataset_root = resolve_dataset_root(dataset_root, task)
        log.info("Dataset root: %s", dataset_root)

        # ── Detect format & prepare ──
        fmt = detect_format(dataset_root)
        log.info("Detected format: %s", fmt)

        if fmt == "badger_yolo":
            classes_txt = dataset_root / "classes.txt"
            if not classes_txt.exists():
                classes_txt = None
            data_info = prepare_badger_yolo(dataset_root, workspace, classes_txt)
        elif fmt == "badger_classifier":
            data_info = prepare_badger_classifier(dataset_root, workspace)
        elif fmt == "coco_archive":
            data_info = prepare_coco_archive(dataset_root, workspace)
        elif fmt == "yolo_flat":
            data_info = prepare_yolo_flat(dataset_root, workspace)
        elif fmt == "coco_flat":
            data_info = prepare_coco_flat(dataset_root, workspace)
        else:
            log.error(
                "Could not detect dataset format in %s\n"
                "  Expected one of:\n"
                "    - Badger Factory YOLO:   images/{train,val}.7z + labels/{train,val}.7z\n"
                "    - Badger Factory Class.: evolving_ds_*/{train,val}.7z\n"
                "    - COCO archive:           {train,val}.7z with coco_instances.json\n"
                "    - Plain YOLO:             images/ + labels/ dirs\n"
                "    - Plain COCO:             annotations.json + images/",
                dataset_root,
            )
            sys.exit(1)

    log.info("Train: %s", data_info["train_img_dir"])
    if data_info.get("val_img_dir"):
        log.info("Val  : %s", data_info["val_img_dir"])

    # ── Num classes & batch size ──
    num_classes = detect_num_classes(data_info["train_ann_file"])
    num_train = count_images(data_info["train_ann_file"])
    log.info("Classes: %d  |  Training images: %d", num_classes, num_train)

    if batch_size <= 0:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            batch_size = auto_batch_size(gpu_mem, img_size)
            log.info("Auto batch size: %d  (%.1f GB VRAM)", batch_size, gpu_mem)
        else:
            batch_size = 2
            log.info("CPU mode — batch size: %d", batch_size)

    # ── Configs ──
    model_cfg, data_cfg = write_configs(
        workspace,
        data_info,
        num_classes,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        lr=lr,
        model_type=model_type,
        train_config=tcfg,
    )

    # ── Train ──
    run_training(
        model_cfg,
        data_cfg,
        data_info,
        device,
        epochs=epochs,
        batch_size=batch_size,
        resume=resume,
        task=task,
        train_config=tcfg,
    )


if __name__ == "__main__":
    main()
