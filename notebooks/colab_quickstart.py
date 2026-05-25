#!/usr/bin/env python3
# -----------------------------------------------------------
# Badger_vision — Google Colab Quickstart (Python script)
#
# Supports Badger Factory datasets (YOLO / COCO / classifier),
# plain zip/7z/tar archives, or already-extracted folders.
#
# Usage in Colab:
#   !pip install -q https://github.com/Dillun-Holmes/BadgerviAI_releases/releases/download/v4.1.0/badger_vision-4.1.0-py3-none-any.whl
#   %run colab_quickstart.py
# -----------------------------------------------------------
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# ── 0. Auto-install extras ─────────────────────────────────
for _pkg in ["tqdm", "py7zr", "rarfile"]:
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

# ── 1. GPU check ───────────────────────────────────────────
print(f"PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"CUDA    : {torch.version.cuda} — {gpu_name} ({gpu_mem:.1f} GB)")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda:0")
else:
    print("WARNING: No GPU detected! Go to Runtime > Change runtime type > GPU")
    print("         Training will be extremely slow on CPU.")
    DEVICE = torch.device("cpu")

# ── 2. Load shared training helpers ────────────────────────
# Clone repo if not present (needed for shared pipeline helpers)
REPO_DIR = Path("/content/AI_vision_model")
if not REPO_DIR.exists():
    subprocess.check_call(
        ["git", "clone", "--depth=1", "https://github.com/Dillun-Holmes/AI_vision_model.git", str(REPO_DIR)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from badger_vision.training import pipeline as lt  # noqa: E402

from badger_vision import Badger_vision  # noqa: E402
from badger_vision.models.badger_resnext import BadgerResNeXtModel  # noqa: E402
from badger_vision.utils.profiler import model_summary  # noqa: E402

# ── 3. Configuration (loaded from train_config.yaml) ──────
TCFG = lt.load_train_config()

DATASET_PATH = TCFG.get("dataset_path", "")
TASK = TCFG.get("task", "detection")
EPOCHS = TCFG.get("epochs", 3)
BATCH_SIZE = TCFG.get("batch_size", 0)
IMG_SIZE = TCFG.get("img_size", 640)
LR = TCFG.get("lr", 0.01)
MODEL_TYPE = TCFG.get("model", "resnext")
DEVICE = lt.resolve_device(TCFG.get("device", "auto"))

# ── 4. Prepare dataset ─────────────────────────────────────
ROOT = Path("/content/badger_vision_demo")
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
IMGS = DATA / "images"
WORKSPACE = ROOT / "workspace"

if DATASET_PATH and Path(DATASET_PATH).exists():
    dataset_path = Path(DATASET_PATH).resolve()
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    if dataset_path.is_file() and lt.is_archive(dataset_path):
        import shutil

        extract_dest = WORKSPACE / "dataset"
        if extract_dest.exists():
            shutil.rmtree(extract_dest)
        dataset_root = lt.extract_archive(dataset_path, extract_dest)
    else:
        dataset_root = dataset_path

    dataset_root = lt.resolve_dataset_root(dataset_root, TASK)
    fmt = lt.detect_format(dataset_root, TASK)
    print(f"Detected format: {fmt}")

    classes_txt = dataset_root / "classes.txt"
    data_info = lt.prepare_dataset(
        dataset_root,
        WORKSPACE,
        fmt=fmt,
        classes_txt=classes_txt if classes_txt.exists() else None,
    )

    num_classes = lt.detect_num_classes(data_info["train_ann_file"])
    num_train = lt.count_images(data_info["train_ann_file"])
    print(f"Classes: {num_classes}  |  Training images: {num_train}")

    if BATCH_SIZE <= 0 and torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        BATCH_SIZE = lt.auto_batch_size(gpu_mem, IMG_SIZE)
        print(f"Auto batch size: {BATCH_SIZE}")
    elif BATCH_SIZE <= 0:
        BATCH_SIZE = 2

    model_cfg, data_cfg = lt.write_configs(
        WORKSPACE,
        data_info,
        num_classes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        lr=LR,
        model_type=MODEL_TYPE,
        train_config=TCFG,
    )
    print(f"\nStarting training: {TASK} | {EPOCHS} epochs | batch {BATCH_SIZE}")
    lt.run_training(
        model_cfg, data_cfg, data_info, DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE, task=TASK, train_config=TCFG
    )

else:
    # Synthetic demo dataset
    print("\nNo DATASET_PATH set — using synthetic demo data.")
    CONFIGS.mkdir(parents=True, exist_ok=True)
    IMGS.mkdir(parents=True, exist_ok=True)

    (CONFIGS / "resnext_nano.yaml").write_text(
        textwrap.dedent("""\
        model:
          name: "BadgerResNeXt-Nano"
          type: "resnext"
          backbone: "resnext_nano"
          neck_channels: 64
          head_depth: 2
          use_ghost: true
          num_classes: 80
          image_size: 640
        training:
          epochs: 3
          base_lr: 0.01
          warmup_epochs: 1
          accumulation_steps: 1
          early_stopping_patience: 30
    """)
    )

    (CONFIGS / "convnext_nano.yaml").write_text(
        textwrap.dedent("""\
        model:
          name: "Badger_vision-Nano"
          backbone: "convnext_tiny"
          channels: 64
          transformer_depth: 2
          token_count: 100
          image_size: 640
        training:
          epochs: 3
    """)
    )

    coco: dict = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
    ann_id = 0
    for i in range(4):
        fname = f"img_{i:04d}.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img.save(str(IMGS / fname))
        coco["images"].append({"id": i, "file_name": fname, "width": 640, "height": 640})
        for _ in range(2):
            x, y = int(np.random.randint(0, 400)), int(np.random.randint(0, 400))
            w, h = int(np.random.randint(30, 200)), int(np.random.randint(30, 200))
            coco["annotations"].append(
                {"id": ann_id, "image_id": i, "category_id": 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0}
            )
            ann_id += 1

    (DATA / "annotations.json").write_text(json.dumps(coco))
    (CONFIGS / "data.yaml").write_text(
        textwrap.dedent(f"""\
        img_dir: "{IMGS}"
        ann_file: "{DATA / "annotations.json"}"
    """)
    )
    print("Synthetic data created.")

    # Model profiling
    for variant in ["resnext_pico", "resnext_nano", "resnext_small"]:
        m = BadgerResNeXtModel(variant=variant, num_classes=80)
        s = model_summary(m)
        print(f"{variant:>16s}  |  {s['params_M']}M params  |  {s['flops_G']} GFLOPs  |  {s['size_mb']} MB")

    # Training
    model = Badger_vision(str(CONFIGS / "resnext_nano.yaml"))
    model.train(data=str(CONFIGS / "data.yaml"), task="detection")

    # Inference
    preds = model.predict(str(IMGS / "img_0000.jpg"))
    for key, val in preds.items():
        print(f"{key}: shape={val.shape}, dtype={val.dtype}")

    # Benchmark
    results = model.benchmark(warmup=3, runs=20)
    print(f"Latency : {results['avg_latency_ms']} ms  |  FPS: {results['fps']}")

    # Export
    model.export(format="onnx")
    size_mb = os.path.getsize("model.onnx") / (1024 * 1024)
    print(f"Exported model.onnx ({size_mb:.1f} MB)")

print("\nDone! Set DATASET_PATH to train on your own data.")
