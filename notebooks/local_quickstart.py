#!/usr/bin/env python3
# -----------------------------------------------------------
# PhantomVision — Local PC Quickstart (Python script)
#
# Works on Windows, macOS, and Linux (CPU, CUDA, or Apple MPS).
# Supports Phantom Factory datasets (YOLO / COCO / classifier),
# plain zip/7z/tar archives, or already-extracted folders.
#
# Setup (first time only):
#   python -m venv .venv && source .venv/bin/activate
#   pip install -e .          # from repo root
#   pip install tqdm py7zr rarfile
#   python notebooks/local_quickstart.py /path/to/dataset --task detection
#
# Or with no dataset (synthetic demo):
#   python notebooks/local_quickstart.py
# -----------------------------------------------------------
from __future__ import annotations

import argparse
import importlib.util
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
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", _pkg],
            stdout=subprocess.DEVNULL,
        )

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from phantomvision import PhantomVision  # noqa: E402
from phantomvision.models.phantom_resnext import PhantomResNeXtModel  # noqa: E402
from phantomvision.utils.profiler import model_summary  # noqa: E402

# ── 1. Detect device ──────────────────────────────────────
print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"CUDA    : {torch.version.cuda} — {gpu_name} ({gpu_mem:.1f} GB)")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Device  : Apple MPS")
    DEVICE = torch.device("mps")
else:
    print("Device  : CPU  (no GPU detected — training will be slow)")
    DEVICE = torch.device("cpu")

# ── 2. Load linux_train helpers ────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pyproject.toml").exists() else SCRIPT_DIR

spec = importlib.util.spec_from_file_location(
    "linux_train", str(SCRIPT_DIR / "linux_train.py"),
)
lt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lt)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PhantomVision — Local PC Quickstart",
    )
    p.add_argument(
        "dataset", nargs="?", default="",
        help="Path to dataset folder or archive (omit for synthetic demo)",
    )
    p.add_argument("--task", choices=["detection", "keypoints", "classification"],
                   default="detection")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--model", choices=["resnext", "convnext"], default="resnext")
    return p.parse_args()


def _run_real_dataset(args: argparse.Namespace) -> None:
    """Train on a real dataset (archive or folder)."""
    import shutil

    dataset_path = Path(args.dataset).resolve()
    workspace = Path("phantomvision_workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if dataset_path.is_file() and lt.is_archive(dataset_path):
        extract_dest = workspace / "dataset"
        if extract_dest.exists():
            shutil.rmtree(extract_dest)
        dataset_root = lt.extract_archive(dataset_path, extract_dest)
    else:
        dataset_root = dataset_path

    dataset_root = lt.resolve_dataset_root(dataset_root, args.task)
    fmt = lt.detect_format(dataset_root)
    print(f"Detected format: {fmt}")

    if fmt == "phantom_yolo":
        classes_txt = dataset_root / "classes.txt"
        data_info = lt.prepare_phantom_yolo(
            dataset_root, workspace,
            classes_txt if classes_txt.exists() else None,
        )
    elif fmt == "phantom_classifier":
        data_info = lt.prepare_phantom_classifier(dataset_root, workspace)
    elif fmt == "coco_archive":
        data_info = lt.prepare_coco_archive(dataset_root, workspace)
    elif fmt == "yolo_flat":
        data_info = lt.prepare_yolo_flat(dataset_root, workspace)
    elif fmt == "coco_flat":
        data_info = lt.prepare_coco_flat(dataset_root, workspace)
    else:
        raise RuntimeError(f"Unknown dataset format in {dataset_root}")

    num_classes = lt.detect_num_classes(data_info["train_ann_file"])
    num_train = lt.count_images(data_info["train_ann_file"])
    print(f"Classes: {num_classes}  |  Training images: {num_train}")

    batch_size = args.batch_size
    if batch_size <= 0 and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        batch_size = lt.auto_batch_size(vram, args.img_size)
        print(f"Auto batch size: {batch_size}")
    elif batch_size <= 0:
        batch_size = 2

    model_cfg, data_cfg = lt.write_configs(
        workspace, data_info, num_classes,
        epochs=args.epochs, batch_size=batch_size,
        img_size=args.img_size, lr=args.lr, model_type=args.model,
    )
    print(f"\nTraining: {args.task} | {args.epochs} epochs | batch {batch_size}")
    lt.run_training(model_cfg, data_cfg, data_info, DEVICE,
                    epochs=args.epochs, batch_size=batch_size)


def _run_synthetic_demo() -> None:
    """Run synthetic demo (profile, train, infer, benchmark, export)."""
    ROOT = Path("phantomvision_demo")
    CONFIGS = ROOT / "configs"
    DATA = ROOT / "data"
    IMGS = DATA / "images"
    CONFIGS.mkdir(parents=True, exist_ok=True)
    IMGS.mkdir(parents=True, exist_ok=True)

    (CONFIGS / "resnext_nano.yaml").write_text(textwrap.dedent("""\
        model:
          name: "PhantomResNeXt-Nano"
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
    """))

    (CONFIGS / "convnext_nano.yaml").write_text(textwrap.dedent("""\
        model:
          name: "PhantomVision-Nano"
          backbone: "convnext_tiny"
          channels: 64
          transformer_depth: 2
          token_count: 100
          image_size: 640
        training:
          epochs: 3
    """))

    coco: dict = {
        "images": [], "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
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
                {"id": ann_id, "image_id": i, "category_id": 1,
                 "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0},
            )
            ann_id += 1

    (DATA / "annotations.json").write_text(json.dumps(coco))
    (CONFIGS / "data.yaml").write_text(textwrap.dedent(f"""\
        img_dir: "{IMGS.resolve()}"
        ann_file: "{(DATA / 'annotations.json').resolve()}"
    """))
    print(f"Synthetic data created at: {ROOT.resolve()}")

    # Model profiling
    for variant in ["resnext_pico", "resnext_nano", "resnext_small"]:
        m = PhantomResNeXtModel(variant=variant, num_classes=80)
        s = model_summary(m)
        print(f"{variant:>16s}  |  {s['params_M']}M params  |  {s['flops_G']} GFLOPs  |  {s['size_mb']} MB")

    # Training
    model = PhantomVision(str(CONFIGS / "resnext_nano.yaml"))
    model.train(data=str(CONFIGS / "data.yaml"), task="detection")

    # Inference
    preds = model.predict(str(IMGS / "img_0000.jpg"))
    for key, val in preds.items():
        print(f"{key}: shape={val.shape}, dtype={val.dtype}")

    # Benchmark
    results = model.benchmark(warmup=3, runs=20)
    print(f"Latency : {results['avg_latency_ms']} ms  |  FPS: {results['fps']}")
    print(f"Params  : {results['params_M']}M  |  FLOPs: {results['flops_G']} G")

    # Dataset analysis
    report_path = model.analyze(data=str(CONFIGS / "data.yaml"))
    with open(report_path) as f:
        report = json.load(f)
    for key, val in report.items():
        print(f"{key}: {val}")

    # Export
    model.export(format="onnx")
    size_mb = os.path.getsize("model.onnx") / (1024 * 1024)
    print(f"Exported model.onnx ({size_mb:.1f} MB)")

    # ConvNeXt
    convnext_model = PhantomVision(str(CONFIGS / "convnext_nano.yaml"))
    preds = convnext_model.predict(str(IMGS / "img_0000.jpg"))
    for key, val in preds.items():
        print(f"{key}: shape={val.shape}")

    print("\nDone! Pass a dataset path as the first argument to train on real data.")


if __name__ == "__main__":
    args = _parse_args()
    if args.dataset and Path(args.dataset).exists():
        _run_real_dataset(args)
    else:
        if args.dataset:
            print(f"WARNING: Dataset not found: {args.dataset}")
            print("Falling back to synthetic demo.\n")
        _run_synthetic_demo()
