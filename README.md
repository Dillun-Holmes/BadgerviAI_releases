<p align="center">
  <h1 align="center">BadgerAI Releases</h1>
  <p align="center">
    <strong>Download and install BadgerviAI directly — no source code needed.</strong>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

---

## Installation

### Option A — Install directly with pip (recommended)

```bash
pip install https://github.com/Dillun-Holmes/BadgerviAI_releases/releases/download/v4.3.2/badger_vision-4.3.2-py3-none-any.whl
```

### Option B — Download and install locally

1. Go to the [Releases page](https://github.com/Dillun-Holmes/BadgerviAI_releases/releases)
2. Download the `.whl` file from the latest release
3. Install with pip:

```bash
pip install badger_vision-4.3.2-py3-none-any.whl
```

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0

---

## Getting Started

After installing, use the CLI:

```bash
badger --help
```

Or use the Python API:

```python
from badger_vision.core.api import Badger_vision

model = Badger_vision("path/to/config.yaml")
predictions = model.predict("path/to/image.jpg")
```

---

## Training Notebooks

Ready-to-run notebooks are included in the [`notebooks/`](notebooks/) folder. These are automatically synced from the [main repository](https://github.com/Dillun-Holmes/AI_vision_model).

| Notebook | Platform | Description |
|----------|----------|-------------|
| [colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb) | Google Colab | One-click training on free GPU — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dillun-Holmes/BadgerviAI_releases/blob/main/notebooks/colab_quickstart.ipynb) |
| [kaggle_quickstart.ipynb](notebooks/kaggle_quickstart.ipynb) | Kaggle | Train on Kaggle's free T4 GPUs |
| [local_quickstart.ipynb](notebooks/local_quickstart.ipynb) | Local PC | Windows / macOS / Linux — GPU or CPU |
| [linux_train.ipynb](notebooks/linux_train.ipynb) | Linux Server | Production training with venv auto-setup |
| [training_demo.ipynb](notebooks/training_demo.ipynb) | Any | Quick training walkthrough |
| [inference_demo.ipynb](notebooks/inference_demo.ipynb) | Any | Run inference & benchmark a trained model |
| [dataset_analytics.ipynb](notebooks/dataset_analytics.ipynb) | Any | Analyze your dataset before training |
| [train_config.yaml](notebooks/train_config.yaml) | Any | **Central config file** — set dataset, task, model, augmentation, epochs here |

All notebooks support Phantom Factory archives (`.7z`, `.zip`, etc.), COCO, and YOLO datasets. Leave the dataset path empty to run a quick demo with synthetic data.

### Configuration

All training settings are now managed from a single config file: [`notebooks/train_config.yaml`](notebooks/train_config.yaml)

Edit it to set:
- **Dataset path** — point to your dataset folder or archive
- **Task type** — `detection`, `keypoints`, or `classification`
- **Model** — `resnext` or `convnext` with variant/size (e.g. `resnext_nano`, `resnext_large`)
- **Epochs, batch size, learning rate, image size**
- **Augmentation** — `true`/`false` master toggle + fine-grained controls
- **Device** — `auto` (GPU if available), `cpu`, or specific GPU IDs

Then just run any notebook — it reads the config automatically. CLI arguments still override config values when provided.

---

## What is BadgerviAI?

BadgerviAI is a production-grade object detection framework featuring:

- **Query-based NMS-free detection** with Hungarian matching
- **Self-correcting training** that tracks and fixes its own failure modes
- **Sparse token routing** for 3–5× compute reduction
- **Evidential uncertainty estimation** — the model knows when it's uncertain
- **Vision-language open-vocabulary detection** via CLIP
- **Edge deployment** support for Jetson, Raspberry Pi, and mobile

For full documentation and source code, see the [main repository](https://github.com/Dillun-Holmes/AI_vision_model).
