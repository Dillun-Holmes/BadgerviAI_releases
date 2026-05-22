<p align="center">
  <h1 align="center">BadgerviAI Releases</h1>
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
pip install https://github.com/Dillun-Holmes/BadgerviAI_releases/releases/download/v1.0.0/badgerviai-4.0.0-py3-none-any.whl
```

### Option B — Download and install locally

1. Go to the [Releases page](https://github.com/Dillun-Holmes/BadgerviAI_releases/releases)
2. Download the `.whl` file from the latest release
3. Install with pip:

```bash
pip install badgerviai-4.0.0-py3-none-any.whl
```

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0

---

## Getting Started

After installing, use the CLI:

```bash
badgerviai --help
```

Or use the Python API:

```python
from badgerviai.core.api import BadgerviAI

model = BadgerviAI("path/to/config.yaml")
predictions = model.predict("path/to/image.jpg")
```

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
