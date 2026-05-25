#!/usr/bin/env python3
"""Robust launcher for Badger_vision training."""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[BadgerVision] {msg}", flush=True)


# -------------------------------------------------------------------
# Ensure correct virtual environment
# -------------------------------------------------------------------

def ensure_venv() -> None:
    current_python = Path(sys.executable).resolve()

    if not VENV_PYTHON.exists():
        log("WARNING: .venv not found.")
        return

    venv_python = VENV_PYTHON.resolve()

    if current_python == venv_python:
        return

    log(f"Switching to venv Python:")
    log(str(venv_python))

    os.execv(
        str(venv_python),
        [str(venv_python), *sys.argv],
    )


# -------------------------------------------------------------------
# Validate Python version
# -------------------------------------------------------------------

def validate_python() -> None:
    major, minor = sys.version_info[:2]

    log(f"Python version detected: {major}.{minor}")

    if major != 3 or minor >= 13:
        log("WARNING:")
        log("PyTorch training stability on Python >=3.13 is poor.")
        log("Recommended versions:")
        log("  - Python 3.10")
        log("  - Python 3.11")


# -------------------------------------------------------------------
# Add repo root
# -------------------------------------------------------------------

def setup_paths() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


# -------------------------------------------------------------------
# Torch diagnostics
# -------------------------------------------------------------------

def torch_diagnostics() -> None:
    start = time.time()

    log("Importing torch...")

    try:
        import torch

        elapsed = time.time() - start

        log(f"Torch imported successfully in {elapsed:.2f}s")
        log(f"Torch version: {torch.__version__}")

        if torch.cuda.is_available():
            log(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            log("CUDA not available")

    except KeyboardInterrupt:
        log("Torch import interrupted.")
        log("Possible causes:")
        log("  - Python version incompatibility")
        log("  - Broken torch installation")
        log("  - CUDA initialization stall")
        log("  - Corrupted virtual environment")

        sys.exit(1)

    except Exception:
        log("Torch import failed.")
        traceback.print_exc()
        sys.exit(1)


# -------------------------------------------------------------------
# Main startup
# -------------------------------------------------------------------

def bootstrap() -> None:
    ensure_venv()
    validate_python()
    setup_paths()
    torch_diagnostics()


bootstrap()

# -------------------------------------------------------------------
# Import pipeline AFTER diagnostics
# -------------------------------------------------------------------

try:
    from badger_vision.training.pipeline import *  # noqa: F401,F403,E402
    from badger_vision.training.pipeline import main  # noqa: E402

except Exception:
    log("Failed to import training pipeline.")
    traceback.print_exc()
    sys.exit(1)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        log("Training interrupted safely.")
        sys.exit(0)

    except Exception:
        log("Fatal training error.")
        traceback.print_exc()
        sys.exit(1)