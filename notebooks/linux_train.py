#!/usr/bin/env python3
"""Thin launcher for the shared Badger_vision training pipeline.

Training logic lives in :mod:`badger_vision.training.pipeline` so notebooks
and quickstart scripts can share one implementation.  Keep user-editable
training settings in ``notebooks/train_config.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badger_vision.training.pipeline import *  # noqa: F401,F403,E402
from badger_vision.training.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
