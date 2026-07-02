from __future__ import annotations

import os
import platform
import random
import subprocess
from typing import Any

import numpy as np


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        return


def capture_repro_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("datasets", "numpy", "pydantic", "torch", "torchvision"):
        try:
            module = __import__(package)
            meta[f"{package}_version"] = getattr(module, "__version__", "unknown")
        except Exception:
            pass
    git_commit = _git_commit()
    if git_commit:
        meta["git_commit"] = git_commit
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        meta["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return meta


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
