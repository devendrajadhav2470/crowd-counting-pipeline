"""Device detection and management utilities."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Detect the best available compute device.

    Returns:
        torch.device: CUDA if available, then MPS (Apple Silicon), else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

