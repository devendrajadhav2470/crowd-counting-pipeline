"""Model checkpoint save/load utilities."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: dict[str, Any],
    is_best: bool,
    checkpoint_dir: str | Path = "./weights",
    filename: str = "checkpoint.pth",
) -> None:
    """Save a training checkpoint.

    Args:
        state: Dictionary containing model state_dict, optimizer state_dict,
            epoch, best metric, etc.
        is_best: If True, also copy this checkpoint as 'best_model.pth'.
        checkpoint_dir: Directory to save checkpoints.
        filename: Name of the checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        shutil.copyfile(filepath, best_path)
        logger.info(f"Best model saved to {best_path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state_dict into.
        optimizer: Optional optimizer to load state_dict into.
        device: Device to map the checkpoint to (e.g., 'cpu' for local inference).

    Returns:
        The full checkpoint dictionary (contains epoch, best_prec1, etc.).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["state_dict"])
    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Optimizer state loaded")

    return checkpoint

