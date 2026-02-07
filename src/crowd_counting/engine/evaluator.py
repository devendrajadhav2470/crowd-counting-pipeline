"""Evaluation engine for crowd counting models.

Provides comprehensive evaluation metrics (MAE, RMSE) and visualization
of predictions versus ground truth density maps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crowd_counting.data.dataset import CrowdDataset, get_default_transform, get_image_paths
from crowd_counting.utils.device import get_device

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate a crowd counting model on a dataset.

    Args:
        model: The crowd counting model.
        device: Torch device to run evaluation on.
    """

    def __init__(
        self, model: nn.Module, device: Optional[torch.device] = None
    ) -> None:
        self.model = model
        self.device = device or get_device()

    @torch.no_grad()
    def evaluate_paths(
        self,
        image_paths: list[str],
        batch_size: int = 1,
    ) -> tuple[float, float]:
        """Evaluate the model on a list of image paths.

        Args:
            image_paths: List of image file paths.
            batch_size: Batch size for evaluation.

        Returns:
            Tuple of (MAE, RMSE).
        """
        dataset = CrowdDataset(
            image_paths,
            transform=get_default_transform(),
            train=False,
            shuffle=False,
        )
        loader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()

        mae_sum = 0.0
        mse_sum = 0.0
        count = 0

        for img, target in loader:
            img = img.to(self.device)
            output = self.model(img)

            predicted_count = output.data.sum().item()
            gt_count = target.sum().item()

            error = abs(predicted_count - gt_count)
            mae_sum += error
            mse_sum += error ** 2
            count += 1

        mae = mae_sum / max(count, 1)
        rmse = (mse_sum / max(count, 1)) ** 0.5

        return mae, rmse

    @torch.no_grad()
    def evaluate_dataset(
        self,
        data_path: str | Path,
        split: str = "test_data",
        batch_size: int = 1,
    ) -> dict[str, float]:
        """Evaluate the model on a dataset split.

        Args:
            data_path: Root path of the dataset (e.g., `data/ShanghaiTech/part_B`).
            split: Either 'test_data' or 'train_data'.
            batch_size: Batch size for evaluation.

        Returns:
            Dictionary with 'mae', 'rmse', and 'count' keys.
        """
        import os

        split_path = os.path.join(str(data_path), split)
        image_paths = get_image_paths(split_path)

        if not image_paths:
            logger.warning(f"No images found at {split_path}")
            return {"mae": 0.0, "rmse": 0.0, "count": 0}

        logger.info(f"Evaluating on {len(image_paths)} images from {split_path}")
        mae, rmse = self.evaluate_paths(image_paths, batch_size)

        logger.info(f"Results — MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        return {"mae": mae, "rmse": rmse, "count": len(image_paths)}

    @torch.no_grad()
    def predict_single(
        self,
        image: torch.Tensor,
    ) -> tuple[float, np.ndarray]:
        """Run inference on a single preprocessed image.

        Args:
            image: Preprocessed image tensor of shape (3, H, W) or (1, 3, H, W).

        Returns:
            Tuple of (predicted_count, density_map as numpy array).
        """
        self.model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        output = self.model(image)

        density_map = output.squeeze().cpu().numpy()
        predicted_count = float(density_map.sum())

        return predicted_count, density_map

