"""Visualization utilities for crowd counting results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm as colormap
from PIL import Image


def overlay_density_map(
    image: np.ndarray,
    density_map: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
) -> np.ndarray:
    """Overlay a density map heatmap on top of the original image.

    Args:
        image: Original image as numpy array (H, W, 3), values in [0, 255].
        density_map: Density map as numpy array (H', W').
        alpha: Opacity of the heatmap overlay.
        cmap: Matplotlib colormap name.

    Returns:
        Blended image as numpy array (H, W, 3) in [0, 255].
    """
    import cv2

    # Resize density map to match image dimensions
    if density_map.shape[:2] != image.shape[:2]:
        density_map = cv2.resize(
            density_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC
        )

    # Normalize density map to [0, 1] for colormap
    dm_min, dm_max = density_map.min(), density_map.max()
    if dm_max > dm_min:
        dm_normalized = (density_map - dm_min) / (dm_max - dm_min)
    else:
        dm_normalized = np.zeros_like(density_map)

    # Apply colormap
    cm = plt.get_cmap(cmap)
    heatmap = cm(dm_normalized)[:, :, :3]  # Drop alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return blended


def plot_prediction(
    image: np.ndarray | Image.Image,
    density_map: np.ndarray,
    predicted_count: float,
    ground_truth_count: Optional[float] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Create a side-by-side visualization of image, density map, and overlay.

    Args:
        image: Original image (PIL Image or numpy array).
        density_map: Predicted density map.
        predicted_count: Predicted crowd count (sum of density map).
        ground_truth_count: Optional ground truth count for comparison.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    overlay = overlay_density_map(image, density_map)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    # Density map
    axes[1].imshow(density_map, cmap="jet")
    count_text = f"Predicted: {predicted_count:.0f}"
    if ground_truth_count is not None:
        count_text += f" | GT: {ground_truth_count:.0f}"
    axes[1].set_title(count_text, fontsize=12)
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comparison_grid(
    images: list[np.ndarray],
    density_maps: list[np.ndarray],
    predicted_counts: list[float],
    ground_truth_counts: Optional[list[float]] = None,
    save_path: Optional[str | Path] = None,
    max_samples: int = 5,
) -> plt.Figure:
    """Create a grid of predictions for multiple samples.

    Args:
        images: List of original images.
        density_maps: List of predicted density maps.
        predicted_counts: List of predicted counts.
        ground_truth_counts: Optional list of ground truth counts.
        save_path: Optional path to save the figure.
        max_samples: Maximum number of samples to show.

    Returns:
        Matplotlib Figure object.
    """
    n = min(len(images), max_samples)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        img = images[i]
        dm = density_maps[i]
        overlay = overlay_density_map(img, dm)

        axes[i][0].imshow(img)
        axes[i][0].set_title("Original Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(dm, cmap="jet")
        title = f"Pred: {predicted_counts[i]:.0f}"
        if ground_truth_counts:
            title += f" | GT: {ground_truth_counts[i]:.0f}"
        axes[i][1].set_title(title)
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay)
        axes[i][2].set_title("Overlay")
        axes[i][2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

