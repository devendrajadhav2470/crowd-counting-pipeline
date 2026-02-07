"""Evaluate a trained crowd counting model.

Usage:
    python scripts/evaluate.py --config configs/shb.yaml --checkpoint weights/best_model.pth
    python scripts/evaluate.py --config configs/sha.yaml --checkpoint weights/best_model.pth --save-viz
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from crowd_counting.config import load_config
from crowd_counting.data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_image_paths
from crowd_counting.engine.evaluator import Evaluator
from crowd_counting.models.csrnet import CSRNet
from crowd_counting.utils.checkpoint import load_checkpoint
from crowd_counting.utils.device import get_device
from crowd_counting.utils.visualization import plot_comparison_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained crowd counting model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/shb.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_data",
        choices=["test_data", "train_data"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualization of predictions",
    )
    parser.add_argument(
        "--num-viz",
        type=int,
        default=5,
        help="Number of sample predictions to visualize",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    device = get_device()

    # Load model
    model = CSRNet(load_weights=True)
    model = model.to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    # Evaluate
    evaluator = Evaluator(model, device)
    results = evaluator.evaluate_dataset(
        config.dataset.root,
        split=args.split,
        batch_size=config.evaluation.batch_size,
    )

    print("\n" + "=" * 50)
    print(f"  Dataset:  {config.dataset.name} ({args.split})")
    print(f"  Images:   {results['count']}")
    print(f"  MAE:      {results['mae']:.3f}")
    print(f"  RMSE:     {results['rmse']:.3f}")
    print("=" * 50)

    # Visualize sample predictions
    if args.save_viz:
        split_path = os.path.join(config.dataset.root, args.split)
        image_paths = get_image_paths(split_path)[: args.num_viz]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        images = []
        density_maps = []
        pred_counts = []
        gt_counts = []

        for img_path in image_paths:
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = transform(img_pil)

            pred_count, dm = evaluator.predict_single(img_tensor)

            images.append(np.array(img_pil))
            density_maps.append(dm)
            pred_counts.append(pred_count)

        save_path = os.path.join(config.output.dir, "evaluation_samples.png")
        plot_comparison_grid(
            images, density_maps, pred_counts,
            save_path=save_path, max_samples=args.num_viz,
        )
        print(f"\nVisualization saved to {save_path}")


if __name__ == "__main__":
    main()

