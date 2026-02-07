"""Run crowd counting inference on a single image.

Usage:
    python scripts/infer.py --image path/to/image.jpg --checkpoint weights/best_model.pth
    python scripts/infer.py --image photo.jpg --checkpoint weights/best_model.pth --save-output
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

from crowd_counting.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from crowd_counting.engine.evaluator import Evaluator
from crowd_counting.models.csrnet import CSRNet
from crowd_counting.utils.checkpoint import load_checkpoint
from crowd_counting.utils.device import get_device
from crowd_counting.utils.visualization import plot_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run crowd counting on a single image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save the visualization to a file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the output visualization (default: <image_name>_prediction.png)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the result (useful for headless environments)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = get_device()
    logging.info(f"Using device: {device}")

    # Load model
    model = CSRNet(load_weights=True)
    model = model.to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    # Load and preprocess image
    img_pil = Image.open(args.image).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img_tensor = transform(img_pil)

    # Run inference
    evaluator = Evaluator(model, device)
    predicted_count, density_map = evaluator.predict_single(img_tensor)

    print(f"\n{'=' * 40}")
    print(f"  Image:           {args.image}")
    print(f"  Predicted Count: {predicted_count:.0f}")
    print(f"{'=' * 40}\n")

    # Determine save path
    save_path = None
    if args.save_output:
        if args.output_path:
            save_path = args.output_path
        else:
            stem = Path(args.image).stem
            save_path = f"{stem}_prediction.png"

    # Create visualization
    fig = plot_prediction(
        np.array(img_pil),
        density_map,
        predicted_count,
        save_path=save_path,
    )

    if save_path:
        print(f"Visualization saved to {save_path}")

    if not args.no_display:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()

