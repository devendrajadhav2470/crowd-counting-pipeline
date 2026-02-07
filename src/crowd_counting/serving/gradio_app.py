"""Gradio web demo for crowd counting.

Provides an interactive UI where users can upload images and get
crowd count predictions with density map visualizations.

Usage:
    python -m crowd_counting.serving.gradio_app
    python -m crowd_counting.serving.gradio_app --checkpoint weights/best_model.pth --port 7860
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lazy-loaded globals
_model = None
_device = None
_transform = None


def _load_model(checkpoint_path: str) -> None:
    """Load the model (called once at startup)."""
    global _model, _device, _transform

    from crowd_counting.data.dataset import IMAGENET_MEAN, IMAGENET_STD
    from crowd_counting.models.csrnet import CSRNet
    from crowd_counting.utils.checkpoint import load_checkpoint
    from crowd_counting.utils.device import get_device

    _device = get_device()
    logger.info(f"Loading model on {_device}...")

    _model = CSRNet(load_weights=True)
    _model = _model.to(_device)
    load_checkpoint(checkpoint_path, _model, device=_device)
    _model.eval()

    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    logger.info("Model loaded successfully")


def predict(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Run crowd counting on an uploaded image.

    Args:
        image: Input image as numpy array (H, W, 3) from Gradio.

    Returns:
        Tuple of (density map overlay image, count text).
    """
    if _model is None:
        return image, "Error: Model not loaded"

    # Convert to PIL for transforms
    img_pil = Image.fromarray(image).convert("RGB")
    img_tensor = _transform(img_pil)

    # Inference
    with torch.no_grad():
        img_input = img_tensor.unsqueeze(0).to(_device)
        output = _model(img_input)

    density_map = output.squeeze().cpu().numpy()
    predicted_count = float(density_map.sum())

    # Create density map overlay
    dm_resized = cv2.resize(
        density_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC
    )

    # Normalize for visualization
    dm_min, dm_max = dm_resized.min(), dm_resized.max()
    if dm_max > dm_min:
        dm_norm = (dm_resized - dm_min) / (dm_max - dm_min)
    else:
        dm_norm = np.zeros_like(dm_resized)

    # Apply jet colormap
    import matplotlib.pyplot as plt

    cm = plt.get_cmap("jet")
    heatmap = (cm(dm_norm)[:, :, :3] * 255).astype(np.uint8)

    # Blend with original
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    count_text = f"Estimated crowd count: {predicted_count:.0f}"

    return overlay, count_text


def create_app(checkpoint_path: str = "weights/best_model.pth") -> "gr.Blocks":
    """Create and return the Gradio app.

    Args:
        checkpoint_path: Path to model checkpoint.

    Returns:
        Gradio Blocks app.
    """
    import gradio as gr

    _load_model(checkpoint_path)

    with gr.Blocks(
        title="Crowd Counting with CSRNet",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Crowd Counting with CSRNet

            Upload an image of a crowd to estimate the number of people.
            The model generates a **density map** whose integral equals the predicted count.

            **Model**: CSRNet (VGG-16 backbone + dilated convolutions)
            **Dataset**: ShanghaiTech
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                )
                submit_btn = gr.Button("Count Crowd", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Density Map Overlay")
                output_text = gr.Textbox(label="Result", lines=1)

        submit_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[output_image, output_text],
        )

        # Also trigger on image upload
        input_image.change(
            fn=predict,
            inputs=[input_image],
            outputs=[output_image, output_text],
        )

        gr.Markdown(
            """
            ---
            **How it works**: The model uses a VGG-16 frontend for feature extraction
            followed by dilated convolutional layers to produce a density map.
            Each pixel in the density map represents the estimated crowd density at that
            location. The total count is the sum of all pixel values.

            [Paper: CSRNet (CVPR 2018)](https://arxiv.org/abs/1712.03400) |
            [GitHub Repository](https://github.com/devendrajadhav2470/crowd-counting-pipeline)
            """
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Gradio crowd counting demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    demo = create_app(args.checkpoint)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

