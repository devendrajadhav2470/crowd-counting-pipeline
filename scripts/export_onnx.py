"""Export a trained CSRNet model to ONNX format.

Usage:
    python scripts/export_onnx.py --checkpoint weights/best_model.pth
    python scripts/export_onnx.py --checkpoint weights/best_model.pth --output model.onnx
"""

from __future__ import annotations

import argparse
import logging

import torch

from crowd_counting.models.csrnet import CSRNet
from crowd_counting.utils.checkpoint import load_checkpoint
from crowd_counting.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CSRNet to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/csrnet.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=384,
        help="Input image height for the ONNX model",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=512,
        help="Input image width for the ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load model on CPU for export
    device = torch.device("cpu")
    model = CSRNet(load_weights=True)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, args.input_height, args.input_width)

    # Export
    logger.info(
        f"Exporting to ONNX (opset={args.opset}, "
        f"input={args.input_height}x{args.input_width})..."
    )

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=args.opset,
        input_names=["image"],
        output_names=["density_map"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "density_map": {0: "batch", 2: "height", 3: "width"},
        },
    )

    logger.info(f"ONNX model saved to {args.output}")

    # Verify
    try:
        import onnx

        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")
    except ImportError:
        logger.info("Install 'onnx' package to verify the exported model")


if __name__ == "__main__":
    main()

