"""Train a crowd counting model.

Usage:
    python scripts/train.py --config configs/shb.yaml
    python scripts/train.py --config configs/sha.yaml --wandb
    python scripts/train.py --config configs/shb.yaml --checkpoint weights/checkpoint.pth
"""

from __future__ import annotations

import argparse
import logging
import sys

from crowd_counting.config import load_config
from crowd_counting.engine.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a crowd counting model (CSRNet)",
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
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.wandb:
        config.wandb.enabled = True
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr

    logging.info(f"Config: dataset={config.dataset.name}, epochs={config.training.epochs}")

    # Create trainer
    trainer = Trainer(config)

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_pretrained(args.checkpoint)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()

