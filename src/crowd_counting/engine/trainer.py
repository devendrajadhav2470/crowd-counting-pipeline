"""Training engine for crowd counting models.

Provides an OOP Trainer class that handles the full training loop,
including validation, checkpointing, learning rate scheduling,
and optional Weights & Biases logging.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from crowd_counting.config import Config
from crowd_counting.data.dataset import CrowdDataset, get_default_transform, get_train_val_split
from crowd_counting.engine.evaluator import Evaluator
from crowd_counting.models.csrnet import CSRNet
from crowd_counting.utils.checkpoint import load_checkpoint, save_checkpoint
from crowd_counting.utils.device import get_device

logger = logging.getLogger(__name__)


class AverageMeter:
    """Tracks the average and current value of a metric."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """Handles the full CSRNet training pipeline.

    Args:
        config: Configuration object with all hyperparameters.
        device: Optional torch device override.
    """

    def __init__(self, config: Config, device: Optional[torch.device] = None) -> None:
        self.config = config
        self.device = device or get_device()
        self.best_mae: float = 1e6
        self.start_epoch: int = 0

        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = CSRNet(load_weights=False)
        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = nn.MSELoss(reduction="sum").to(self.device)

        # Optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.training.lr,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )

        # Evaluator
        self.evaluator = Evaluator(self.model, self.device)

        # Weights & Biases
        self.wandb_run = None
        if config.wandb.enabled:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                config={
                    "model": self.config.model.name,
                    "dataset": self.config.dataset.name,
                    "epochs": self.config.training.epochs,
                    "lr": self.config.training.lr,
                    "batch_size": self.config.training.batch_size,
                    "momentum": self.config.training.momentum,
                    "weight_decay": self.config.training.weight_decay,
                    "sigma": self.config.dataset.sigma,
                },
            )
            logger.info(f"W&B initialized: {wandb.run.url}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.config.wandb.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.config.wandb.enabled = False

    def load_pretrained(self, checkpoint_path: str | Path) -> None:
        """Load a pretrained checkpoint to resume training.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.device
        )
        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_mae = checkpoint.get("best_prec1", 1e6)
        logger.info(
            f"Resumed from epoch {self.start_epoch}, best MAE: {self.best_mae:.3f}"
        )

    def train(self) -> None:
        """Run the full training loop."""
        cfg = self.config

        # Set random seed
        torch.manual_seed(cfg.training.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(cfg.training.seed)

        # Prepare data
        train_paths, val_paths = get_train_val_split(
            os.path.join(cfg.dataset.root, "train_data"),
            test_size=cfg.dataset.test_split,
            random_state=cfg.training.seed,
        )
        logger.info(f"Train: {len(train_paths)} images, Val: {len(val_paths)} images")

        for epoch in range(self.start_epoch, cfg.training.epochs):
            # Adjust learning rate
            lr = self._adjust_lr(epoch)

            # Train one epoch
            train_loss = self._train_epoch(train_paths, epoch)

            # Validate
            val_mae, val_rmse = self.evaluator.evaluate_paths(
                val_paths, batch_size=cfg.evaluation.batch_size
            )

            # Check if best
            is_best = val_mae < self.best_mae
            self.best_mae = min(val_mae, self.best_mae)

            logger.info(
                f"Epoch [{epoch + 1}/{cfg.training.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val MAE: {val_mae:.3f} | Val RMSE: {val_rmse:.3f} | "
                f"Best MAE: {self.best_mae:.3f} | LR: {lr:.2e}"
            )

            # Save checkpoint
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_prec1": self.best_mae,
                    "optimizer": self.optimizer.state_dict(),
                },
                is_best=is_best,
                checkpoint_dir=cfg.output.checkpoint_dir,
            )

            # Log to W&B
            if self.wandb_run:
                import wandb

                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/mae": val_mae,
                    "val/rmse": val_rmse,
                    "val/best_mae": self.best_mae,
                    "lr": lr,
                })

        if self.wandb_run:
            import wandb

            wandb.finish()

        logger.info(f"Training complete. Best MAE: {self.best_mae:.3f}")

    def _train_epoch(self, train_paths: list[str], epoch: int) -> float:
        """Run one training epoch.

        Args:
            train_paths: List of training image paths.
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        cfg = self.config.training

        train_dataset = CrowdDataset(
            train_paths, transform=get_default_transform(), train=True, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )

        self.model.train()
        losses = AverageMeter("Loss")
        batch_time = AverageMeter("Batch")

        end = time.time()
        for i, (img, target) in enumerate(train_loader):
            img = img.to(self.device)
            target = target.unsqueeze(0).to(self.device)

            output = self.model(img)
            loss = self.criterion(output, target)

            losses.update(loss.item(), img.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.print_freq == 0:
                logger.info(
                    f"Epoch [{epoch + 1}][{i}/{len(train_loader)}] "
                    f"Loss: {losses.val:.4f} (avg: {losses.avg:.4f}) "
                    f"Time: {batch_time.val:.3f}s"
                )

        return losses.avg

    def _adjust_lr(self, epoch: int) -> float:
        """Adjust learning rate based on step schedule.

        Args:
            epoch: Current epoch.

        Returns:
            Current learning rate.
        """
        cfg = self.config.training
        lr = cfg.lr

        for i, step in enumerate(cfg.lr_steps):
            scale = cfg.lr_scales[i] if i < len(cfg.lr_scales) else 1.0
            if epoch >= step:
                lr = lr * scale
                if epoch == step:
                    break
            else:
                break

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

