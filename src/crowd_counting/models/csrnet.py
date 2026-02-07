"""CSRNet: Congested Scene Recognition Network.

Architecture:
    - Frontend: First 10 layers of VGG-16 (up to conv4_3) as feature extractor
    - Backend: Dilated convolutional layers for density map estimation
    - Output: 1x1 convolution producing a single-channel density map

Reference:
    Li et al., "CSRNet: Dilated Convolutional Neural Networks for Understanding
    the Highly Congested Scenes", CVPR 2018.
    https://arxiv.org/abs/1712.03400
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights


class CSRNet(nn.Module):
    """CSRNet for crowd counting via density map estimation.

    The model uses a VGG-16 frontend (pretrained on ImageNet) followed by
    dilated convolution backend layers to produce a density map whose integral
    approximates the crowd count.

    Args:
        load_weights: If True, skip loading pretrained VGG-16 weights.
            Set to True when loading from a saved checkpoint.
    """

    # VGG-16 frontend config (first 10 conv layers, 'M' = MaxPool)
    FRONTEND_FEATURES: list[Union[int, str]] = [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512,
    ]
    # Dilated convolution backend config
    BACKEND_FEATURES: list[int] = [512, 512, 512, 256, 128, 64]

    def __init__(self, load_weights: bool = False) -> None:
        super().__init__()
        self.seen: int = 0

        self.frontend = _make_layers(self.FRONTEND_FEATURES)
        self.backend = _make_layers(
            self.BACKEND_FEATURES, in_channels=512, dilation=True
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            self._initialize_weights()
            self._load_vgg16_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image -> density map.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Density map of shape (B, 1, H/8, W/8).
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self) -> None:
        """Initialize backend and output layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_vgg16_frontend(self) -> None:
        """Load pretrained VGG-16 weights into the frontend layers."""
        vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        frontend_state = list(self.frontend.state_dict().items())
        vgg16_state = list(vgg16.state_dict().items())

        for i in range(len(frontend_state)):
            frontend_state[i][1].data[:] = vgg16_state[i][1].data[:]


def _make_layers(
    cfg: list[Union[int, str]],
    in_channels: int = 3,
    batch_norm: bool = False,
    dilation: bool = False,
) -> nn.Sequential:
    """Build a sequential block of Conv2d + ReLU (+ optional BatchNorm) layers.

    Args:
        cfg: Layer configuration. Integers specify output channels for Conv2d;
            'M' inserts a MaxPool2d layer.
        in_channels: Number of input channels for the first Conv2d.
        batch_norm: Whether to add BatchNorm2d after each Conv2d.
        dilation: Whether to use dilated convolutions (rate=2) in the backend.

    Returns:
        nn.Sequential module.
    """
    d_rate = 2 if dilation else 1
    layers: list[nn.Module] = []

    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            assert isinstance(v, int)
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

