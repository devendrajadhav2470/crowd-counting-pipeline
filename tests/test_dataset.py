"""Unit tests for dataset utilities."""

from __future__ import annotations

import pytest
from torchvision import transforms

from crowd_counting.data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_default_transform


class TestDatasetUtils:
    """Tests for dataset utility functions."""

    def test_imagenet_constants(self) -> None:
        """ImageNet mean and std should have 3 channels."""
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3

    def test_default_transform(self) -> None:
        """Default transform should be a Compose with ToTensor and Normalize."""
        transform = get_default_transform()
        assert isinstance(transform, transforms.Compose)
        assert len(transform.transforms) == 2

    def test_default_transform_applies(self) -> None:
        """Default transform should convert a PIL image to a normalized tensor."""
        from PIL import Image
        import numpy as np

        # Create a dummy RGB image
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        transform = get_default_transform()
        tensor = transform(img)

        assert tensor.shape == (3, 64, 64)
        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert tensor.min() > -5.0
        assert tensor.max() < 5.0

