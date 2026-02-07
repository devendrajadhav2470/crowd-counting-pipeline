"""Unit tests for CSRNet model architecture."""

from __future__ import annotations

import pytest
import torch

from crowd_counting.models.csrnet import CSRNet, _make_layers


class TestCSRNet:
    """Tests for the CSRNet model."""

    def test_forward_pass_shape(self) -> None:
        """CSRNet should output density map at 1/8 spatial resolution."""
        model = CSRNet(load_weights=True)  # Skip pretrained download
        model.eval()

        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)

        # Output should be 1/8 of input spatial size
        assert output.shape == (1, 1, 32, 32), f"Expected (1, 1, 32, 32), got {output.shape}"

    def test_forward_pass_batch(self) -> None:
        """CSRNet should handle batch sizes > 1."""
        model = CSRNet(load_weights=True)
        model.eval()

        x = torch.randn(4, 3, 128, 128)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1, 16, 16)

    def test_forward_pass_rectangular_input(self) -> None:
        """CSRNet should handle non-square inputs."""
        model = CSRNet(load_weights=True)
        model.eval()

        x = torch.randn(1, 3, 384, 512)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1, 48, 64)

    def test_output_has_gradients(self) -> None:
        """Output should support gradient computation for training."""
        model = CSRNet(load_weights=True)
        model.train()

        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        output = model(x)

        # Should be able to backprop through the model
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_single_channel_output(self) -> None:
        """Output should always be single-channel (density map)."""
        model = CSRNet(load_weights=True)
        model.eval()

        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(x)

        assert output.shape[1] == 1, "Density map should be single-channel"

    def test_model_parameter_count(self) -> None:
        """CSRNet should have a reasonable number of parameters."""
        model = CSRNet(load_weights=True)
        num_params = sum(p.numel() for p in model.parameters())

        # CSRNet has ~16M parameters (VGG16 frontend + dilated backend)
        assert 10_000_000 < num_params < 25_000_000, f"Unexpected param count: {num_params}"


class TestMakeLayers:
    """Tests for the layer builder utility."""

    def test_basic_layers(self) -> None:
        """Should build Conv + ReLU layers."""
        layers = _make_layers([64, 128], in_channels=3)
        x = torch.randn(1, 3, 32, 32)
        output = layers(x)
        assert output.shape == (1, 128, 32, 32)

    def test_maxpool(self) -> None:
        """MaxPool ('M') should halve spatial dimensions."""
        layers = _make_layers([64, "M"], in_channels=3)
        x = torch.randn(1, 3, 32, 32)
        output = layers(x)
        assert output.shape == (1, 64, 16, 16)

    def test_dilated_layers(self) -> None:
        """Dilated layers should preserve spatial dimensions."""
        layers = _make_layers([64, 32], in_channels=512, dilation=True)
        x = torch.randn(1, 512, 32, 32)
        output = layers(x)
        assert output.shape == (1, 32, 32, 32)

