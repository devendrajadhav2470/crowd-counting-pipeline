"""Unit tests for density map generation."""

from __future__ import annotations

import numpy as np
import pytest

from crowd_counting.data.density_map import generate_density_map


class TestGenerateDensityMap:
    """Tests for the Gaussian density map generator."""

    def test_empty_points(self) -> None:
        """No annotation points should produce a zero density map."""
        dm = generate_density_map(
            image_shape=(100, 100),
            points=np.array([]).reshape(0, 2),
            sigma=15,
        )
        assert dm.shape == (100, 100)
        assert np.sum(dm) == 0.0

    def test_single_point(self) -> None:
        """Single point should produce a density map that sums to ~1."""
        points = np.array([[50, 50]])  # (x, y)
        dm = generate_density_map(
            image_shape=(100, 100),
            points=points,
            sigma=15,
        )
        assert dm.shape == (100, 100)
        assert abs(np.sum(dm) - 1.0) < 0.1, f"Expected sum ~1.0, got {np.sum(dm)}"

    def test_multiple_points_sum(self) -> None:
        """Density map sum should equal the number of annotation points."""
        n_points = 10
        rng = np.random.RandomState(42)
        points = rng.randint(10, 90, size=(n_points, 2)).astype(float)

        dm = generate_density_map(
            image_shape=(100, 100),
            points=points,
            sigma=15,
        )
        assert abs(np.sum(dm) - n_points) < 0.5, (
            f"Expected sum ~{n_points}, got {np.sum(dm)}"
        )

    def test_output_dtype(self) -> None:
        """Density map should be float32."""
        points = np.array([[50, 50]])
        dm = generate_density_map(image_shape=(100, 100), points=points, sigma=15)
        assert dm.dtype == np.float32

    def test_output_non_negative(self) -> None:
        """Density map values should be non-negative."""
        rng = np.random.RandomState(42)
        points = rng.randint(10, 90, size=(20, 2)).astype(float)

        dm = generate_density_map(image_shape=(100, 100), points=points, sigma=15)
        assert np.all(dm >= 0), "Density map contains negative values"

    def test_point_near_edge(self) -> None:
        """Points near the image edge should not cause errors."""
        points = np.array([[0, 0], [99, 99], [0, 99], [99, 0]])
        dm = generate_density_map(image_shape=(100, 100), points=points, sigma=15)
        assert dm.shape == (100, 100)
        assert np.sum(dm) > 0

    def test_adaptive_sigma(self) -> None:
        """Sigma=4 should use adaptive sigma based on KNN distances."""
        rng = np.random.RandomState(42)
        points = rng.randint(10, 90, size=(20, 2)).astype(float)

        dm = generate_density_map(image_shape=(100, 100), points=points, sigma=4)
        assert dm.shape == (100, 100)
        assert abs(np.sum(dm) - 20) < 1.0, f"Expected sum ~20, got {np.sum(dm)}"

    def test_rectangular_image(self) -> None:
        """Should work with non-square image shapes."""
        points = np.array([[100, 50], [200, 100]])
        dm = generate_density_map(image_shape=(150, 300), points=points, sigma=15)
        assert dm.shape == (150, 300)

