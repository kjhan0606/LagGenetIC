#!/usr/bin/env python3
"""Synthetic tests for the offline web-classifier comparison."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_web_classifiers import (
    largest_symmetric_eigenvalue,
    periodic_dilate,
    tweb_lambda_max,
    vweb_lambda_max,
)


class WebClassifierTests(unittest.TestCase):
    def test_largest_symmetric_eigenvalue(self) -> None:
        shape = (3, 2, 2)
        components = [np.full(shape, value, dtype=np.float32) for value in (1, 2, 3, 0, 0, 0)]
        np.testing.assert_allclose(largest_symmetric_eigenvalue(components), 3.0)

    def test_vweb_is_invariant_to_bulk_velocity(self) -> None:
        grid = 32
        coordinate = (np.arange(grid) + 0.5) / grid
        vx = -np.sin(2.0 * np.pi * coordinate)[:, None, None]
        velocity = [
            np.broadcast_to(vx, (grid, grid, grid)).copy(),
            np.zeros((grid, grid, grid), dtype=np.float64),
            np.zeros((grid, grid, grid), dtype=np.float64),
        ]
        baseline, divergence = vweb_lambda_max(velocity, 1.0 / grid, 1.0)
        shifted = [item.copy() for item in velocity]
        shifted[0] += 1234.5
        shifted[1] -= 876.0
        shifted_result, shifted_divergence = vweb_lambda_max(
            shifted, 1.0 / grid, 1.0
        )
        np.testing.assert_allclose(shifted_result, baseline, atol=2.0e-3)
        np.testing.assert_allclose(shifted_divergence, divergence, atol=2.0e-3)

    def test_tweb_plane_wave(self) -> None:
        grid = 32
        coordinate = (np.arange(grid) + 0.5) / grid
        density = np.cos(2.0 * np.pi * coordinate)[:, None, None]
        density = np.broadcast_to(density, (grid, grid, grid)).astype(np.float32)
        result = tweb_lambda_max(density)
        expected = np.maximum(density, 0.0)
        np.testing.assert_allclose(result, expected, atol=2.0e-5)

    def test_periodic_dilation_wraps(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=np.bool_)
        mask[0, 0, 0] = True
        expanded = periodic_dilate(mask, 1)
        self.assertTrue(expanded[-1, 0, 0])
        self.assertTrue(expanded[0, -1, 0])
        self.assertTrue(expanded[0, 0, -1])
        self.assertEqual(int(expanded.sum()), 7)


if __name__ == "__main__":
    unittest.main()
