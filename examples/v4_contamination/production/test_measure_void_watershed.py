#!/usr/bin/env python3
"""Focused tests for the periodic grid-watershed analysis."""

from __future__ import annotations

import unittest

import numpy as np

from measure_void_watershed import (
    merge_zones,
    steepest_descent_zones,
    zone_adjacency,
)


class WatershedTests(unittest.TestCase):
    def test_periodic_single_basin(self) -> None:
        grid = 9
        coordinates = np.indices((grid,) * 3)
        distances = np.minimum(coordinates, grid - coordinates)
        field = np.sum(distances**2, axis=0).astype(np.float32)
        zones, roots = steepest_descent_zones(field)
        self.assertEqual(roots.tolist(), [0])
        self.assertTrue(np.all(zones == 0))

    def test_two_zones_merge_across_low_ridge(self) -> None:
        x = np.arange(12)
        distance_left = np.minimum(abs(x - 2), 12 - abs(x - 2))
        distance_right = np.minimum(abs(x - 8), 12 - abs(x - 8))
        profile = np.minimum(distance_left, distance_right).astype(np.float32)
        field = np.broadcast_to(profile[:, None, None], (12, 12, 12)).copy()
        zones, roots = steepest_descent_zones(field)
        left, right, saddles = zone_adjacency(field, zones, roots.size)
        self.assertEqual(roots.size, 2)
        components, count = merge_zones(roots.size, left, right, saddles, 4.1)
        self.assertEqual(count, 1)
        self.assertEqual(components[0], components[1])

    def test_high_ridge_keeps_zones_separate(self) -> None:
        zones = np.array([0, 1], dtype=np.int32)
        components, count = merge_zones(
            2,
            np.array([0]),
            np.array([1]),
            np.array([-0.5]),
            0.2,
        )
        self.assertEqual(count, 2)
        self.assertNotEqual(components[zones[0]], components[zones[1]])


if __name__ == "__main__":
    unittest.main()
