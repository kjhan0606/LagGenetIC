#!/usr/bin/env python3
"""Focused tests for watershed halo assignment and tier counting."""

from __future__ import annotations

import unittest

import numpy as np

from count_watershed_halos import (
    aggregate_tiers,
    component_at_position,
    particle_mass_msun_h,
)


class WatershedHaloTests(unittest.TestCase):
    def test_periodic_component_assignment(self) -> None:
        labels = np.arange(64, dtype=np.int32).reshape((4, 4, 4))
        self.assertEqual(component_at_position(labels, (0.01, 0.01, 0.01)), 0)
        self.assertEqual(component_at_position(labels, (0.99, 0.99, 0.99)), 63)
        self.assertEqual(component_at_position(labels, (1.01, -0.01, 0.01)), 12)

    def test_particle_mass_scales_by_octants(self) -> None:
        parent = particle_mass_msun_h(0.3, 512.0, 512)
        fine = parent / 8**5
        self.assertAlmostEqual(parent / fine, 32768.0)

    def test_tier_aggregation(self) -> None:
        targets = [
            {
                "tier": "compact",
                "watershed_component": 1,
                "merged_volume_mpc_h3": 100.0,
                "n_halo_ge_20_particles": 3,
            },
            {
                "tier": "compact",
                "watershed_component": 2,
                "merged_volume_mpc_h3": 300.0,
                "n_halo_ge_20_particles": 5,
            },
        ]
        tiers = aggregate_tiers(targets, (20,))
        self.assertEqual(tiers[0]["n_halo_ge_20_particles"], 8)
        self.assertAlmostEqual(
            tiers[0]["number_density_ge_20_particles_mpc_h_minus3"], 0.02
        )


if __name__ == "__main__":
    unittest.main()
