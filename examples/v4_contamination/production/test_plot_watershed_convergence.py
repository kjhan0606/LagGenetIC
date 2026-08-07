#!/usr/bin/env python3
"""Tests for the grid-watershed convergence summary."""

from __future__ import annotations

import unittest

from plot_watershed_convergence import summarise_documents


def document(grid: int, current: float, compact: float) -> dict[str, object]:
    return {
        "parameters": {"analysis_grid": grid, "zone_count": 10, "component_count": 8},
        "targets": [
            {
                "target_key": "current_rank_1",
                "tier": "current",
                "merged_effective_radius_mpc_h": current,
                "merged_volume_mpc_h3": current**3,
            },
            {
                "target_key": "compact_rank_2",
                "tier": "compact",
                "merged_effective_radius_mpc_h": compact,
                "merged_volume_mpc_h3": compact**3,
            },
        ],
    }


class WatershedConvergenceTests(unittest.TestCase):
    def test_sorts_grids_and_measures_finest_pair(self) -> None:
        summary = summarise_documents(
            [document(256, 20.0, 10.0), document(128, 18.0, 9.0)]
        )
        self.assertEqual(
            [result["analysis_grid"] for result in summary["results"]], [128, 256]
        )
        changes = summary["finest_pair_convergence"][
            "target_radius_fractional_change"
        ]
        self.assertAlmostEqual(changes["current_rank_1"], 20.0 / 18.0 - 1.0)
        fractions = summary["results"][-1]["tier_volume_fraction_of_current"]
        self.assertAlmostEqual(fractions["compact"], 0.125)


if __name__ == "__main__":
    unittest.main()
