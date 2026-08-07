#!/usr/bin/env python3
"""Plot the pre-watershed void-depth and level-14 cost comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("current_properties", type=Path)
    parser.add_argument("comparison_properties", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-level", type=int, default=9)
    parser.add_argument("--production-level", type=int, default=14)
    args = parser.parse_args()

    current = json.loads(args.current_properties.read_text())["targets"]
    comparison = json.loads(args.comparison_properties.read_text())["targets"]
    tiers = {
        "current top 3": current,
        "moderate": [target for target in comparison if target["comparison_tier"] == "moderate"],
        "compact": [target for target in comparison if target["comparison_tier"] == "compact"],
    }
    if any(len(targets) != 3 for targets in tiers.values()):
        raise ValueError("the cost-depth comparison requires three targets per tier")

    level_factor = 8 ** (args.production_level - args.base_level)
    colors = {
        "current top 3": "#333333",
        "moderate": "#0072B2",
        "compact": "#D55E00",
    }
    markers = {"current top 3": "o", "moderate": "s", "compact": "D"}
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)

    annotation_offsets = {
        "current top 3": ((-28, 8), (-28, -13), (-28, 6)),
        "compact": ((5, 5), (5, -14), (5, 5)),
    }
    for tier, targets in tiers.items():
        costs = np.asarray(
            [
                target.get("level14_selected_particles")
                or target["particle_count"] * level_factor
                for target in targets
            ],
            dtype=np.int64,
        )
        depths = np.asarray(
            [target["smoothed_minimum_delta"] for target in targets], dtype=float
        )
        radii = np.asarray(
            [target["r_enclosed_delta_minus_0p8_mpc_h"] for target in targets],
            dtype=float,
        )
        labels = [
            f"R{target.get('mass_rank') or target['target_rank']}"
            for target in targets
        ]
        axes[0].scatter(
            costs / 10**6,
            depths,
            s=55,
            marker=markers[tier],
            color=colors[tier],
            label=tier,
            zorder=3,
        )
        if tier == "moderate":
            axes[0].annotate(
                "R130/R134",
                (float(np.mean(costs[:2])) / 10**6, float(np.mean(depths[:2]))),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=8,
            )
            axes[0].annotate(
                labels[2],
                (costs[2] / 10**6, depths[2]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        else:
            for cost, depth, label, offset in zip(
                costs, depths, labels, annotation_offsets[tier], strict=True
            ):
                axes[0].annotate(
                    label,
                    (cost / 10**6, depth),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                )
        summary[tier] = {
            "mass_ranks": [target.get("mass_rank") for target in targets],
            "level14_selected_particles": costs.tolist(),
            "smoothed_minimum_delta": depths.tolist(),
            "r_enclosed_delta_minus_0p8_mpc_h": radii.tolist(),
            "total_level14_selected_particles": int(costs.sum()),
            "spherical_volume_proxy": float(np.sum(radii**3)),
        }

    axes[0].set_xscale("log")
    axes[0].set_xlim(100.0, 1000.0)
    axes[0].set_xlabel(r"projected selected level-14 particles [$10^6$]")
    axes[0].set_ylabel(r"smoothed minimum density contrast")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=8)

    names = list(tiers)
    current_cost = summary["current top 3"]["total_level14_selected_particles"]
    current_volume = summary["current top 3"]["spherical_volume_proxy"]
    relative_cost = [summary[name]["total_level14_selected_particles"] / current_cost for name in names]
    relative_volume = [summary[name]["spherical_volume_proxy"] / current_volume for name in names]
    x = np.arange(len(names))
    width = 0.34
    axes[1].bar(
        x - width / 2,
        relative_cost,
        width,
        color="#8172B2",
        label="selected-particle cost",
    )
    axes[1].bar(
        x + width / 2,
        relative_volume,
        width,
        color="#55A868",
        label=r"$\sum R_{-0.8}^3$ proxy",
    )
    for positions, values in (
        (x - width / 2, relative_cost),
        (x + width / 2, relative_volume),
    ):
        for position, value in zip(positions, values, strict=True):
            axes[1].text(
                position,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[1].set_xticks(x, ("current", "moderate", "compact"))
    axes[1].set_ylim(0.0, 1.16)
    axes[1].set_ylabel("fraction of current top-3 value")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    axes[0].text(
        0.98, 0.96, "(a)", transform=axes[0].transAxes, ha="right", va="top",
        fontsize=12, fontweight="bold"
    )
    axes[1].text(
        0.98, 0.04, "(b)", transform=axes[1].transAxes, ha="right", va="bottom",
        fontsize=12, fontweight="bold"
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["relative_to_current"] = {
        name: {"cost": cost, "spherical_volume_proxy": volume}
        for name, cost, volume in zip(names, relative_cost, relative_volume, strict=True)
    }
    (output_dir / "candidate_cost_depth_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    fig.savefig(output_dir / "fig_candidate_cost_depth.png", dpi=260)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
