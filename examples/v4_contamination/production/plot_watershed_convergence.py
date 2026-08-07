#!/usr/bin/env python3
"""Plot grid-resolution convergence of target watershed measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_result(value: str) -> Path:
    path = Path(value).resolve()
    if path.is_dir():
        path = path / "watershed_properties.json"
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"watershed result is absent: {path}")
    return path


def summarise_documents(documents: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(documents, key=lambda item: item["parameters"]["analysis_grid"])
    target_keys = [target["target_key"] for target in ordered[0]["targets"]]
    tiers = list(dict.fromkeys(str(target["tier"]) for target in ordered[0]["targets"]))
    results: list[dict[str, object]] = []
    for document in ordered:
        by_key = {target["target_key"]: target for target in document["targets"]}
        if list(by_key) != target_keys:
            raise ValueError("watershed results contain different target sets or ordering")
        tier_volumes = {
            tier: float(
                sum(
                    target["merged_volume_mpc_h3"]
                    for target in document["targets"]
                    if target["tier"] == tier
                )
            )
            for tier in tiers
        }
        reference_volume = tier_volumes[tiers[0]]
        results.append(
            {
                "analysis_grid": int(document["parameters"]["analysis_grid"]),
                "zone_count": int(document["parameters"]["zone_count"]),
                "component_count": int(document["parameters"]["component_count"]),
                "target_effective_radii_mpc_h": {
                    key: float(by_key[key]["merged_effective_radius_mpc_h"])
                    for key in target_keys
                },
                "tier_volumes_mpc_h3": tier_volumes,
                "tier_volume_fraction_of_current": {
                    tier: tier_volumes[tier] / reference_volume for tier in tiers
                },
            }
        )
    convergence: dict[str, object] = {}
    if len(results) >= 2:
        previous, final = results[-2:]
        convergence = {
            "grid_pair": [previous["analysis_grid"], final["analysis_grid"]],
            "target_radius_fractional_change": {
                key: (
                    final["target_effective_radii_mpc_h"][key]
                    / previous["target_effective_radii_mpc_h"][key]
                    - 1.0
                )
                for key in target_keys
            },
            "tier_volume_fractional_change": {
                tier: (
                    final["tier_volumes_mpc_h3"][tier]
                    / previous["tier_volumes_mpc_h3"][tier]
                    - 1.0
                )
                for tier in tiers
            },
        }
    return {
        "status": "preliminary periodic grid-watershed convergence diagnostic",
        "target_keys": target_keys,
        "tiers": tiers,
        "results": results,
        "finest_pair_convergence": convergence,
    }


def plot_summary(
    summary: dict[str, object], documents: list[dict[str, object]], output: Path
) -> None:
    ordered_documents = sorted(
        documents, key=lambda item: item["parameters"]["analysis_grid"]
    )
    grids = np.array([result["analysis_grid"] for result in summary["results"]])
    tier_colors = {"current": "#303030", "moderate": "#2b83ba", "compact": "#d95f02"}
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    first_by_key = {target["target_key"]: target for target in ordered_documents[0]["targets"]}

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    label_offsets = {
        "moderate_rank_130": -7,
        "moderate_rank_134": -5,
        "moderate_rank_140": 5,
        "compact_rank_726": 7,
        "compact_rank_727": -3,
        "compact_rank_729": 1,
    }
    for index, key in enumerate(summary["target_keys"]):
        target = first_by_key[key]
        radii = [
            result["target_effective_radii_mpc_h"][key] for result in summary["results"]
        ]
        color = tier_colors.get(str(target["tier"]), "#777777")
        axes[0].plot(
            grids,
            radii,
            color=color,
            marker=markers[index % len(markers)],
            linewidth=1.4,
            markersize=5,
            alpha=0.9,
        )
        axes[0].annotate(
            f"R{target['mass_rank']}",
            (grids[-1], radii[-1]),
            xytext=(4, label_offsets.get(key, 0)),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=color,
        )
    axes[0].set(
        xlabel="analysis grid per axis",
        ylabel=r"watershed effective radius [$h^{-1}$ Mpc]",
        xticks=grids,
    )
    axes[0].set_xlim(grids[0] - 6, grids[-1] + 10)
    for tier in summary["tiers"]:
        axes[1].plot(
            grids,
            [
                result["tier_volume_fraction_of_current"][tier]
                for result in summary["results"]
            ],
            color=tier_colors.get(tier, "#777777"),
            marker="o",
            linewidth=1.8,
            label=tier,
        )
    axes[1].set(
        xlabel="analysis grid per axis",
        ylabel="watershed volume / current volume",
        xticks=grids,
    )
    axes[1].legend(frameon=False)
    for axis, panel, panel_y in zip(axes, ("(a)", "(b)"), (0.97, 0.90), strict=True):
        axis.grid(alpha=0.2)
        axis.text(
            0.98,
            panel_y,
            panel,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("results", nargs="+", type=parse_result)
    args = parser.parse_args()
    if len(args.results) < 2:
        raise ValueError("at least two watershed resolutions are required")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: output directory is not empty")
    documents = [json.loads(path.read_text()) for path in args.results]
    summary = summarise_documents(documents)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "watershed_convergence.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    plot_summary(summary, documents, output_dir / "watershed_convergence.png")
    pair = summary["finest_pair_convergence"]["grid_pair"]
    print(f"compared watershed grids {pair[0]} and {pair[1]}")
    for tier, change in summary["finest_pair_convergence"][
        "tier_volume_fractional_change"
    ].items():
        print(f"{tier}: finest-pair volume change={100.0 * change:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
