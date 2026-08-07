#!/usr/bin/env python3
"""Count inverted-parent HOP haloes inside target watershed components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from select_parent_targets import Candidate, parse_catalogue  # noqa: E402


CRITICAL_DENSITY_MSUN_H2_MPC3 = 2.77536627e11


def parse_thresholds(value: str) -> tuple[int, ...]:
    thresholds = tuple(sorted({int(item) for item in value.split(",")}))
    if not thresholds or min(thresholds) <= 0:
        raise argparse.ArgumentTypeError("positive comma-separated thresholds are required")
    return thresholds


def particle_mass_msun_h(
    omega_m: float, box_size_mpc_h: float, grid_size: int
) -> float:
    return float(
        CRITICAL_DENSITY_MSUN_H2_MPC3
        * omega_m
        * (box_size_mpc_h / grid_size) ** 3
    )


def component_at_position(labels: np.ndarray, position: tuple[float, float, float]) -> int:
    grid = labels.shape[0]
    if labels.shape != (grid, grid, grid):
        raise ValueError("watershed component field must be cubic")
    cell = tuple((np.floor(np.asarray(position) * grid).astype(np.int64) % grid).tolist())
    return int(labels[cell])


def assign_halo_components(
    haloes: list[Candidate], labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    components = np.fromiter(
        (component_at_position(labels, halo.eulerian_centre) for halo in haloes),
        dtype=np.int32,
        count=len(haloes),
    )
    particles = np.fromiter(
        (halo.particle_count for halo in haloes), dtype=np.int64, count=len(haloes)
    )
    return components, particles


def count_targets(
    watershed: dict[str, object],
    halo_components: np.ndarray,
    halo_particles: np.ndarray,
    thresholds: tuple[int, ...],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for target in watershed["targets"]:
        component = int(target["watershed_component"])
        inside = halo_components == component
        local_particles = halo_particles[inside]
        counts = {
            f"n_halo_ge_{threshold}_particles": int(
                np.count_nonzero(local_particles >= threshold)
            )
            for threshold in thresholds
        }
        results.append(
            {
                **target,
                "halo_count_all_regrouped": int(local_particles.size),
                **counts,
                "maximum_halo_particle_count": (
                    int(local_particles.max()) if local_particles.size else 0
                ),
                "halo_particles_in_regrouped_catalogue": int(local_particles.sum()),
            }
        )
    return results


def aggregate_tiers(
    targets: list[dict[str, object]], thresholds: tuple[int, ...]
) -> list[dict[str, object]]:
    tiers: list[dict[str, object]] = []
    for tier in dict.fromkeys(str(target["tier"]) for target in targets):
        selected = [target for target in targets if target["tier"] == tier]
        components = {int(target["watershed_component"]) for target in selected}
        if len(components) != len(selected):
            raise ValueError(f"tier {tier} contains overlapping watershed components")
        volume = float(sum(target["merged_volume_mpc_h3"] for target in selected))
        entry: dict[str, object] = {
            "tier": tier,
            "target_count": len(selected),
            "watershed_volume_mpc_h3": volume,
        }
        for threshold in thresholds:
            key = f"n_halo_ge_{threshold}_particles"
            count = int(sum(target[key] for target in selected))
            entry[key] = count
            entry[f"number_density_ge_{threshold}_particles_mpc_h_minus3"] = (
                count / volume
            )
        tiers.append(entry)
    return tiers


def plot_counts(
    targets: list[dict[str, object]], thresholds: tuple[int, ...], output: Path
) -> None:
    labels = [f"R{target['mass_rank']}" for target in targets]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    x = np.arange(len(targets))
    width = 0.75 / len(thresholds)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for index, threshold in enumerate(thresholds):
        key = f"n_halo_ge_{threshold}_particles"
        offset = (index - 0.5 * (len(thresholds) - 1)) * width
        axes[0].bar(
            x + offset,
            [target[key] for target in targets],
            width,
            color=colors[index % len(colors)],
            label=rf"$N_{{\rm p}}\geq {threshold}$",
        )
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylabel("halo count in watershed")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    primary = thresholds[0]
    key = f"n_halo_ge_{primary}_particles"
    tier_colors = {"current": "#303030", "moderate": "#2b83ba", "compact": "#d95f02"}
    for target in targets:
        density = target[key] / target["merged_volume_mpc_h3"] * 1.0e5
        axes[1].scatter(
            target["merged_effective_radius_mpc_h"],
            density,
            color=tier_colors.get(str(target["tier"]), "#777777"),
            s=48,
        )
        axes[1].annotate(
            f"R{target['mass_rank']}",
            (target["merged_effective_radius_mpc_h"], density),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set(
        xlabel=r"watershed effective radius [$h^{-1}$ Mpc]",
        ylabel=rf"$N_{{\rm p}}\geq {primary}$ halo density [$10^{{-5}} h^3$ Mpc$^{{-3}}$]",
    )
    for axis, panel in zip(axes, ("(a)", "(b)"), strict=True):
        axis.grid(alpha=0.2)
        axis.text(
            0.98,
            0.97,
            panel,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
        )
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.28, 1.01),
        ncol=len(thresholds),
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hop_catalogue", type=Path)
    parser.add_argument("watershed_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--thresholds", type=parse_thresholds, default=(20, 50, 100, 300))
    parser.add_argument("--omega-m", type=float, default=0.3099)
    parser.add_argument("--box-size", type=float, default=512.0)
    parser.add_argument("--parent-grid", type=int, default=512)
    parser.add_argument("--production-level", type=int, default=14)
    args = parser.parse_args()

    hop_catalogue = args.hop_catalogue.resolve()
    watershed_dir = args.watershed_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: output directory is not empty")
    watershed = json.loads((watershed_dir / "watershed_properties.json").read_text())
    labels = np.load(watershed_dir / "watershed_components.npy", mmap_mode="r")
    expected_grid = int(watershed["parameters"]["analysis_grid"])
    if labels.shape != (expected_grid,) * 3:
        raise ValueError("watershed label shape does not match its JSON metadata")
    haloes = parse_catalogue(hop_catalogue)
    halo_components, halo_particles = assign_halo_components(haloes, labels)
    targets = count_targets(watershed, halo_components, halo_particles, args.thresholds)
    tiers = aggregate_tiers(targets, args.thresholds)

    parent_mass = particle_mass_msun_h(args.omega_m, args.box_size, args.parent_grid)
    parent_level = int(np.log2(args.parent_grid))
    if 2**parent_level != args.parent_grid or args.production_level < parent_level:
        raise ValueError("parent grid must be a power of two below the production level")
    refinement_factor = 8 ** (args.production_level - parent_level)
    production_mass = parent_mass / refinement_factor
    parameters = {
        "hop_catalogue": str(hop_catalogue),
        "watershed_dir": str(watershed_dir),
        "halo_count": len(haloes),
        "thresholds_particles": args.thresholds,
        "omega_m": args.omega_m,
        "box_size_mpc_h": args.box_size,
        "parent_grid": args.parent_grid,
        "parent_particle_mass_msun_h": parent_mass,
        "production_level": args.production_level,
        "projected_level14_dmo_particle_mass_msun_h": production_mass,
        "projected_level14_dmo_100_particle_halo_mass_msun_h": 100.0
        * production_mass,
        "assignment": "HOP halo centre inside the periodic grid-watershed component",
        "status": "parent-resolution halo-yield diagnostic",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "watershed_halo_counts.json").write_text(
        json.dumps({"parameters": parameters, "tiers": tiers, "targets": targets}, indent=2)
        + "\n"
    )
    with (output_dir / "watershed_halo_counts.tsv").open("w") as handle:
        count_columns = [f"n_halo_ge_{threshold}_particles" for threshold in args.thresholds]
        handle.write(
            "target_key\ttier\tmass_rank\tcomponent\tvolume_mpc_h3\t"
            + "\t".join(count_columns)
            + "\n"
        )
        for target in targets:
            handle.write(
                f"{target['target_key']}\t{target['tier']}\t{target['mass_rank']}\t"
                f"{target['watershed_component']}\t{target['merged_volume_mpc_h3']:.9e}\t"
                + "\t".join(str(target[column]) for column in count_columns)
                + "\n"
            )
    plot_counts(targets, args.thresholds, output_dir / "watershed_halo_yield.png")
    print(
        f"assigned {len(haloes)} inverted-parent HOP haloes; parent particle mass="
        f"{parent_mass:.6e} Msun/h"
    )
    print(
        f"projected level-{args.production_level} DMO particle mass="
        f"{production_mass:.6e} Msun/h"
    )
    for tier in tiers:
        counts = ", ".join(
            f"N>={threshold}: {tier[f'n_halo_ge_{threshold}_particles']}"
            for threshold in args.thresholds
        )
        print(f"{tier['tier']}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
