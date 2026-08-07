#!/usr/bin/env python3
"""Measure periodic grid-watershed basins around inverted-parent voids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402


HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent.parent / "v2_hop_id_file"
sys.path.insert(0, str(V2_DIR))

from hop_to_genetic_id import find_part_files  # noqa: E402
from verify_evolved_mask import read_positions_and_ids  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def parse_catalogue(value: str) -> tuple[str, Path]:
    try:
        label, path_text = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("catalogue must have the form label=path") from error
    path = Path(path_text).resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError("catalogue label and existing JSON file are required")
    return label, path


def read_targets(catalogues: list[tuple[str, Path]]) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    keys: set[str] = set()
    for sample, path in catalogues:
        document = json.loads(path.read_text())
        for entry in document["targets"]:
            mass_rank = entry.get("mass_rank") or entry["halo_id"]
            tier = entry.get("comparison_tier") or sample
            key = f"{tier}_rank_{mass_rank}"
            if key in keys:
                raise ValueError(f"duplicate target key {key}")
            keys.add(key)
            targets.append(
                {
                    **entry,
                    "sample": sample,
                    "tier": tier,
                    "mass_rank": mass_rank,
                    "target_key": key,
                    "source_catalogue": str(path),
                }
            )
    if not targets:
        raise ValueError("no target centres were found")
    return targets


def build_density(
    snapshot: Path, grid_size: int, analysis_grid: int
) -> tuple[np.ndarray, int]:
    cell_count = analysis_grid**3
    counts = np.zeros(cell_count, dtype=np.int64)
    particle_total = 0
    files = find_part_files(snapshot)
    for path in files:
        ncpu, _, positions, ids = read_positions_and_ids(path)
        if ncpu != len(files):
            raise ValueError(f"{path}: declares {ncpu} ranks, found {len(files)}")
        cells = np.floor(positions * analysis_grid).astype(np.int64)
        cells %= analysis_grid
        flat = (
            cells[:, 0] * analysis_grid * analysis_grid
            + cells[:, 1] * analysis_grid
            + cells[:, 2]
        )
        counts += np.bincount(flat, minlength=cell_count)
        particle_total += ids.size
    expected = grid_size**3
    if particle_total != expected:
        raise ValueError(f"snapshot contains {particle_total} particles, expected {expected}")
    density = counts.reshape((analysis_grid,) * 3).astype(np.float32)
    density /= float(particle_total / cell_count)
    return density, particle_total


def steepest_descent_zones(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return six-neighbour periodic catchment labels and their root cells."""
    if field.ndim != 3 or len(set(field.shape)) != 1:
        raise ValueError("watershed field must be cubic")
    if field.size >= np.iinfo(np.int32).max:
        raise ValueError("watershed grid exceeds the int32 index range")
    values = np.asarray(field, dtype=np.float32)
    indices = np.arange(values.size, dtype=np.int32).reshape(values.shape)
    parents = indices.copy().ravel()
    best_values = values.copy().ravel()
    for axis in range(3):
        for shift in (-1, 1):
            candidate_values = np.roll(values, shift=shift, axis=axis).ravel()
            candidate_indices = np.roll(indices, shift=shift, axis=axis).ravel()
            better = candidate_values < best_values
            tied = (candidate_values == best_values) & (candidate_indices < parents)
            update = better | tied
            parents[update] = candidate_indices[update]
            best_values[update] = candidate_values[update]

    while True:
        next_parents = parents[parents]
        if np.array_equal(next_parents, parents):
            break
        parents = next_parents
    roots, labels = np.unique(parents, return_inverse=True)
    if not np.all(parents[roots] == roots):
        raise AssertionError("steepest-descent graph did not terminate at roots")
    return labels.astype(np.int32).reshape(values.shape), roots.astype(np.int32)


def reduce_saddles(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if keys.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    order = np.argsort(keys)
    ordered_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_keys)) + 1]
    return ordered_keys[starts], np.minimum.reduceat(values[order], starts)


def zone_adjacency(
    field: np.ndarray, zones: np.ndarray, zone_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_chunks: list[np.ndarray] = []
    saddle_chunks: list[np.ndarray] = []
    for axis in range(3):
        other_zones = np.roll(zones, shift=-1, axis=axis)
        boundary = zones != other_zones
        left = zones[boundary].astype(np.int64)
        right = other_zones[boundary].astype(np.int64)
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        keys = low * zone_count + high
        other_field = np.roll(field, shift=-1, axis=axis)
        saddles = np.maximum(field[boundary], other_field[boundary])
        unique_keys, minimum_saddles = reduce_saddles(keys, saddles)
        key_chunks.append(unique_keys)
        saddle_chunks.append(minimum_saddles)
    keys, saddles = reduce_saddles(
        np.concatenate(key_chunks), np.concatenate(saddle_chunks)
    )
    return keys // zone_count, keys % zone_count, saddles


def merge_zones(
    zone_count: int,
    left: np.ndarray,
    right: np.ndarray,
    saddle_delta: np.ndarray,
    ridge_density_ratio: float,
) -> tuple[np.ndarray, int]:
    selected = saddle_delta + 1.0 <= ridge_density_ratio
    graph = coo_matrix(
        (
            np.ones(2 * np.count_nonzero(selected), dtype=np.uint8),
            (
                np.r_[left[selected], right[selected]],
                np.r_[right[selected], left[selected]],
            ),
        ),
        shape=(zone_count, zone_count),
    )
    component_count, components = connected_components(
        graph.tocsr(), directed=False, return_labels=True
    )
    return components.astype(np.int32), int(component_count)


def effective_radius(cell_count: int, cell_volume: float) -> float:
    return float((3.0 * cell_count * cell_volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def summarise_targets(
    targets: list[dict[str, object]],
    field: np.ndarray,
    zones: np.ndarray,
    roots: np.ndarray,
    zone_components: np.ndarray,
    box_size: float,
) -> tuple[list[dict[str, object]], list[list[str]]]:
    grid = field.shape[0]
    cell_volume = (box_size / grid) ** 3
    zone_count = roots.size
    component_count = int(zone_components.max()) + 1
    flat_zones = zones.ravel()
    zone_cells = np.bincount(flat_zones, minlength=zone_count)
    zone_density_sum = np.bincount(
        flat_zones, weights=field.ravel(), minlength=zone_count
    )
    component_cells = np.bincount(
        zone_components, weights=zone_cells, minlength=component_count
    ).astype(np.int64)
    component_density_sum = np.bincount(
        zone_components, weights=zone_density_sum, minlength=component_count
    )
    zone_minima = field.ravel()[roots]
    component_minima = np.full(component_count, np.inf, dtype=np.float32)
    np.minimum.at(component_minima, zone_components, zone_minima)

    summaries: list[dict[str, object]] = []
    members: dict[int, list[str]] = {}
    for target in targets:
        centre = np.asarray(target["void_centre"], dtype=np.float64)
        index = tuple((np.floor(centre * grid).astype(np.int64) % grid).tolist())
        zone = int(zones[index])
        component = int(zone_components[zone])
        root_index = np.unravel_index(int(roots[zone]), field.shape)
        root_centre = tuple(float((value + 0.5) / grid) for value in root_index)
        members.setdefault(component, []).append(str(target["target_key"]))
        summaries.append(
            {
                **target,
                "watershed_zone": zone,
                "watershed_component": component,
                "zone_minimum_centre": root_centre,
                "zone_minimum_delta": float(zone_minima[zone]),
                "zone_cell_count": int(zone_cells[zone]),
                "zone_volume_mpc_h3": float(zone_cells[zone] * cell_volume),
                "zone_effective_radius_mpc_h": effective_radius(
                    int(zone_cells[zone]), cell_volume
                ),
                "merged_cell_count": int(component_cells[component]),
                "merged_volume_mpc_h3": float(component_cells[component] * cell_volume),
                "merged_effective_radius_mpc_h": effective_radius(
                    int(component_cells[component]), cell_volume
                ),
                "merged_minimum_delta": float(component_minima[component]),
                "merged_mean_delta": float(
                    component_density_sum[component] / component_cells[component]
                ),
            }
        )
    overlaps = [keys for keys in members.values() if len(keys) > 1]
    return summaries, overlaps


def plot_summary(targets: list[dict[str, object]], output: Path) -> None:
    labels = [f"R{target['mass_rank']}" for target in targets]
    colors = {
        "current": "#303030",
        "moderate": "#2b83ba",
        "compact": "#d95f02",
    }
    target_colors = [colors.get(str(target["tier"]), "#777777") for target in targets]
    x = np.arange(len(targets))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(
        x - 0.18,
        [target["zone_effective_radius_mpc_h"] for target in targets],
        0.36,
        color=target_colors,
        alpha=0.55,
        label="unmerged zone",
    )
    axes[0].bar(
        x + 0.18,
        [target["merged_effective_radius_mpc_h"] for target in targets],
        0.36,
        color=target_colors,
        label="merged watershed",
    )
    axes[0].set_ylabel(r"effective radius [$h^{-1}$ Mpc]")
    axes[0].legend(frameon=False)
    axes[1].scatter(
        [target["merged_effective_radius_mpc_h"] for target in targets],
        [target["merged_minimum_delta"] for target in targets],
        c=target_colors,
        s=48,
    )
    for target, label in zip(targets, labels, strict=True):
        axes[1].annotate(
            label,
            (target["merged_effective_radius_mpc_h"], target["merged_minimum_delta"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set(
        xlabel=r"merged effective radius [$h^{-1}$ Mpc]",
        ylabel=r"minimum density contrast",
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
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--catalogue", action="append", type=parse_catalogue, required=True)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--analysis-grid", type=int, default=256)
    parser.add_argument("--box-size", type=float, default=512.0)
    parser.add_argument("--smoothing", type=float, default=4.0)
    parser.add_argument("--ridge-density-ratio", type=float, default=0.2)
    args = parser.parse_args()

    if not 0.0 < args.ridge_density_ratio < 1.0:
        raise ValueError("ridge density ratio must lie between zero and one")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: output directory is not empty")
    targets = read_targets(args.catalogue)
    snapshot = args.snapshot.resolve()
    density, particle_total = build_density(snapshot, args.grid_size, args.analysis_grid)
    sigma_cells = args.smoothing / (args.box_size / args.analysis_grid)
    smoothed_delta = gaussian_filter(density, sigma=sigma_cells, mode="wrap") - 1.0
    zones, roots = steepest_descent_zones(smoothed_delta)
    left, right, saddle_delta = zone_adjacency(smoothed_delta, zones, roots.size)
    zone_components, component_count = merge_zones(
        roots.size,
        left,
        right,
        saddle_delta,
        args.ridge_density_ratio,
    )
    summaries, overlaps = summarise_targets(
        targets,
        smoothed_delta,
        zones,
        roots,
        zone_components,
        args.box_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    component_labels = zone_components[zones]
    np.save(output_dir / "watershed_components.npy", component_labels)
    parameters = {
        "grid_size": args.grid_size,
        "analysis_grid": args.analysis_grid,
        "box_size_mpc_h": args.box_size,
        "smoothing_mpc_h": args.smoothing,
        "ridge_density_ratio": args.ridge_density_ratio,
        "ridge_delta_threshold": args.ridge_density_ratio - 1.0,
        "particle_total": particle_total,
        "zone_count": int(roots.size),
        "component_count": component_count,
        "connectivity": "six-neighbour periodic steepest descent",
        "status": "preliminary grid watershed, not a particle-Voronoi ZOBOV catalogue",
        "catalogues": [
            {"label": label, "path": str(path), "sha256": sha256(path)}
            for label, path in args.catalogue
        ],
    }
    document = {"parameters": parameters, "overlapping_targets": overlaps, "targets": summaries}
    (output_dir / "watershed_properties.json").write_text(
        json.dumps(document, indent=2) + "\n"
    )
    plot_summary(summaries, output_dir / "watershed_target_summary.png")
    print(
        f"measured {roots.size} zones and {component_count} merged components "
        f"on the {args.analysis_grid}^3 grid"
    )
    for target in summaries:
        print(
            f"{target['target_key']}: R_eff={target['merged_effective_radius_mpc_h']:.3f} "
            f"h^-1 Mpc, component={target['watershed_component']}"
        )
    if overlaps:
        print(f"target overlaps: {overlaps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
