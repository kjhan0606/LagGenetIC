#!/usr/bin/env python3
"""Measure preliminary void centers and spherical profiles in the inverted parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402


HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent.parent / "v2_hop_id_file"
sys.path.insert(0, str(V2_DIR))

from hop_to_genetic_id import find_part_files  # noqa: E402
from verify_evolved_mask import read_positions_and_ids  # noqa: E402


def periodic_mean(values: np.ndarray) -> tuple[float, float]:
    phase = np.exp(2.0j * np.pi * np.asarray(values, dtype=np.float64))
    mean_phase = phase.mean()
    centre = float((np.angle(mean_phase) / (2.0 * np.pi)) % 1.0)
    return centre, float(abs(mean_phase))


def find_local_minimum(
    density: np.ndarray,
    guess: tuple[float, float, float],
    search_radius: float,
    box_size: float,
) -> tuple[tuple[float, float, float], float]:
    grid_size = density.shape[0]
    if density.shape != (grid_size, grid_size, grid_size):
        raise ValueError("density field must be cubic")
    cell_size = box_size / grid_size
    radius_cells = int(np.ceil(search_radius / cell_size))
    guess_index = np.floor(np.asarray(guess) * grid_size).astype(np.int64)
    offsets = np.arange(-radius_cells, radius_cells + 1, dtype=np.int64)
    dx, dy, dz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    distance2 = (dx * dx + dy * dy + dz * dz) * cell_size**2
    inside = distance2 <= search_radius**2
    ix = (guess_index[0] + dx[inside]) % grid_size
    iy = (guess_index[1] + dy[inside]) % grid_size
    iz = (guess_index[2] + dz[inside]) % grid_size
    values = density[ix, iy, iz]
    minimum = int(np.argmin(values))
    centre = tuple(
        float((index[minimum] + 0.5) / grid_size) for index in (ix, iy, iz)
    )
    return centre, float(values[minimum])


def periodic_radii(
    positions: np.ndarray, centre: tuple[float, float, float], box_size: float
) -> np.ndarray:
    delta = np.abs(positions - np.asarray(centre))
    delta = np.minimum(delta, 1.0 - delta)
    return np.sqrt(np.einsum("ij,ij->i", delta, delta)) * box_size


def outermost_threshold_radius(
    radii: np.ndarray, enclosed_delta: np.ndarray, threshold: float
) -> float | None:
    indices = np.flatnonzero(enclosed_delta <= threshold)
    return float(radii[indices[-1]]) if indices.size else None


def first_compensation_radius(
    radii: np.ndarray, enclosed_delta: np.ndarray
) -> float | None:
    minimum = int(np.argmin(enclosed_delta))
    indices = np.flatnonzero(enclosed_delta[minimum:] >= 0.0)
    return float(radii[minimum + indices[0]]) if indices.size else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--analysis-grid", type=int, default=256)
    parser.add_argument("--box-size", type=float, default=512.0)
    parser.add_argument("--smoothing", type=float, default=4.0)
    parser.add_argument("--search-radius", type=float, default=64.0)
    parser.add_argument("--profile-radius", type=float, default=128.0)
    parser.add_argument("--radial-bins", type=int, default=64)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    target_root = args.targets.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    catalogue = json.loads(
        (target_root / "parent_target_candidates.json").read_text()
    )
    targets = sorted(
        (entry for entry in catalogue["candidates"] if entry["selected"]),
        key=lambda entry: entry["target_rank"],
    )
    if not targets:
        raise AssertionError("the target catalogue contains no selected targets")

    expected_particles = args.grid_size**3
    target_lookup = np.full(expected_particles, -1, dtype=np.int16)
    target_ids: list[np.ndarray] = []
    for index, target in enumerate(targets):
        ids = np.loadtxt(target_root / target["id_file"], dtype=np.int64, ndmin=1)
        if np.any(ids < 0) or np.any(ids >= expected_particles):
            raise ValueError(f"target {index + 1} contains an invalid particle ID")
        if np.any(target_lookup[ids] >= 0):
            raise ValueError("selected HOP targets overlap in particle ID")
        target_lookup[ids] = index
        target_ids.append(ids)

    analysis_cells = args.analysis_grid**3
    density_counts = np.zeros(analysis_cells, dtype=np.int64)
    tracked_chunks: list[list[np.ndarray]] = [[] for _ in targets]
    particle_total = 0
    files = find_part_files(snapshot)
    for path in files:
        ncpu, _, positions, ids = read_positions_and_ids(path)
        if ncpu != len(files):
            raise ValueError(f"{path}: declares {ncpu} ranks, found {len(files)}")
        labels = target_lookup[ids]
        for index in range(len(targets)):
            if np.any(labels == index):
                tracked_chunks[index].append(positions[labels == index])
        cells = np.floor(positions * args.analysis_grid).astype(np.int64)
        cells %= args.analysis_grid
        flat = (
            cells[:, 0] * args.analysis_grid * args.analysis_grid
            + cells[:, 1] * args.analysis_grid
            + cells[:, 2]
        )
        density_counts += np.bincount(flat, minlength=analysis_cells)
        particle_total += ids.size
    if particle_total != expected_particles:
        raise ValueError(f"snapshot contains {particle_total}, expected {expected_particles}")

    tracked = [np.concatenate(chunks) for chunks in tracked_chunks]
    for index, (positions, ids) in enumerate(zip(tracked, target_ids, strict=True)):
        if positions.shape != (ids.size, 3):
            raise AssertionError(
                f"target {index + 1}: tracked {positions.shape[0]} of {ids.size} IDs"
            )

    density = density_counts.reshape(
        (args.analysis_grid, args.analysis_grid, args.analysis_grid)
    ).astype(np.float32)
    density /= float(particle_total / analysis_cells)
    density -= 1.0
    sigma_cells = args.smoothing / (args.box_size / args.analysis_grid)
    smoothed = gaussian_filter(density, sigma=sigma_cells, mode="wrap")

    results: list[dict[str, object]] = []
    centers: list[tuple[float, float, float]] = []
    for target, positions in zip(targets, tracked, strict=True):
        means = [periodic_mean(positions[:, axis]) for axis in range(3)]
        tracked_centre = tuple(item[0] for item in means)
        centre, minimum_delta = find_local_minimum(
            smoothed,
            tracked_centre,
            args.search_radius,
            args.box_size,
        )
        centers.append(centre)
        results.append(
            {
                "target_rank": target["target_rank"],
                "mass_rank": target.get("mass_rank"),
                "halo_id": target["halo_id"],
                "particle_count": target["particle_count"],
                "comparison_tier": target.get("comparison_tier"),
                "maximum_lagrangian_width_mpc_h": (
                    max(target["lagrangian_width"]) * args.box_size
                    if target.get("lagrangian_width") is not None
                    else None
                ),
                "level14_selected_particles": target.get(
                    "level14_selected_particles"
                ),
                "tracked_circular_centre": tracked_centre,
                "tracked_circular_concentration": tuple(item[1] for item in means),
                "void_centre": centre,
                "smoothed_minimum_delta": minimum_delta,
            }
        )

    edges = np.linspace(0.0, args.profile_radius, args.radial_bins + 1)
    histograms = np.zeros((len(targets), args.radial_bins), dtype=np.int64)
    for path in files:
        _, _, positions, _ = read_positions_and_ids(path)
        for index, centre in enumerate(centers):
            histograms[index] += np.histogram(
                periodic_radii(positions, centre, args.box_size), bins=edges
            )[0]

    shell_volume = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    enclosed_volume = 4.0 * np.pi / 3.0 * edges[1:] ** 3
    number_density = particle_total / args.box_size**3
    radii = edges[1:]
    profiles: dict[str, np.ndarray] = {"radius": radii}
    for index, result in enumerate(results):
        shell_delta = histograms[index] / (number_density * shell_volume) - 1.0
        enclosed_delta = np.cumsum(histograms[index]) / (
            number_density * enclosed_volume
        ) - 1.0
        r_delta80 = outermost_threshold_radius(radii, enclosed_delta, -0.8)
        result["r_enclosed_delta_minus_0p8_mpc_h"] = r_delta80
        result["compensation_radius_mpc_h"] = first_compensation_radius(
            radii, enclosed_delta
        )
        profiles[f"target_{index + 1:02d}_shell_delta"] = shell_delta
        profiles[f"target_{index + 1:02d}_enclosed_delta"] = enclosed_delta

    parameters = {
        "grid_size": args.grid_size,
        "analysis_grid": args.analysis_grid,
        "box_size_mpc_h": args.box_size,
        "smoothing_mpc_h": args.smoothing,
        "search_radius_mpc_h": args.search_radius,
        "profile_radius_mpc_h": args.profile_radius,
        "radial_bins": args.radial_bins,
        "radius_definition": "outermost spherical bin with enclosed delta <= -0.8",
        "status": "preliminary pre-watershed diagnostic",
    }
    (output_dir / "void_properties.json").write_text(
        json.dumps({"parameters": parameters, "targets": results}, indent=2) + "\n"
    )
    np.savez(output_dir / "void_profiles.npz", **profiles)

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(results)))
    for index, result in enumerate(results):
        axis.plot(
            radii,
            profiles[f"target_{index + 1:02d}_enclosed_delta"],
            color=colors[index],
            label=(
                f"{result['comparison_tier']} rank {result['mass_rank']}"
                if result["comparison_tier"] is not None
                else f"target {index + 1} (HOP {result['halo_id']})"
            ),
        )
        radius = result["r_enclosed_delta_minus_0p8_mpc_h"]
        if radius is not None:
            axis.axvline(radius, color=colors[index], alpha=0.35, linewidth=1.0)
    axis.axhline(-0.8, color="black", linestyle="--", linewidth=1.0)
    axis.axhline(0.0, color="0.5", linestyle=":", linewidth=1.0)
    axis.set(xlabel=r"$r\ [h^{-1}{\rm Mpc}]$", ylabel=r"$\bar{\delta}(<r)$")
    axis.set_ylim(-1.05, 1.0)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "void_enclosed_density_profiles.png", dpi=180)
    plt.close(figure)

    for result in results:
        print(
            f"target {result['target_rank']} HOP {result['halo_id']}: "
            f"delta_min={result['smoothed_minimum_delta']:.4f}, "
            f"R_delta80={result['r_enclosed_delta_minus_0p8_mpc_h']} h^-1 Mpc"
        )
    print(f"wrote preliminary void measurements to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
