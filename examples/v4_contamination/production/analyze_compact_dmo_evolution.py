#!/usr/bin/env python3
"""Measure and plot the intermediate evolution of the compact DMO pilot."""

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

from hop_to_genetic_id import (  # noqa: E402
    FortranRecordReader,
    _scalar,
    find_part_files,
)


def array_from_record(payload: bytes, endian: str, count: int, kind: str) -> np.ndarray:
    for size in (8, 4):
        if len(payload) == size * count:
            return np.frombuffer(payload, dtype=np.dtype(f"{endian}{kind}{size}"))
    raise ValueError(
        f"record has {len(payload)} bytes for {count} {kind}-valued entries"
    )


def read_positions_and_masses(
    path: Path,
) -> tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    with FortranRecordReader(path) as records:
        endian = records.endian
        ncpu = _scalar(records.read(), endian, f"{path}: ncpu")
        ndim = _scalar(records.read(), endian, f"{path}: ndim")
        npart = _scalar(records.read(), endian, f"{path}: npart")
        records.skip()
        nstar_tot = _scalar(records.read(), endian, f"{path}: nstar_tot")
        records.skip()
        records.skip()
        records.skip()
        positions = tuple(
            array_from_record(records.read(), endian, npart, "f")
            for _ in range(ndim)
        )
        for _ in range(ndim):
            records.skip()
        masses = array_from_record(records.read(), endian, npart, "f")

    if ndim != 3:
        raise ValueError(f"{path}: expected ndim=3, found {ndim}")
    if nstar_tot != 0:
        raise ValueError(f"{path}: expected a DMO snapshot, found stars")
    return ncpu, positions, masses


def info_value(path: Path, key: str) -> float:
    for line in path.read_text().splitlines():
        if line.strip().startswith(key):
            return float(line.split("=", maxsplit=1)[1].replace("D", "E"))
    raise ValueError(f"{path}: missing {key}")


def find_local_minimum(
    density: np.ndarray,
    guess: np.ndarray,
    search_radius_mpc_h: float,
    box_size_mpc_h: float,
) -> tuple[np.ndarray, float]:
    grid = density.shape[0]
    cell_size = box_size_mpc_h / grid
    radius_cells = int(np.ceil(search_radius_mpc_h / cell_size))
    guess_index = np.floor(guess * grid).astype(np.int64)
    offsets = np.arange(-radius_cells, radius_cells + 1, dtype=np.int64)
    dx, dy, dz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    inside = (dx * dx + dy * dy + dz * dz) * cell_size**2 <= search_radius_mpc_h**2
    ix = (guess_index[0] + dx[inside]) % grid
    iy = (guess_index[1] + dy[inside]) % grid
    iz = (guess_index[2] + dz[inside]) % grid
    values = density[ix, iy, iz]
    minimum = int(np.argmin(values))
    centre = np.array(
        [(axis[minimum] + 0.5) / grid for axis in (ix, iy, iz)], dtype=np.float64
    )
    return centre, float(values[minimum])


def enclosed_profile(
    mass_grid: np.ndarray,
    centre: np.ndarray,
    total_mass: float,
    box_size_mpc_h: float,
    radius_max_mpc_h: float,
    radial_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = mass_grid.shape[0]
    edges = np.linspace(0.0, radius_max_mpc_h, radial_bins + 1)
    shell_mass = np.zeros(radial_bins, dtype=np.float64)
    coordinates = (np.arange(grid, dtype=np.float64) + 0.5) / grid
    dy = ((coordinates - centre[1] + 0.5) % 1.0 - 0.5) * box_size_mpc_h
    dz = ((coordinates - centre[2] + 0.5) % 1.0 - 0.5) * box_size_mpc_h
    dy2_dz2 = dy[:, None] ** 2 + dz[None, :] ** 2
    for ix, coordinate in enumerate(coordinates):
        dx = ((coordinate - centre[0] + 0.5) % 1.0 - 0.5) * box_size_mpc_h
        radius = np.sqrt(dx * dx + dy2_dz2)
        shell_mass += np.histogram(
            radius.ravel(), bins=edges, weights=mass_grid[ix].ravel()
        )[0]
    radii = edges[1:]
    enclosed_volume = 4.0 * np.pi / 3.0 * radii**3
    mean_density = total_mass / box_size_mpc_h**3
    enclosed_delta = np.cumsum(shell_mass) / (mean_density * enclosed_volume) - 1.0
    return radii, enclosed_delta


def deposit_snapshot(
    output: Path,
    analysis_grid: int,
    box_size_mpc_h: float,
    smoothing_mpc_h: float,
    search_radius_mpc_h: float,
    profile_radius_mpc_h: float,
    radial_bins: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, float]:
    files = find_part_files(output)
    ncell = analysis_grid**3
    mass_grid = np.zeros(ncell, dtype=np.float64)
    total_mass = 0.0
    particle_count = 0
    finest_mass = np.inf
    finest_count = 0
    finest_position_sum = np.zeros(3, dtype=np.float64)

    for rank, path in enumerate(files, start=1):
        ncpu, positions, masses = read_positions_and_masses(path)
        if ncpu != len(files):
            raise ValueError(f"{path}: declares {ncpu} ranks, found {len(files)}")
        local_minimum = float(np.min(masses))
        if local_minimum < finest_mass:
            finest_mass = local_minimum
            finest_count = 0
            finest_position_sum[:] = 0.0
        if local_minimum == finest_mass:
            selected = masses == finest_mass
            finest_count += int(np.count_nonzero(selected))
            for axis in range(3):
                finest_position_sum[axis] += float(
                    np.sum(positions[axis][selected], dtype=np.float64)
                )

        ix = np.floor(np.mod(positions[0], 1.0) * analysis_grid).astype(np.int64)
        iy = np.floor(np.mod(positions[1], 1.0) * analysis_grid).astype(np.int64)
        iz = np.floor(np.mod(positions[2], 1.0) * analysis_grid).astype(np.int64)
        flat = (ix * analysis_grid + iy) * analysis_grid + iz
        mass_grid += np.bincount(flat, weights=masses, minlength=ncell)
        total_mass += float(np.sum(masses, dtype=np.float64))
        particle_count += masses.size
        if rank % 8 == 0 or rank == len(files):
            print(
                f"{output.name}: density ranks {rank}/{len(files)} "
                f"particles={particle_count}",
                flush=True,
            )

    if finest_count == 0:
        raise AssertionError(f"{output}: no finest particles")
    patch_centre = np.mod(finest_position_sum / finest_count, 1.0)
    mass_grid = mass_grid.reshape((analysis_grid,) * 3)
    mean_cell_mass = total_mass / ncell
    density = mass_grid / mean_cell_mass - 1.0
    sigma_cells = smoothing_mpc_h / (box_size_mpc_h / analysis_grid)
    smoothed = gaussian_filter(density, sigma=sigma_cells, mode="wrap")
    void_centre, minimum_delta = find_local_minimum(
        smoothed,
        patch_centre,
        search_radius_mpc_h,
        box_size_mpc_h,
    )
    radius, enclosed_delta = enclosed_profile(
        mass_grid,
        void_centre,
        total_mass,
        box_size_mpc_h,
        profile_radius_mpc_h,
        radial_bins,
    )
    below = np.flatnonzero(enclosed_delta <= -0.8)
    r_delta80 = float(radius[below[-1]]) if below.size else None

    suffix = output.name.removeprefix("output_")
    info = output / f"info_{suffix}.txt"
    header = output / f"header_{suffix}.txt"
    aexp = info_value(info, "aexp")
    coarse_step = int(round(info_value(info, "nstep_coarse")))
    header_particles = None
    if header.is_file():
        lines = header.read_text().splitlines()
        for index, line in enumerate(lines[:-1]):
            if line.strip() == "Total number of particles":
                header_particles = int(lines[index + 1])
                break
    if header_particles is not None and header_particles != particle_count:
        raise AssertionError(
            f"{output}: read {particle_count} particles, header has {header_particles}"
        )

    summary: dict[str, object] = {
        "snapshot": output.name,
        "coarse_step": coarse_step,
        "scale_factor": aexp,
        "redshift": 1.0 / aexp - 1.0,
        "particles": particle_count,
        "total_code_mass": total_mass,
        "finest_particle_mass": finest_mass,
        "finest_particles": finest_count,
        "finest_patch_centre": patch_centre.tolist(),
        "void_centre": void_centre.tolist(),
        "smoothed_minimum_delta": minimum_delta,
        "r_enclosed_delta_minus_0p8_mpc_h": r_delta80,
    }
    return summary, mass_grid, radius, enclosed_delta


def projected_map(
    output: Path,
    centre: np.ndarray,
    total_mass: float,
    box_size_mpc_h: float,
    field_mpc_h: float,
    slab_mpc_h: float,
    pixels: int,
    display_sigma_pixels: float,
) -> np.ndarray:
    pixel_size = field_mpc_h / pixels
    padding_pixels = int(np.ceil(4.0 * display_sigma_pixels))
    padded_pixels = pixels + 2 * padding_pixels
    padded_field_mpc_h = padded_pixels * pixel_size
    image = np.zeros((padded_pixels, padded_pixels), dtype=np.float64)
    files = find_part_files(output)
    half_field = 0.5 * padded_field_mpc_h
    half_slab = 0.5 * slab_mpc_h
    for rank, path in enumerate(files, start=1):
        _, positions, masses = read_positions_and_masses(path)
        dz = ((positions[2] - centre[2] + 0.5) % 1.0 - 0.5) * box_size_mpc_h
        in_slab = np.abs(dz) < half_slab
        if np.any(in_slab):
            dx = (
                (positions[0][in_slab] - centre[0] + 0.5) % 1.0 - 0.5
            ) * box_size_mpc_h
            dy = (
                (positions[1][in_slab] - centre[1] + 0.5) % 1.0 - 0.5
            ) * box_size_mpc_h
            inside = (np.abs(dx) < half_field) & (np.abs(dy) < half_field)
            image += np.histogram2d(
                dx[inside],
                dy[inside],
                bins=padded_pixels,
                range=((-half_field, half_field), (-half_field, half_field)),
                weights=masses[in_slab][inside],
            )[0]
        if rank % 8 == 0 or rank == len(files):
            print(
                f"{output.name}: projection ranks {rank}/{len(files)}",
                flush=True,
            )
    expected_mass = (
        total_mass / box_size_mpc_h**3 * slab_mpc_h * pixel_size * pixel_size
    )
    surface_ratio = image / expected_mass
    surface_ratio = gaussian_filter(
        surface_ratio, sigma=display_sigma_pixels, mode="nearest"
    )
    if padding_pixels > 0:
        surface_ratio = surface_ratio[
            padding_pixels:-padding_pixels, padding_pixels:-padding_pixels
        ]
    return surface_ratio.T


def plot_evolution(
    summaries: list[dict[str, object]],
    maps: list[np.ndarray],
    field_mpc_h: float,
    asinh_scale: float,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        1, len(maps), figsize=(13.8, 4.55), sharex=True, sharey=True
    )
    labels = "abcdefghijklmnopqrstuvwxyz"
    extent = (-0.5 * field_mpc_h, 0.5 * field_mpc_h) * 2
    image = None
    for index, (axis, summary, density_map) in enumerate(zip(axes, summaries, maps)):
        display_map = np.arcsinh((density_map - 1.0) / asinh_scale)
        image = axis.imshow(
            display_map,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-3.5,
            vmax=3.5,
            interpolation="bilinear",
            rasterized=True,
        )
        axis.text(
            0.035,
            0.95,
            f"({labels[index]})",
            transform=axis.transAxes,
            color="white",
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        axis.text(
            0.965,
            0.95,
            rf"$z={float(summary['redshift']):.3f}$",
            transform=axis.transAxes,
            color="white",
            fontsize=11,
            ha="right",
            va="top",
        )
        axis.set_xlabel(r"$x-x_{\rm v}\ [h^{-1}{\rm Mpc}]$")
        axis.tick_params(direction="in", colors="white", which="both")
        for spine in axis.spines.values():
            spine.set_color("white")
    axes[0].set_ylabel(r"$y-y_{\rm v}\ [h^{-1}{\rm Mpc}]$")
    for axis in axes:
        axis.xaxis.label.set_color("black")
        axis.yaxis.label.set_color("black")
        axis.tick_params(labelcolor="black")
    if image is None:
        raise AssertionError("no map was supplied")
    figure.subplots_adjust(
        left=0.065, right=0.905, bottom=0.14, top=0.985, wspace=0.025
    )
    colorbar_axis = figure.add_axes((0.92, 0.18, 0.012, 0.70))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(
        rf"$\sinh^{{-1}}[(\Sigma_{{\rm DM}}/\overline{{\Sigma}}_{{\rm DM}}-1)/{asinh_scale:g}]$"
    )
    figure.savefig(output, dpi=300)
    plt.close(figure)


def plot_profiles(
    summaries: list[dict[str, object]],
    radius: np.ndarray,
    profiles: list[np.ndarray],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for summary, profile in zip(summaries, profiles):
        axis.plot(radius, profile, linewidth=1.8, label=rf"$z={float(summary['redshift']):.3f}$")
    axis.axhline(-0.8, color="black", linestyle="--", linewidth=1.0)
    axis.axhline(0.0, color="0.5", linestyle=":", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlim(radius[0], radius[-1])
    axis.set_ylim(-1.05, 1.0)
    axis.set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")
    axis.set_ylabel(r"$\overline{\delta}_{\rm DM}(<r)$")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=240)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--analysis-grid", type=int, default=256)
    parser.add_argument("--box-size", type=float, default=512.0)
    parser.add_argument("--smoothing", type=float, default=4.0)
    parser.add_argument("--search-radius", type=float, default=16.0)
    parser.add_argument("--profile-radius", type=float, default=128.0)
    parser.add_argument("--radial-bins", type=int, default=64)
    parser.add_argument("--field", type=float, default=64.0)
    parser.add_argument("--slab", type=float, default=16.0)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--display-smoothing-pixels", type=float, default=8.0)
    parser.add_argument("--asinh-scale", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    mass_grids: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    common_radius: np.ndarray | None = None
    for snapshot in args.snapshots:
        snapshot = snapshot.resolve()
        summary, mass_grid, radius, profile = deposit_snapshot(
            snapshot,
            args.analysis_grid,
            args.box_size,
            args.smoothing,
            args.search_radius,
            args.profile_radius,
            args.radial_bins,
        )
        summaries.append(summary)
        mass_grids.append(mass_grid)
        profiles.append(profile)
        common_radius = radius

    if common_radius is None:
        raise AssertionError("no snapshot was analyzed")
    reference_centre = np.asarray(summaries[-1]["void_centre"], dtype=np.float64)
    reliable_radius = max(
        args.smoothing, 2.0 * args.box_size / args.analysis_grid
    )
    profiles.clear()
    for snapshot, summary, mass_grid in zip(args.snapshots, summaries, mass_grids):
        mean_cell_mass = float(summary["total_code_mass"]) / args.analysis_grid**3
        density = mass_grid / mean_cell_mass - 1.0
        sigma_cells = args.smoothing / (args.box_size / args.analysis_grid)
        smoothed = gaussian_filter(density, sigma=sigma_cells, mode="wrap")
        reference_index = np.floor(reference_centre * args.analysis_grid).astype(
            np.int64
        )
        reference_delta = float(smoothed[tuple(reference_index)])
        radius, profile = enclosed_profile(
            mass_grid,
            reference_centre,
            float(summary["total_code_mass"]),
            args.box_size,
            args.profile_radius,
            args.radial_bins,
        )
        below = np.flatnonzero(
            (radius >= reliable_radius) & (profile <= -0.8)
        )
        summary["candidate_void_centre"] = summary.pop("void_centre")
        summary["reference_void_centre"] = reference_centre.tolist()
        summary["smoothed_delta_at_reference_centre"] = reference_delta
        summary["r_enclosed_delta_minus_0p8_mpc_h"] = (
            float(radius[below[-1]]) if below.size else None
        )
        profiles.append(profile)
        maps.append(
            projected_map(
                snapshot.resolve(),
                reference_centre,
                float(summary["total_code_mass"]),
                args.box_size,
                args.field,
                args.slab,
                args.pixels,
                args.display_smoothing_pixels,
            )
        )
    common_radius = radius
    del mass_grids
    arrays: dict[str, np.ndarray] = {"radius_mpc_h": common_radius}
    for index, (density_map, profile) in enumerate(zip(maps, profiles), start=1):
        arrays[f"map_{index:02d}"] = density_map
        arrays[f"enclosed_delta_{index:02d}"] = profile
    np.savez_compressed(output_dir / "compact726_dmo_evolution.npz", **arrays)
    parameters = {
        "analysis_grid": args.analysis_grid,
        "box_size_mpc_h": args.box_size,
        "smoothing_mpc_h": args.smoothing,
        "search_radius_mpc_h": args.search_radius,
        "profile_radius_mpc_h": args.profile_radius,
        "radial_bins": args.radial_bins,
        "map_field_mpc_h": args.field,
        "map_slab_mpc_h": args.slab,
        "map_pixels": args.pixels,
        "display_smoothing_pixels": args.display_smoothing_pixels,
        "display_asinh_scale": args.asinh_scale,
        "profile_reliable_radius_min_mpc_h": reliable_radius,
        "deposition": "mass-weighted NGP with Gaussian display smoothing",
    }
    (output_dir / "compact726_dmo_evolution.json").write_text(
        json.dumps({"parameters": parameters, "snapshots": summaries}, indent=2) + "\n"
    )
    plot_evolution(
        summaries,
        maps,
        args.field,
        args.asinh_scale,
        output_dir / "fig_compact726_dmo_evolution.png",
    )
    plot_profiles(
        summaries,
        common_radius,
        profiles,
        output_dir / "fig_compact726_dmo_profiles.png",
    )
    for summary in summaries:
        print(
            f"{summary['snapshot']}: z={float(summary['redshift']):.6f} "
            f"delta_ref={float(summary['smoothed_delta_at_reference_centre']):.4f} "
            f"R_-0.8={summary['r_enclosed_delta_minus_0p8_mpc_h']}",
            flush=True,
        )
    print(f"wrote compact DMO evolution analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
