#!/usr/bin/env python3
"""Compare V-web and T-web wall indicators against a void watershed.

The classifier is intentionally offline.  It deposits the fixed-grid DMO
parent particles on the same grid used by the existing watershed analysis,
constructs smoothed velocity and density fields, and measures how well the
largest eigenvalue of each web tensor follows a selected watershed boundary.
No literature threshold is assumed.  Each tensor is calibrated independently
against the supplied watershed component.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import fft  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402


HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent.parent / "v2_hop_id_file"
sys.path.insert(0, str(V2_DIR))

from hop_to_genetic_id import FortranRecordReader, _scalar, find_part_files  # noqa: E402
from verify_evolved_mask import _array  # noqa: E402


def info_value(text: str, key: str) -> float:
    match = re.search(
        rf"^\s*{re.escape(key)}\s*=\s*([+\-0-9.EeDd]+)", text, re.MULTILINE
    )
    if match is None:
        raise ValueError(f"snapshot info does not contain {key}")
    return float(match.group(1).replace("D", "E").replace("d", "e"))


def read_snapshot_info(snapshot: Path) -> dict[str, float]:
    suffix = snapshot.name.removeprefix("output_")
    path = snapshot / f"info_{suffix}.txt"
    text = path.read_text(errors="replace")
    keys = ("ncpu", "levelmin", "levelmax", "aexp", "H0", "omega_m", "omega_l", "unit_l", "unit_t")
    values = {key: info_value(text, key) for key in keys}
    values["path"] = str(path.resolve())
    return values


def read_positions_velocities(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Read position and code-velocity records from one RAMSES DMO part file."""
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
        positions = [_array(records.read(), endian, npart, "f") for _ in range(ndim)]
        velocities = [_array(records.read(), endian, npart, "f") for _ in range(ndim)]

    if ndim != 3:
        raise ValueError(f"{path}: expected ndim=3, found {ndim}")
    if nstar_tot != 0:
        raise ValueError(f"{path}: expected a DMO snapshot, found stars")
    return ncpu, np.column_stack(positions), np.column_stack(velocities)


def build_ngp_fields(snapshot: Path, grid: int) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Deposit particle count and momentum on a periodic NGP grid."""
    size = grid**3
    count = np.zeros(size, dtype=np.int64)
    momentum = [np.zeros(size, dtype=np.float64) for _ in range(3)]
    files = find_part_files(snapshot)
    total = 0
    for path in files:
        ncpu, positions, velocities = read_positions_velocities(path)
        if ncpu != len(files):
            raise ValueError(f"{path}: declares {ncpu} ranks, found {len(files)}")
        cells = np.floor(positions * grid).astype(np.int64)
        cells %= grid
        flat = cells[:, 0] * grid * grid + cells[:, 1] * grid + cells[:, 2]
        count += np.bincount(flat, minlength=size)
        for axis in range(3):
            momentum[axis] += np.bincount(
                flat, weights=velocities[:, axis], minlength=size
            )
        total += positions.shape[0]
    shape = (grid, grid, grid)
    return count.reshape(shape), [item.reshape(shape) for item in momentum], total


def smooth_particle_fields(
    count: np.ndarray,
    momentum: list[np.ndarray],
    sigma_cells: float,
    velocity_unit_km_s: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return smoothed density contrast and mass-weighted peculiar velocity."""
    if sigma_cells <= 0.0:
        raise ValueError("smoothing scale must be positive")
    weight = gaussian_filter(count.astype(np.float32), sigma=sigma_cells, mode="wrap")
    mean_count = float(count.mean())
    delta = weight / mean_count - 1.0
    velocity: list[np.ndarray] = []
    valid = weight > max(1.0e-8 * mean_count, np.finfo(np.float32).tiny)
    for component in momentum:
        smoothed = gaussian_filter(component, sigma=sigma_cells, mode="wrap")
        local = np.zeros_like(smoothed, dtype=np.float32)
        local[valid] = (smoothed[valid] / weight[valid]) * velocity_unit_km_s
        velocity.append(local)
    return delta.astype(np.float32, copy=False), velocity


def central_difference(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (
        2.0 * spacing
    )


def largest_symmetric_eigenvalue(components: list[np.ndarray], chunk: int = 500_000) -> np.ndarray:
    """Return lambda_max for symmetric components xx, yy, zz, xy, xz, yz."""
    if len(components) != 6:
        raise ValueError("six tensor components are required")
    shape = components[0].shape
    if any(item.shape != shape for item in components):
        raise ValueError("tensor component shapes differ")
    flat = [np.asarray(item, dtype=np.float32).ravel() for item in components]
    result = np.empty(flat[0].size, dtype=np.float32)
    for start in range(0, result.size, chunk):
        stop = min(start + chunk, result.size)
        matrix = np.empty((stop - start, 3, 3), dtype=np.float32)
        matrix[:, 0, 0] = flat[0][start:stop]
        matrix[:, 1, 1] = flat[1][start:stop]
        matrix[:, 2, 2] = flat[2][start:stop]
        matrix[:, 0, 1] = matrix[:, 1, 0] = flat[3][start:stop]
        matrix[:, 0, 2] = matrix[:, 2, 0] = flat[4][start:stop]
        matrix[:, 1, 2] = matrix[:, 2, 1] = flat[5][start:stop]
        result[start:stop] = np.linalg.eigvalsh(matrix)[:, -1]
    return result.reshape(shape)


def vweb_lambda_max(
    velocity: list[np.ndarray], spacing_mpc_h: float, expansion_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute V-web lambda_max and div(v)/(a H / h)."""
    gradient = [
        [central_difference(velocity[i], j, spacing_mpc_h) for j in range(3)]
        for i in range(3)
    ]
    components = [
        -gradient[0][0] / expansion_rate,
        -gradient[1][1] / expansion_rate,
        -gradient[2][2] / expansion_rate,
        -0.5 * (gradient[0][1] + gradient[1][0]) / expansion_rate,
        -0.5 * (gradient[0][2] + gradient[2][0]) / expansion_rate,
        -0.5 * (gradient[1][2] + gradient[2][1]) / expansion_rate,
    ]
    divergence = (
        gradient[0][0] + gradient[1][1] + gradient[2][2]
    ) / expansion_rate
    return largest_symmetric_eigenvalue(components), divergence.astype(np.float32)


def tweb_lambda_max(delta: np.ndarray, workers: int = 1) -> np.ndarray:
    """Compute lambda_max of T_ij = partial_i partial_j Phi, nabla^2 Phi=delta."""
    grid = delta.shape[0]
    if delta.shape != (grid, grid, grid):
        raise ValueError("density field must be cubic")
    if workers < 1:
        raise ValueError("workers must be positive")
    delta_k = fft.rfftn(delta.astype(np.float32, copy=False), workers=workers)
    kx = fft.fftfreq(grid).astype(np.float32)[:, None, None]
    ky = fft.fftfreq(grid).astype(np.float32)[None, :, None]
    kz = fft.rfftfreq(grid).astype(np.float32)[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz
    inverse_k2 = np.zeros_like(k2)
    inverse_k2[k2 > 0.0] = 1.0 / k2[k2 > 0.0]
    wave = (kx, ky, kz)
    components: list[np.ndarray] = []
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        tensor_k = delta_k * (wave[i] * wave[j] * inverse_k2)
        components.append(
            fft.irfftn(tensor_k, s=delta.shape, workers=workers).astype(np.float32)
        )
    return largest_symmetric_eigenvalue(components)


def periodic_dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    result = np.asarray(mask, dtype=np.bool_).copy()
    for _ in range(steps):
        expanded = result.copy()
        for axis in range(3):
            expanded |= np.roll(result, 1, axis=axis)
            expanded |= np.roll(result, -1, axis=axis)
        result = expanded
    return result


def wall_and_scope(
    component_mask: np.ndarray, wall_cells: int, scope_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    if wall_cells < 1 or scope_cells < wall_cells:
        raise ValueError("require 1 <= wall_cells <= scope_cells")
    near_inside = periodic_dilate(~component_mask, wall_cells) & component_mask
    near_outside = periodic_dilate(component_mask, wall_cells) & ~component_mask
    wall = near_inside | near_outside
    scope = periodic_dilate(component_mask, scope_cells)
    return wall & scope, scope


def best_threshold(values: np.ndarray, reference: np.ndarray, scope: np.ndarray) -> dict[str, float | int]:
    local_values = np.asarray(values[scope], dtype=np.float64)
    local_reference = np.asarray(reference[scope], dtype=np.bool_)
    finite = np.isfinite(local_values)
    local_values = local_values[finite]
    local_reference = local_reference[finite]
    positives = int(local_reference.sum())
    if positives == 0 or positives == local_reference.size:
        raise ValueError("reference must contain positive and negative cells")
    order = np.argsort(local_values)[::-1]
    labels = local_reference[order]
    sorted_values = local_values[order]
    true_positive = np.cumsum(labels, dtype=np.int64)
    predicted = np.arange(1, labels.size + 1, dtype=np.int64)
    change = np.r_[sorted_values[1:] != sorted_values[:-1], True]
    candidates = np.flatnonzero(change)
    tp = true_positive[candidates].astype(np.float64)
    fp = predicted[candidates].astype(np.float64) - tp
    fn = positives - tp
    f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1.0)
    best = int(candidates[int(np.argmax(f1))])
    threshold = float(sorted_values[best])
    flagged = values >= threshold
    tp_count = int(np.count_nonzero(flagged & reference & scope))
    fp_count = int(np.count_nonzero(flagged & ~reference & scope))
    fn_count = int(np.count_nonzero(~flagged & reference & scope))
    precision = tp_count / max(tp_count + fp_count, 1)
    recall = tp_count / max(tp_count + fn_count, 1)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, np.finfo(float).tiny),
        "true_positive_cells": tp_count,
        "false_positive_cells": fp_count,
        "false_negative_cells": fn_count,
        "flagged_cells": int(np.count_nonzero(flagged & scope)),
        "scope_cells": int(np.count_nonzero(scope)),
        "reference_wall_cells": int(np.count_nonzero(reference & scope)),
    }


def centered_slice(field: np.ndarray, centre: np.ndarray, half_width: int) -> np.ndarray:
    grid = field.shape[0]
    centre_index = np.floor(centre * grid).astype(np.int64) % grid
    offsets = np.arange(-half_width, half_width + 1)
    ix = (centre_index[0] + offsets) % grid
    iy = (centre_index[1] + offsets) % grid
    iz = int(centre_index[2])
    return field[np.ix_(ix, iy, np.asarray([iz]))][:, :, 0]


def plot_comparison(
    delta: np.ndarray,
    v_lambda: np.ndarray,
    t_lambda: np.ndarray,
    component: np.ndarray,
    wall: np.ndarray,
    v_flag: np.ndarray,
    t_flag: np.ndarray,
    centre: np.ndarray,
    box_size: float,
    half_width: int,
    output: Path,
) -> None:
    fields = [
        centered_slice(item, centre, half_width)
        for item in (delta, v_lambda, t_lambda, component, wall, v_flag, t_flag)
    ]
    density, v_slice, t_slice, component_slice, wall_slice, vf_slice, tf_slice = fields
    width = (2 * half_width + 1) * box_size / delta.shape[0]
    extent = (-0.5 * width, 0.5 * width, -0.5 * width, 0.5 * width)
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.5), constrained_layout=True)
    panels = axes.ravel()
    density_limit = float(np.nanpercentile(np.abs(density), 98.0))
    image = panels[0].imshow(
        density.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-density_limit,
        vmax=density_limit,
    )
    panels[0].contour(component_slice.T, levels=[0.5], colors="black", linewidths=1.0, extent=extent)
    figure.colorbar(image, ax=panels[0], label=r"$\delta$")
    for axis, field, title in (
        (panels[1], v_slice, r"V-web $\lambda_{\max}$"),
        (panels[2], t_slice, r"T-web $\lambda_{\max}$"),
    ):
        low, high = np.nanpercentile(field, (2.0, 98.0))
        image = axis.imshow(field.T, origin="lower", extent=extent, cmap="viridis", vmin=low, vmax=high)
        axis.contour(component_slice.T, levels=[0.5], colors="white", linewidths=0.9, extent=extent)
        figure.colorbar(image, ax=axis)
        axis.set_title(title)
    panels[0].set_title("Density and watershed component")
    panels[3].imshow(density.T, origin="lower", extent=extent, cmap="Greys", vmin=-1.0, vmax=1.0)
    panels[3].contour(wall_slice.T, levels=[0.5], colors="#e41a1c", linewidths=1.4, extent=extent)
    panels[3].set_title("Watershed wall reference")
    for axis, flag_slice, title in (
        (panels[4], vf_slice, "Best V-web wall flag"),
        (panels[5], tf_slice, "Best T-web wall flag"),
    ):
        axis.imshow(density.T, origin="lower", extent=extent, cmap="Greys", vmin=-1.0, vmax=1.0)
        axis.contour(wall_slice.T, levels=[0.5], colors="#e41a1c", linewidths=1.2, extent=extent)
        axis.contour(flag_slice.T, levels=[0.5], colors="#377eb8", linewidths=1.0, extent=extent)
        axis.set_title(title)
    for index, axis in enumerate(panels):
        axis.text(
            0.02,
            0.97,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            color="black" if index in (0, 3, 4, 5) else "white",
            bbox={"facecolor": "white" if index in (0, 3, 4, 5) else "black", "alpha": 0.65, "edgecolor": "none"},
        )
        axis.set_xlabel(r"$\Delta x\ [h^{-1}\,{\rm Mpc}]$")
        axis.set_ylabel(r"$\Delta y\ [h^{-1}\,{\rm Mpc}]$")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("watershed_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-rank", type=int, default=726)
    parser.add_argument("--box-size", type=float, default=512.0)
    parser.add_argument("--smoothing", type=float, default=4.0)
    parser.add_argument("--wall-cells", type=int, default=1)
    parser.add_argument("--scope-cells", type=int, default=4)
    parser.add_argument("--slice-half-width", type=int, default=18)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    watershed_dir = args.watershed_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: output directory is not empty")

    labels = np.load(watershed_dir / "watershed_components.npy", mmap_mode="r")
    grid = labels.shape[0]
    if labels.shape != (grid, grid, grid):
        raise ValueError("watershed component grid must be cubic")
    document = json.loads((watershed_dir / "watershed_properties.json").read_text())
    matches = [
        item for item in document["targets"] if int(item["mass_rank"]) == args.target_rank
    ]
    if len(matches) != 1:
        raise ValueError(f"found {len(matches)} watershed targets for rank {args.target_rank}")
    target = matches[0]
    component_id = int(target["watershed_component"])
    component_mask = np.asarray(labels == component_id)
    wall, scope = wall_and_scope(component_mask, args.wall_cells, args.scope_cells)

    info = read_snapshot_info(snapshot)
    count, momentum, particle_total = build_ngp_fields(snapshot, grid)
    expected_particles = int(round(2.0 ** (3 * info["levelmin"])))
    if particle_total != expected_particles:
        raise ValueError(
            f"snapshot has {particle_total} particles, expected {expected_particles}"
        )
    spacing = args.box_size / grid
    sigma_cells = args.smoothing / spacing
    velocity_unit_km_s = info["unit_l"] / info["unit_t"] / 1.0e5
    delta, velocity = smooth_particle_fields(
        count, momentum, sigma_cells, velocity_unit_km_s
    )
    del count, momentum

    hubble_ratio = np.sqrt(
        info["omega_m"] / info["aexp"] ** 3 + info["omega_l"]
    )
    expansion_rate = 100.0 * info["aexp"] * hubble_ratio
    v_lambda, divergence = vweb_lambda_max(velocity, spacing, expansion_rate)
    del velocity
    t_lambda = tweb_lambda_max(delta, workers=args.workers)

    v_metrics = best_threshold(v_lambda, wall, scope)
    t_metrics = best_threshold(t_lambda, wall, scope)
    v_flag = v_lambda >= float(v_metrics["threshold"])
    t_flag = t_lambda >= float(t_metrics["threshold"])
    agreement = float(np.count_nonzero(v_flag & t_flag & scope) / max(np.count_nonzero((v_flag | t_flag) & scope), 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "offline classifier comparison; thresholds calibrated to one snapshot",
        "snapshot": str(snapshot),
        "snapshot_info": info,
        "watershed_dir": str(watershed_dir),
        "target_rank": args.target_rank,
        "watershed_component": component_id,
        "void_centre": target["void_centre"],
        "grid": grid,
        "cell_size_mpc_h": spacing,
        "smoothing_mpc_h": args.smoothing,
        "smoothing_cells": sigma_cells,
        "wall_cells": args.wall_cells,
        "scope_cells": args.scope_cells,
        "particle_total": particle_total,
        "fft_workers": args.workers,
        "velocity_unit_km_s": velocity_unit_km_s,
        "vweb_denominator_aH_over_h_km_s_per_mpc_h": expansion_rate,
        "vweb": v_metrics,
        "tweb": t_metrics,
        "flag_jaccard_with_each_other_in_scope": agreement,
        "normalization": {
            "vweb": "Sigma_ij=-(dv_i/dx_j+dv_j/dx_i)/(2 a H/h)",
            "tweb": "T_ij=partial_i partial_j Phi with laplacian(Phi)=delta",
        },
    }
    (output_dir / "web_classifier_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    centre = np.asarray(target["void_centre"], dtype=np.float64)
    plot_comparison(
        delta,
        v_lambda,
        t_lambda,
        component_mask,
        wall,
        v_flag,
        t_flag,
        centre,
        args.box_size,
        args.slice_half_width,
        output_dir / "web_classifier_comparison.png",
    )
    np.savez_compressed(
        output_dir / "web_classifier_centre_slice.npz",
        delta=centered_slice(delta, centre, args.slice_half_width),
        divergence=centered_slice(divergence, centre, args.slice_half_width),
        v_lambda=centered_slice(v_lambda, centre, args.slice_half_width),
        t_lambda=centered_slice(t_lambda, centre, args.slice_half_width),
        wall=centered_slice(wall, centre, args.slice_half_width),
        v_flag=centered_slice(v_flag, centre, args.slice_half_width),
        t_flag=centered_slice(t_flag, centre, args.slice_half_width),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
