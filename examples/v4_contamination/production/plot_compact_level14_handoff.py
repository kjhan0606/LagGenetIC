#!/usr/bin/env python3
"""Summarize and plot the verified compact level-14 DMO hand-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def grafic_shape(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        marker = handle.read(4)
        if len(marker) != 4:
            raise ValueError(f"{path}: missing GRAFIC header")
        length = struct.unpack("<i", marker)[0]
        payload = handle.read(length)
        trailer = handle.read(4)
    if len(payload) < 12 or trailer != marker:
        raise ValueError(f"{path}: invalid GRAFIC header record")
    return struct.unpack("<iii", payload[:12])


def read_times(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", maxsplit=1)
        values[key] = float(value)
    return values


def human_count(value: int) -> str:
    if value >= 10**9:
        return f"{value / 10**9:.2f}B"
    if value >= 10**6:
        return f"{value / 10**6:.1f}M"
    if value >= 10**3:
        return f"{value / 10**3:.1f}k"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if not (args.run / ".complete").exists():
        raise FileNotFoundError(f"{args.run}: completion gate is absent")
    args.output.mkdir(parents=True, exist_ok=True)

    prefix = "v4_compact726_inverted.grafic_"
    hierarchy = []
    for directory in (args.run / "genetic").glob(f"{prefix}*"):
        effective_size = int(directory.name.removeprefix(prefix))
        shape = grafic_shape(directory / "ic_deltab")
        hierarchy.append((effective_size, shape, int(np.prod(shape, dtype=np.int64))))
    hierarchy.sort()
    levels = np.asarray([int(round(np.log2(size))) for size, _, _ in hierarchy])
    dense_cells = np.asarray([cells for _, _, cells in hierarchy], dtype=np.int64)
    if levels.tolist() != [9, 10, 11, 12, 13, 14]:
        raise ValueError(f"unexpected compact hierarchy: {levels.tolist()}")

    verification = (args.run / "verify_ramses.log").read_text()
    measured_grids = {
        int(level): int(count)
        for level, count in re.findall(r"level-(\d+)=(\d+)", verification)
    }
    particle_match = re.search(r"(\d+) particles over (\d+) ranks", verification)
    mean_match = re.search(r"target mean delta=([+\-0-9.eE]+)", verification)
    if (
        particle_match is None
        or mean_match is None
        or sorted(measured_grids) != levels[1:].tolist()
    ):
        raise ValueError("verified RAMSES hierarchy summary is incomplete")

    timing_files = {
        "lagCAMB": args.run / "camb" / "camb.time",
        "GenetIC": args.run / "genetic" / "genetic.time",
        "IC check": args.run / "verify_ic.time",
        "RAMSES": args.run / "ramses" / "ramses.time",
        "ID check": args.run / "verify_ramses.time",
    }
    timings = {name: read_times(path) for name, path in timing_files.items()}
    failed_times = sorted(
        (args.run / "ramses").glob("ramses.ngridmax1200000.failed.*.time")
    )
    capacity_probe = read_times(failed_times[-1]) if failed_times else None

    transfer = np.loadtxt(args.run / "camb" / "camb_transfer_z49_level14.dat")
    peak_match = re.search(
        r"Peak memory usage:\s*([0-9.]+)([GM]B)",
        (args.run / "genetic" / "genetic.log").read_text(),
    )
    peak_memory = None if peak_match is None else f"{peak_match.group(1)}{peak_match.group(2)}"
    report = {
        "run": str(args.run.resolve()),
        "effective_hierarchy": [size for size, _, _ in hierarchy],
        "grafic_shapes": {str(size): list(shape) for size, shape, _ in hierarchy},
        "dense_grafic_cells_by_level": {
            str(level): int(cells) for level, cells in zip(levels, dense_cells, strict=True)
        },
        "dense_grafic_cells_total": int(dense_cells.sum()),
        "initial_ramses_grids": measured_grids,
        "initial_particles": int(particle_match.group(1)),
        "mpi_ranks": int(particle_match.group(2)),
        "target_mean_delta": float(mean_match.group(1)),
        "transfer_rows": int(transfer.shape[0]),
        "transfer_k_min_h_mpc": float(transfer[0, 0]),
        "transfer_k_max_h_mpc": float(transfer[-1, 0]),
        "genetic_peak_memory": peak_memory,
        "timings_seconds": timings,
        "discarded_capacity_probe_seconds": capacity_probe,
    }
    (args.output / "compact_level14_handoff.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), constrained_layout=True)
    x = np.arange(levels.size)
    axes[0].bar(
        x - 0.18,
        dense_cells,
        width=0.36,
        color="#4c78a8",
        label="dense GRAFIC cells",
    )
    grid_values = np.asarray(
        [measured_grids.get(level, np.nan) for level in levels], dtype=float
    )
    axes[0].bar(
        x + 0.18,
        grid_values,
        width=0.36,
        color="#f58518",
        label="RAMSES AMR grids",
    )
    for xpos, value in zip(x - 0.18, dense_cells, strict=True):
        axes[0].annotate(
            human_count(int(value)),
            (xpos, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            rotation=45,
        )
    for xpos, value in zip(x + 0.18, grid_values, strict=True):
        if np.isfinite(value):
            axes[0].annotate(
                human_count(int(value)),
                (xpos, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                rotation=45,
            )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [str(level) for level in levels])
    axes[0].set_xlabel("mesh level")
    axes[0].set_ylabel("number per level")
    axes[0].grid(axis="y", alpha=0.25, which="both")
    axes[0].legend(fontsize=8, loc="upper left")

    stage_names = list(timings)
    wall_times = np.asarray([timings[name]["real_seconds"] for name in stage_names])
    colors = ["#72b7b2", "#4c78a8", "#54a24b", "#e45756", "#b279a2"]
    bars = axes[1].barh(stage_names, wall_times, color=colors)
    axes[1].bar_label(bars, fmt="%.1f s", padding=3, fontsize=8)
    axes[1].set_xlim(0.0, 1.17 * float(wall_times.max()))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("wall time [s]")
    axes[1].grid(axis="x", alpha=0.25)
    if peak_memory is not None:
        axes[1].text(
            0.98,
            0.04,
            f"GenetIC peak: {peak_memory}",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )

    for label, axis in zip(("(a)", "(b)"), axes, strict=True):
        axis.text(
            0.98,
            0.96,
            label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
    fig.savefig(args.output / "fig_compact_level14_handoff.png", dpi=260)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
