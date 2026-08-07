#!/usr/bin/env python3
"""Summarize and plot the measured target-1 multilevel scale gate."""

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


def directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


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
    parser.add_argument("--max-level", type=int, default=14)
    args = parser.parse_args()

    if not (args.run / ".complete").exists():
        raise FileNotFoundError(f"{args.run}: completion gate is absent")
    args.output.mkdir(parents=True, exist_ok=True)

    prefix = "v4_target1_normal.grafic_"
    hierarchy = []
    for directory in (args.run / "normal").glob(f"{prefix}*"):
        effective_size = int(directory.name.removeprefix(prefix))
        shape = grafic_shape(directory / "ic_deltab")
        hierarchy.append((effective_size, shape, int(np.prod(shape))))
    hierarchy.sort()
    if not hierarchy:
        raise FileNotFoundError("normal GRAFIC hierarchy is absent")
    base_level = int(round(np.log2(hierarchy[0][0])))
    measured_level = int(round(np.log2(hierarchy[-1][0])))
    if measured_level > args.max_level:
        raise ValueError("maximum plotted level lies below the measured hierarchy")

    levels = np.arange(base_level, args.max_level + 1)
    cumulative_cells = []
    cumulative = 0
    measured_cells = {int(round(np.log2(size))): cells for size, _, cells in hierarchy}
    last_axis = hierarchy[-1][1][0]
    for level in levels:
        if level in measured_cells:
            cumulative += measured_cells[level]
        else:
            last_axis *= 2
            cumulative += last_axis**3
        cumulative_cells.append(cumulative)

    verification = (args.run / "verify_zoom.log").read_text()
    measured_grids = {
        int(level): int(count)
        for level, count in re.findall(r"level-(\d+)=(\d+)", verification)
    }
    particle_match = re.search(r"(\d+) particles over (\d+) ranks", verification)
    if particle_match is None:
        raise ValueError("particle summary is absent from verify_zoom.log")
    target_count = np.loadtxt(args.run / "target1.id", dtype=np.int64, ndmin=1).size
    refined_levels = np.arange(base_level + 1, args.max_level + 1)
    minimum_grids = np.asarray(
        [target_count * 8 ** (level - base_level - 1) for level in refined_levels],
        dtype=np.int64,
    )

    timing_files = {
        "GenetIC normal": args.run / "normal" / "genetic.time",
        "GenetIC inverted": args.run / "inverted" / "genetic.time",
        "IC verification": args.run / "verify_ic.time",
        "RAMSES ingestion": args.run / "ramses" / "ramses.time",
        "ID verification": args.run / "verify_zoom.time",
    }
    timings = {name: read_times(path) for name, path in timing_files.items()}
    report = {
        "run": str(args.run.resolve()),
        "effective_hierarchy": [size for size, _, _ in hierarchy],
        "grafic_shapes": {str(size): list(shape) for size, shape, _ in hierarchy},
        "exact_sign_cells": sum(cells for _, _, cells in hierarchy),
        "target_parent_cells": int(target_count),
        "initial_ramses_grids": measured_grids,
        "initial_particles": int(particle_match.group(1)),
        "mpi_ranks": int(particle_match.group(2)),
        "timings_seconds": timings,
        "total_bytes": directory_bytes(args.run),
        "projection": {
            "levels": levels.tolist(),
            "cumulative_grafic_cells": [int(value) for value in cumulative_cells],
            "minimum_refined_grids": {
                str(level): int(value)
                for level, value in zip(refined_levels, minimum_grids, strict=True)
            },
        },
    }
    (args.output / "target1_level11_scaling.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    measured_mask = levels <= measured_level
    axes[0].plot(
        levels[measured_mask],
        np.asarray(cumulative_cells)[measured_mask],
        "o-",
        color="#1f77b4",
        linewidth=2,
        label="generated and verified",
    )
    projection_levels = levels[levels >= measured_level]
    projection_cells = np.asarray(cumulative_cells)[levels >= measured_level]
    axes[0].plot(
        projection_levels,
        projection_cells,
        "o--",
        color="#1f77b4",
        markerfacecolor="white",
        markevery=range(1, len(projection_levels)),
        linewidth=1.7,
        label="fixed-patch projection",
    )
    for level, cells in zip(levels, cumulative_cells, strict=True):
        axes[0].annotate(
            human_count(int(cells)),
            (level, cells),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("maximum IC level")
    axes[0].set_ylabel("cumulative GRAFIC cells")
    axes[0].set_xticks(levels)
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(
        refined_levels,
        minimum_grids,
        "o--",
        color="#333333",
        markerfacecolor="white",
        linewidth=1.7,
        label="selected-mask minimum",
    )
    measured_x = np.asarray(sorted(measured_grids))
    measured_y = np.asarray([measured_grids[level] for level in measured_x])
    axes[1].plot(
        measured_x,
        measured_y,
        "s-",
        color="#d62728",
        linewidth=2,
        label="RAMSES after buffer expansion",
    )
    for level, count in measured_grids.items():
        axes[1].annotate(
            f"{count:,}",
            (level, count),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("refined mesh level")
    axes[1].set_ylabel("refined grids")
    axes[1].set_xticks(refined_levels)
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8, loc="upper left")

    for label, axis in zip(("(a)", "(b)"), axes, strict=True):
        axis.text(
            0.98,
            0.04,
            label,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    fig.savefig(args.output / "fig_target1_level11_scaling.png", dpi=260)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
