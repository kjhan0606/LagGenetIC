#!/usr/bin/env python3
"""Verify the compact rank-729 level-14 GRAFIC and lagRamses hand-off."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

import numpy as np

from verify_target1_zoom import (
    check_ramses,
    discover_effective_sizes,
    fortran_record_stream,
    grafic_shape,
)


PREFIX = "v4_compact729_inverted"
EXPECTED_LEVELS = (
    (512, (512, 512, 512)),
    (1024, (128, 128, 128)),
    (2048, (128, 128, 128)),
    (4096, (256, 256, 256)),
    (8192, (512, 512, 512)),
    (16384, (1024, 1024, 1024)),
)
FLOAT_FIELDS = (
    "ic_deltab",
    "ic_refmap",
    "ic_poscx",
    "ic_poscy",
    "ic_poscz",
    "ic_velcx",
    "ic_velcy",
    "ic_velcz",
    "ic_pvar_00001",
)


def expected_record_file_size(path: Path, shape: tuple[int, int, int], itemsize: int) -> int:
    with path.open("rb") as handle:
        marker = handle.read(4)
        if len(marker) != 4:
            raise ValueError(f"{path}: missing GRAFIC header")
        header_size = struct.unpack("<i", marker)[0]
    nx, ny, nz = shape
    return header_size + 8 + nz * (nx * ny * itemsize + 8)


def check_file_layout(path: Path, shape: tuple[int, int, int], itemsize: int) -> None:
    if grafic_shape(path) != shape:
        raise AssertionError(f"{path}: unexpected GRAFIC shape {grafic_shape(path)}")
    expected = expected_record_file_size(path, shape, itemsize)
    if path.stat().st_size != expected:
        raise AssertionError(
            f"{path}: size {path.stat().st_size} differs from {expected}"
        )


def count_target_mask(case: Path, target_count: int) -> float:
    directory = case / f"{PREFIX}.grafic_512"
    density_records = fortran_record_stream(directory / "ic_deltab")
    mask_records = fortran_record_stream(directory / "ic_refmap")
    if next(density_records) != next(mask_records):
        raise AssertionError("base density and refinement-map headers differ")
    count = 0
    density_sum = 0.0
    for density_payload, mask_payload in zip(
        density_records, mask_records, strict=True
    ):
        density = np.frombuffer(density_payload, dtype="<f4")
        mask = np.frombuffer(mask_payload, dtype="<f4") != 0.0
        count += int(np.count_nonzero(mask))
        density_sum += float(np.sum(density[mask], dtype=np.float64))
    if count != target_count:
        raise AssertionError(f"base refmap selects {count} cells, expected {target_count}")
    mean_density = density_sum / count
    if mean_density >= 0.0:
        raise AssertionError(
            f"inverted compact target has nonnegative mean delta {mean_density}"
        )
    return mean_density


def check_grafic(case: Path, target_count: int) -> tuple[list[int], int, float]:
    expected_sizes = [effective_size for effective_size, _ in EXPECTED_LEVELS]
    sizes = discover_effective_sizes(case, PREFIX)
    if sizes != expected_sizes:
        raise AssertionError(f"GRAFIC hierarchy {sizes} differs from {expected_sizes}")
    cube_cells: list[int] = []
    for effective_size, shape in EXPECTED_LEVELS:
        directory = case / f"{PREFIX}.grafic_{effective_size}"
        for filename in FLOAT_FIELDS:
            check_file_layout(directory / filename, shape, 4)
        check_file_layout(directory / "ic_particle_ids", shape, 8)
        cube_cells.append(int(np.prod(shape, dtype=np.int64)))
    target_mean = count_target_mask(case, target_count)
    return cube_cells, sum(cube_cells), target_mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("genetic_case", type=Path)
    parser.add_argument("target_id_file", type=Path)
    parser.add_argument("--ramses-case", type=Path)
    parser.add_argument("--ranks", type=int, default=64)
    args = parser.parse_args()

    target_ids = np.loadtxt(args.target_id_file, dtype=np.int64, ndmin=1)
    if target_ids.size != 3490 or np.any(np.diff(target_ids) <= 0):
        raise ValueError("compact rank-729 ID file must contain 3490 ordered IDs")
    cube_cells, total_cells, target_mean = check_grafic(
        args.genetic_case, target_ids.size
    )
    print(
        f"compact rank 729: {total_cells} verified GRAFIC cells; "
        f"target mean delta={target_mean:.12e}"
    )
    if args.ramses_case is not None:
        grid_counts, particles = check_ramses(
            args.ramses_case,
            target_ids.size,
            512,
            cube_cells[1:],
            args.ranks,
        )
        mesh = ", ".join(
            f"level-{level}={count}" for level, count in grid_counts.items()
        )
        print(f"lagRamses: {mesh}; {particles} particles over {args.ranks} ranks")
    print("V4 COMPACT RANK-729 LEVEL-14 DMO HAND-OFF PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
