#!/usr/bin/env python3
"""Verify the V4 target-1 sign pair and multilevel lagRamses hand-off."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
import re
import struct

import numpy as np


def fortran_record_stream(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while marker := handle.read(4):
            if len(marker) != 4:
                raise ValueError(f"{path}: truncated leading record marker")
            length = struct.unpack("<i", marker)[0]
            if length < 0:
                raise ValueError(f"{path}: negative record length {length}")
            payload = handle.read(length)
            trailer = handle.read(4)
            if len(payload) != length or len(trailer) != 4:
                raise ValueError(f"{path}: truncated Fortran record")
            if struct.unpack("<i", trailer)[0] != length:
                raise ValueError(f"{path}: record marker mismatch")
            yield payload


def grafic_shape(path: Path) -> tuple[int, int, int]:
    header = next(fortran_record_stream(path))
    return struct.unpack("<iii", header[:12])


def files_equal(first: Path, second: Path, chunk_size: int = 8 * 1024**2) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as first_handle, second.open("rb") as second_handle:
        while first_chunk := first_handle.read(chunk_size):
            if first_chunk != second_handle.read(chunk_size):
                return False
        return second_handle.read(1) == b""


def discover_effective_sizes(case: Path, prefix: str) -> list[int]:
    sizes = []
    pattern = re.compile(rf"{re.escape(prefix)}\.grafic_(\d+)$")
    for directory in case.glob(f"{prefix}.grafic_*"):
        match = pattern.fullmatch(directory.name)
        if directory.is_dir() and match:
            sizes.append(int(match.group(1)))
    return sorted(sizes)


def check_sign_file(normal: Path, inverted: Path) -> tuple[int, float]:
    normal_records = fortran_record_stream(normal)
    inverted_records = fortran_record_stream(inverted)
    normal_header = next(normal_records)
    inverted_header = next(inverted_records)
    if normal_header != inverted_header:
        raise AssertionError(f"{normal.name}: normal/inverted headers differ")
    values = 0
    maximum_residual = 0.0
    for plane, (normal_payload, inverted_payload) in enumerate(
        zip(normal_records, inverted_records, strict=True)
    ):
        normal_values = np.frombuffer(normal_payload, dtype="<f4")
        inverted_values = np.frombuffer(inverted_payload, dtype="<f4")
        if normal_values.shape != inverted_values.shape:
            raise AssertionError(f"{normal.name}: plane {plane} shape differs")
        residual = inverted_values + normal_values
        if not np.all(np.isfinite(residual)):
            raise AssertionError(f"{normal.name}: plane {plane} contains non-finite data")
        if np.any(residual != 0.0):
            maximum_residual = max(maximum_residual, float(np.max(np.abs(residual))))
        values += normal_values.size
    if maximum_residual != 0.0:
        raise AssertionError(
            f"{normal.name}: maximum sign-reversal residual {maximum_residual}"
        )
    return values, maximum_residual


def count_mask_and_mean_density(case: Path, effective_size: int) -> tuple[int, float]:
    directory = case / f"v4_target1_normal.grafic_{effective_size}"
    density_records = fortran_record_stream(directory / "ic_deltab")
    refmap_records = fortran_record_stream(directory / "ic_refmap")
    if next(density_records) != next(refmap_records):
        raise AssertionError("density and refmap GRAFIC headers differ")
    count = 0
    density_sum = 0.0
    for density_payload, mask_payload in zip(
        density_records, refmap_records, strict=True
    ):
        density = np.frombuffer(density_payload, dtype="<f4")
        mask = np.frombuffer(mask_payload, dtype="<f4") != 0.0
        count += int(np.count_nonzero(mask))
        density_sum += float(np.sum(density[mask], dtype=np.float64))
    if count == 0:
        raise AssertionError("the coarse refinement mask is empty")
    return count, density_sum / count


def check_ic_pair(
    normal_case: Path,
    inverted_case: Path,
    target_count: int,
    base_size: int,
) -> tuple[int, float, list[int], list[int]]:
    normal_prefix = "v4_target1_normal"
    inverted_prefix = "v4_target1_inverted"
    effective_sizes = discover_effective_sizes(normal_case, normal_prefix)
    inverted_sizes = discover_effective_sizes(inverted_case, inverted_prefix)
    if effective_sizes != inverted_sizes:
        raise AssertionError(
            "normal/inverted effective GRAFIC levels differ: "
            f"{effective_sizes} != {inverted_sizes}"
        )
    if not effective_sizes or effective_sizes[0] != base_size:
        raise AssertionError(
            f"the GRAFIC hierarchy must begin at effective size {base_size}"
        )
    for coarse, fine in zip(effective_sizes, effective_sizes[1:]):
        if fine != 2 * coarse:
            raise AssertionError(
                f"non-consecutive GRAFIC hierarchy: {coarse} -> {fine}"
            )

    total_values = 0
    cube_cells = []
    for effective_size in effective_sizes:
        normal_dir = normal_case / f"{normal_prefix}.grafic_{effective_size}"
        inverted_dir = inverted_case / f"{inverted_prefix}.grafic_{effective_size}"
        normal_density = normal_dir / "ic_deltab"
        inverted_density = inverted_dir / "ic_deltab"
        normal_shape = grafic_shape(normal_density)
        if normal_shape != grafic_shape(inverted_density):
            raise AssertionError(f"level {effective_size}: grid shapes differ")
        cube_cells.append(int(np.prod(normal_shape, dtype=np.int64)))
        values, _ = check_sign_file(normal_density, inverted_density)
        total_values += values
        for filename in ("ic_particle_ids", "ic_refmap"):
            if not files_equal(normal_dir / filename, inverted_dir / filename):
                raise AssertionError(
                    f"level {effective_size}: {filename} changed under reverse"
                )

    mask_count, target_mean = count_mask_and_mean_density(normal_case, base_size)
    if mask_count != target_count:
        raise AssertionError(
            f"coarse refmap contains {mask_count} cells, expected {target_count}"
        )
    if target_mean <= 0.0:
        raise AssertionError(f"normal target mean delta is not positive: {target_mean}")
    return total_values, target_mean, effective_sizes, cube_cells


def read_part_ids(path: Path) -> np.ndarray:
    records = fortran_record_stream(path)
    _ = next(records)
    ndim = struct.unpack("<i", next(records)[:4])[0]
    npart = struct.unpack("<i", next(records)[:4])[0]
    identity_index = 8 + 2 * ndim + 1
    payload = None
    for record_index, record in enumerate(records, start=3):
        if record_index == identity_index:
            payload = record
            break
    if payload is None:
        raise ValueError(f"{path}: particle-ID record is absent")
    if len(payload) == 8 * npart:
        ids = np.frombuffer(payload, dtype="<i8").astype(np.int64, copy=True)
    elif len(payload) == 4 * npart:
        ids = np.frombuffer(payload, dtype="<i4").astype(np.int64, copy=True)
    else:
        raise ValueError(f"{path}: unsupported particle-ID record")
    return ids


def check_ramses(
    ramses_case: Path,
    target_count: int,
    base_size: int,
    fine_cube_cells: list[int],
    ranks: int,
) -> tuple[dict[int, int], int]:
    log = (ramses_case / "ramses.log").read_text()
    reported_grids: dict[int, int] = {}
    for level_text, grids_text in re.findall(
        r"Level\s+(\d+) has\s+(\d+) grids", log
    ):
        reported_grids.setdefault(int(level_text), int(grids_text))
    base_level = base_size.bit_length() - 1
    if 2**base_level != base_size:
        raise ValueError("base size must be a power of two")
    expected_levels = list(
        range(base_level + 1, base_level + 1 + len(fine_cube_cells))
    )
    grid_counts = {}
    for offset, (level, capacity) in enumerate(
        zip(expected_levels, fine_cube_cells, strict=True)
    ):
        if level not in reported_grids:
            raise AssertionError(f"lagRamses did not report a level-{level} mesh")
        grids = reported_grids[level]
        minimum = target_count * 8**offset
        if grids < minimum:
            raise AssertionError(
                f"lagRamses loaded {grids} level-{level} grids, fewer than the "
                f"{minimum} cells implied by the selected target"
            )
        if grids > capacity:
            raise AssertionError(
                f"lagRamses loaded {grids} level-{level} grids, exceeding the "
                f"{capacity}-cell level patch"
            )
        grid_counts[level] = grids
    outputs = sorted(ramses_case.glob("output_[0-9][0-9][0-9][0-9][0-9]"))
    if not outputs:
        raise FileNotFoundError("lagRamses did not write an initial snapshot")
    part_files = sorted(outputs[0].glob("part_*.out*"))
    if len(part_files) != ranks:
        raise AssertionError(f"RAMSES wrote {len(part_files)} files, expected {ranks}")
    expected_particles = base_size**3 + 7 * sum(grid_counts.values())
    id_capacity = base_size**3 + sum(fine_cube_cells)
    seen = np.zeros(id_capacity, dtype=np.bool_)
    particle_count = 0
    fine_particles = 0
    for path in part_files:
        ids = read_part_ids(path)
        if np.unique(ids).size != ids.size:
            raise AssertionError(f"{path}: duplicate particle IDs within MPI rank")
        if np.any(ids < 0) or np.any(ids >= id_capacity):
            raise AssertionError(
                f"{path}: particle ID outside 0..{id_capacity - 1}"
            )
        if np.any(seen[ids]):
            raise AssertionError(f"{path}: particle ID duplicated across MPI ranks")
        seen[ids] = True
        particle_count += ids.size
        fine_particles += int(np.count_nonzero(ids >= base_size**3))
    if particle_count != expected_particles:
        raise AssertionError(
            f"RAMSES wrote {particle_count} particles, expected {expected_particles}"
        )
    first_level = expected_levels[0]
    expected_fine_particles = 8 * grid_counts[first_level]
    expected_fine_particles += 7 * sum(
        grid_counts[level] for level in expected_levels[1:]
    )
    if fine_particles != expected_fine_particles:
        raise AssertionError(
            f"RAMSES wrote {fine_particles} fine particles, "
            f"expected {expected_fine_particles}"
        )
    return grid_counts, particle_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("normal_case", type=Path)
    parser.add_argument("inverted_case", type=Path)
    parser.add_argument("target_id_file", type=Path)
    parser.add_argument("--ramses-case", type=Path)
    parser.add_argument("--base-size", type=int, default=512)
    parser.add_argument("--ranks", type=int, default=16)
    args = parser.parse_args()

    target_ids = np.loadtxt(args.target_id_file, dtype=np.int64, ndmin=1)
    if target_ids.size == 0 or np.any(np.diff(target_ids) <= 0):
        raise ValueError("target ID file must be non-empty and strictly increasing")
    values, target_mean, effective_sizes, cube_cells = check_ic_pair(
        args.normal_case,
        args.inverted_case,
        target_ids.size,
        args.base_size,
    )
    print(
        f"IC pair: exact sign reversal over {values} cells; "
        f"target mean delta={target_mean:.12e}; "
        f"effective sizes={','.join(map(str, effective_sizes))}"
    )
    if args.ramses_case is not None:
        grid_counts, particles = check_ramses(
            args.ramses_case,
            target_ids.size,
            args.base_size,
            cube_cells[1:],
            args.ranks,
        )
        mesh_summary = ", ".join(
            f"level-{level}={count}" for level, count in grid_counts.items()
        )
        print(
            f"lagRamses: {mesh_summary}; "
            f"{particles} particles over {args.ranks} ranks"
        )
    print("V4 TARGET-1 MULTILEVEL ZOOM PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
