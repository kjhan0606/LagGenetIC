#!/usr/bin/env python3
"""Verify the V4 target-1 sign pair and 16-rank lagRamses hand-off."""

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
) -> tuple[int, float]:
    total_values = 0
    for effective_size in (base_size, 2 * base_size):
        normal_dir = normal_case / f"v4_target1_normal.grafic_{effective_size}"
        inverted_dir = inverted_case / f"v4_target1_inverted.grafic_{effective_size}"
        normal_density = normal_dir / "ic_deltab"
        inverted_density = inverted_dir / "ic_deltab"
        if grafic_shape(normal_density) != grafic_shape(inverted_density):
            raise AssertionError(f"level {effective_size}: grid shapes differ")
        values, _ = check_sign_file(normal_density, inverted_density)
        total_values += values
        for filename in ("ic_particle_ids", "ic_refmap"):
            if (normal_dir / filename).read_bytes() != (
                inverted_dir / filename
            ).read_bytes():
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
    return total_values, target_mean


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
    fine_cube_cells: int,
    ranks: int,
) -> tuple[int, int]:
    log = (ramses_case / "ramses.log").read_text()
    matches = re.findall(r"Level\s+10 has\s+(\d+) grids", log)
    if not matches:
        raise AssertionError("lagRamses did not report a level-10 mesh")
    fine_grids = int(matches[0])
    if fine_grids != target_count:
        raise AssertionError(
            f"lagRamses loaded {fine_grids} level-10 grids, expected {target_count}"
        )
    outputs = sorted(ramses_case.glob("output_[0-9][0-9][0-9][0-9][0-9]"))
    if not outputs:
        raise FileNotFoundError("lagRamses did not write an initial snapshot")
    part_files = sorted(outputs[0].glob("part_*.out*"))
    if len(part_files) != ranks:
        raise AssertionError(f"RAMSES wrote {len(part_files)} files, expected {ranks}")
    expected_particles = base_size**3 + 7 * fine_grids
    id_capacity = base_size**3 + fine_cube_cells
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
    expected_fine_particles = 8 * fine_grids
    if fine_particles != expected_fine_particles:
        raise AssertionError(
            f"RAMSES wrote {fine_particles} fine particles, "
            f"expected {expected_fine_particles}"
        )
    return fine_grids, particle_count


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
    values, target_mean = check_ic_pair(
        args.normal_case,
        args.inverted_case,
        target_ids.size,
        args.base_size,
    )
    fine_shape = grafic_shape(
        args.inverted_case
        / f"v4_target1_inverted.grafic_{2 * args.base_size}"
        / "ic_particle_ids"
    )
    print(
        f"IC pair: exact sign reversal over {values} cells; "
        f"target mean delta={target_mean:.12e}"
    )
    if args.ramses_case is not None:
        fine_grids, particles = check_ramses(
            args.ramses_case,
            target_ids.size,
            args.base_size,
            int(np.prod(fine_shape)),
            args.ranks,
        )
        print(
            f"lagRamses: {fine_grids} level-10 grids, "
            f"{particles} particles over {args.ranks} ranks"
        )
    print("V4 TARGET-1 ONE-LEVEL ZOOM PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
