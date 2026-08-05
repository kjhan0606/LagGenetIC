#!/usr/bin/env python3
"""Verify an evolved HOP halo and its initial Lagrangian ID mask."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from hop_to_genetic_id import FortranRecordReader, _scalar, find_part_files


def _array(payload: bytes, endian: str, count: int, kind: str) -> np.ndarray:
    for size in (8, 4):
        if len(payload) == size * count:
            dtype = np.dtype(f"{endian}{kind}{size}")
            return np.frombuffer(payload, dtype=dtype).copy()
    raise ValueError(
        f"record has {len(payload)} bytes for {count} {kind}-valued entries"
    )


def read_positions_and_ids(path: Path) -> tuple[int, int, np.ndarray, np.ndarray]:
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
        coordinates = [
            _array(records.read(), endian, npart, "f") for _ in range(ndim)
        ]
        for _ in range(ndim + 1):
            records.skip()
        ids = _array(records.read(), endian, npart, "i").astype(
            np.int64, copy=False
        )

    if ndim != 3:
        raise ValueError(f"{path}: expected ndim=3, found {ndim}")
    if nstar_tot != 0:
        raise ValueError(f"{path}: expected a DMO snapshot, found stars")
    return ncpu, nstar_tot, np.column_stack(coordinates), ids


def load_initial_positions(output_dir: Path, grid_size: int) -> np.ndarray:
    count = grid_size**3
    positions = np.empty((count, 3), dtype=np.float64)
    seen = np.zeros(count, dtype=np.bool_)
    files = find_part_files(output_dir)
    declared_ncpu: int | None = None
    for path in files:
        ncpu, _, local_positions, ids = read_positions_and_ids(path)
        if declared_ncpu is None:
            declared_ncpu = ncpu
        if ncpu != declared_ncpu:
            raise ValueError(f"{path}: inconsistent ncpu")
        if np.any(ids < 0) or np.any(ids >= count):
            raise ValueError(f"{path}: initial particle ID lies outside the grid")
        if np.any(seen[ids]):
            raise ValueError(f"{path}: duplicate initial particle ID")
        positions[ids] = local_positions
        seen[ids] = True
    if declared_ncpu != len(files):
        raise ValueError("initial snapshot has the wrong number of rank files")
    if not np.all(seen):
        raise ValueError("initial snapshot does not contain the full ID permutation")
    return positions


def catalogue_count(path: Path, halo_id: int) -> int:
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if int(fields[0]) == halo_id:
            return int(fields[1])
    raise ValueError(f"halo {halo_id} is absent from {path}")


def periodic_interval(values: np.ndarray) -> tuple[float, float, float]:
    ordered = np.sort(np.mod(values, 1.0))
    wrapped = np.concatenate((ordered, ordered[:1] + 1.0))
    gaps = np.diff(wrapped)
    gap_index = int(np.argmax(gaps))
    start = float(wrapped[gap_index + 1] % 1.0)
    width = float(1.0 - gaps[gap_index])
    centre = float((start + 0.5 * width) % 1.0)
    return start, width, centre


def check_genetic(binary: Path, ids: Path, grid_size: int) -> None:
    with tempfile.TemporaryDirectory(prefix="voidsim_v2_genetic_") as directory:
        root = Path(directory)
        roundtrip = root / "roundtrip.id"
        parameter_file = root / "genetic.txt"
        parameter_file.write_text(
            "\n".join(
                [
                    f"base_grid 64.0 {grid_size}",
                    f"id_file {ids.resolve()}",
                    f"dump_id_file {roundtrip}",
                    "",
                ]
            )
        )
        result = subprocess.run(
            [str(binary.resolve()), str(parameter_file)],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"GenetIC round trip failed:\n{result.stdout}")
        if ids.read_text() != roundtrip.read_text():
            raise AssertionError("GenetIC changed the evolved halo ID mask")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_output", type=Path)
    parser.add_argument("halo_catalogue", type=Path)
    parser.add_argument("id_file", type=Path)
    parser.add_argument("--halo-id", type=int, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--genetic-binary", type=Path, required=True)
    args = parser.parse_args()

    selected = np.loadtxt(args.id_file, dtype=np.int64, ndmin=1)
    expected_count = catalogue_count(args.halo_catalogue, args.halo_id)
    if selected.size != expected_count:
        raise AssertionError(
            f"id_file has {selected.size} IDs but HOP reports {expected_count}"
        )
    if np.any(selected[1:] <= selected[:-1]):
        raise AssertionError("id_file is not strictly sorted")

    positions = load_initial_positions(args.initial_output, args.grid_size)
    lagrangian_positions = positions[selected]
    if not np.all(np.isfinite(lagrangian_positions)):
        raise AssertionError("Lagrangian mask contains a non-finite position")

    intervals = [
        periodic_interval(lagrangian_positions[:, dimension])
        for dimension in range(3)
    ]
    check_genetic(args.genetic_binary, args.id_file, args.grid_size)

    widths = ", ".join(f"{interval[1]:.6f}" for interval in intervals)
    centres = ", ".join(f"{interval[2]:.6f}" for interval in intervals)
    print(
        f"PASS: halo {args.halo_id} has {selected.size} particles and an exact "
        "GenetIC id_file round trip"
    )
    print(f"Lagrangian periodic envelope widths: ({widths})")
    print(f"Lagrangian periodic envelope centre: ({centres})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
