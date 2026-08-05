#!/usr/bin/env python3
"""Verify the V3 normal/inverted zoom pair and lagRamses ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import struct

import numpy as np


def fortran_records(path: Path) -> list[bytes]:
    data = path.read_bytes()
    records: list[bytes] = []
    offset = 0
    while offset + 4 <= len(data):
        length = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        payload = data[offset : offset + length]
        offset += length
        trailer = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        if trailer != length:
            raise ValueError(f"{path}: record marker mismatch")
        records.append(payload)
    if offset != len(data):
        raise ValueError(f"{path}: trailing bytes")
    return records


def read_grafic_cube(path: Path, dtype: str) -> np.ndarray:
    records = fortran_records(path)
    n1, n2, n3 = struct.unpack("<iii", records[0][:12])
    return np.stack(
        [
            np.frombuffer(records[1 + iz], dtype).reshape(n2, n1)
            for iz in range(n3)
        ],
        axis=0,
    )


def read_part(path: Path) -> np.ndarray:
    records = fortran_records(path)
    ndim = struct.unpack("<i", records[1][:4])[0]
    npart = struct.unpack("<i", records[2][:4])[0]
    identity_index = 8 + 2 * ndim + 1
    payload = records[identity_index]
    if len(payload) == 8 * npart:
        dtype = "<i8"
    elif len(payload) == 4 * npart:
        dtype = "<i4"
    else:
        raise ValueError(f"{path}: unsupported particle-ID record")
    return np.frombuffer(payload, dtype).astype(np.int64, copy=True)


def read_output_ids(output_dir: Path) -> tuple[np.ndarray, int]:
    files = sorted(output_dir.glob("part_*.out*"))
    if not files:
        raise FileNotFoundError(f"{output_dir}: no particle files")
    return np.concatenate([read_part(path) for path in files]), len(files)


def check_sign_pair(
    normal: Path,
    inverted: Path,
    halo_ids: np.ndarray,
    grid_size: int,
) -> tuple[float, float]:
    for effective_size in (grid_size, 2 * grid_size):
        normal_dir = normal / f"v3_normal.grafic_{effective_size}"
        inverted_dir = inverted / f"v3_inverted.grafic_{effective_size}"
        density = read_grafic_cube(normal_dir / "ic_deltab", "<f4")
        density_inverted = read_grafic_cube(
            inverted_dir / "ic_deltab", "<f4"
        )
        if not np.array_equal(density_inverted, -density):
            residual = float(np.max(np.abs(density_inverted + density)))
            raise AssertionError(
                f"level {effective_size}: sign reversal residual {residual}"
            )
        for name in ("ic_particle_ids", "ic_refmap"):
            if (normal_dir / name).read_bytes() != (inverted_dir / name).read_bytes():
                raise AssertionError(
                    f"level {effective_size}: {name} changed under reverse"
                )

    coarse = read_grafic_cube(
        normal / f"v3_normal.grafic_{grid_size}" / "ic_deltab", "<f4"
    )
    coarse_inverted = read_grafic_cube(
        inverted / f"v3_inverted.grafic_{grid_size}" / "ic_deltab", "<f4"
    )
    refmap = read_grafic_cube(
        normal / f"v3_normal.grafic_{grid_size}" / "ic_refmap", "<f4"
    )
    ix = halo_ids // (grid_size * grid_size)
    iy = (halo_ids // grid_size) % grid_size
    iz = halo_ids % grid_size
    if np.count_nonzero(refmap) != halo_ids.size:
        raise AssertionError("coarse refmap size differs from the halo mask")
    if not np.all(refmap[iz, iy, ix] == 1):
        raise AssertionError("coarse refmap omits a selected halo cell")
    normal_mean = float(coarse[iz, iy, ix].mean())
    inverted_mean = float(coarse_inverted[iz, iy, ix].mean())
    if normal_mean <= 0 or inverted_mean != -normal_mean:
        raise AssertionError(
            "selected parent peak did not become an exactly inverted trough"
        )
    return normal_mean, inverted_mean


def check_ramses(
    ramses_case: Path,
    halo_ids: np.ndarray,
    grid_size: int,
) -> tuple[int, int, int]:
    log = (ramses_case / "ramses.log").read_text()
    matches = re.findall(r"Level\s+7 has\s+(\d+) grids", log)
    if not matches:
        raise AssertionError("lagRamses did not report a level-7 mesh")
    fine_grids = int(matches[0])
    if fine_grids <= 0 or fine_grids >= grid_size**3:
        raise AssertionError(f"invalid masked level-7 grid count {fine_grids}")

    outputs = sorted(ramses_case.glob("output_[0-9][0-9][0-9][0-9][0-9]"))
    if not outputs:
        raise FileNotFoundError("lagRamses did not write an initial snapshot")
    ids, nfiles = read_output_ids(outputs[0])
    expected_particles = grid_size**3 + 7 * fine_grids
    if ids.size != expected_particles:
        raise AssertionError(
            f"lagRamses wrote {ids.size} particles, expected {expected_particles}"
        )
    if np.unique(ids).size != ids.size:
        raise AssertionError("lagRamses particle IDs are not unique")
    if np.intersect1d(ids, halo_ids).size != 0:
        raise AssertionError("selected coarse halo particles were not replaced")
    fine_particles = int(np.count_nonzero(ids >= grid_size**3))
    if fine_particles != 8 * fine_grids:
        raise AssertionError(
            f"found {fine_particles} fine particles, expected {8 * fine_grids}"
        )
    return fine_grids, ids.size, nfiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("normal_case", type=Path)
    parser.add_argument("inverted_case", type=Path)
    parser.add_argument("ramses_case", type=Path)
    parser.add_argument("--halo-id-file", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    args = parser.parse_args()

    halo_ids = np.loadtxt(args.halo_id_file, dtype=np.int64, ndmin=1)
    normal_mean, inverted_mean = check_sign_pair(
        args.normal_case, args.inverted_case, halo_ids, args.grid_size
    )
    fine_grids, particles, nfiles = check_ramses(
        args.ramses_case, halo_ids, args.grid_size
    )

    print("V3 SIGN-INVERTED CLOSED-LOOP REGRESSION PASSED")
    print(
        f"halo mean delta: {normal_mean:.12e} -> {inverted_mean:.12e}"
    )
    print(
        f"lagRamses: {fine_grids} level-7 grids, {particles} particles, "
        f"{nfiles} MPI files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
