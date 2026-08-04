#!/usr/bin/env python3
"""Validate the two 128^3 monofonIC -> GenetIC -> lagRamses V1 cases."""

from __future__ import annotations

import argparse
import glob
import os
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

POS_TOL = 1.0e-6
GRAFIC_FIELDS = (
    "ic_deltab",
    "ic_particle_ids",
    "ic_poscx",
    "ic_poscy",
    "ic_poscz",
    "ic_pvar_00001",
    "ic_velcx",
    "ic_velcy",
    "ic_velcz",
)


def fortran_records(path: Path) -> list[bytes]:
    data = path.read_bytes()
    records: list[bytes] = []
    offset = 0
    while offset + 4 <= len(data):
        (length,) = struct.unpack_from("<i", data, offset)
        offset += 4
        payload = data[offset : offset + length]
        offset += length
        (trailer,) = struct.unpack_from("<i", data, offset)
        offset += 4
        if trailer != length:
            raise ValueError(f"{path}: record marker mismatch {length} != {trailer}")
        records.append(payload)
    if offset != len(data):
        raise ValueError(f"{path}: trailing {len(data) - offset} bytes")
    return records


def read_grafic_cube(path: Path, dtype: str) -> tuple[tuple[int, int, int], np.ndarray]:
    records = fortran_records(path)
    dims = struct.unpack("<iii", records[0][:12])
    n1, n2, n3 = dims
    cube = np.stack(
        [np.frombuffer(records[1 + iz], dtype).reshape(n2, n1) for iz in range(n3)],
        axis=0,
    )
    return dims, cube


def read_part(path: Path) -> tuple[np.ndarray, np.ndarray]:
    records = fortran_records(path)
    (ndim,) = struct.unpack("<i", records[1][:4])
    (npart,) = struct.unpack("<i", records[2][:4])
    pos0 = 8
    idp_index = pos0 + 2 * ndim + 1
    expected = npart * 8
    if len(records[pos0]) != expected or len(records[idp_index]) != expected:
        raise ValueError(
            f"{path}: RAMSES part layout changed "
            f"(npart={npart}, pos={len(records[pos0])}, idp={len(records[idp_index])})"
        )
    ids = np.frombuffer(records[idp_index], "<i8")
    xyz = np.stack(
        [np.frombuffer(records[pos0 + axis], "<f8") for axis in range(3)], axis=1
    )
    return ids, xyz


def read_part_dir(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    files = sorted(Path(p) for p in glob.glob(str(path / "part_*.out*")))
    if not files:
        raise FileNotFoundError(f"no part_*.out* files under {path}")
    ids, xyz = zip(*(read_part(p) for p in files))
    return np.concatenate(ids), np.concatenate(xyz, axis=0), len(files)


def check_white_noise(slab_h5: Path, ksec_h5: Path) -> None:
    with h5py.File(slab_h5, "r") as left, h5py.File(ksec_h5, "r") as right:
        a = left["WhiteNoise"][...]
        b = right["WhiteNoise"][...]
    if a.shape != (128, 128, 128) or b.shape != a.shape:
        raise AssertionError(f"unexpected white-noise shapes: {a.shape}, {b.shape}")
    if not np.array_equal(a, b):
        delta = float(np.max(np.abs(a - b)))
        raise AssertionError(f"slab/k-section white noise differs; max abs={delta:.3e}")
    print(
        f"  white noise: BIT-IDENTICAL shape={a.shape} "
        f"mean={a.mean():.3e} std={a.std():.6f}"
    )


def check_grafic_identity(slab_dir: Path, ksec_dir: Path) -> None:
    for name in GRAFIC_FIELDS:
        left = slab_dir / name
        right = ksec_dir / name
        if left.read_bytes() != right.read_bytes():
            raise AssertionError(f"GenetIC grafic field differs: {name}")
    print(f"  GenetIC grafic: BIT-IDENTICAL ({len(GRAFIC_FIELDS)} fields)")


def position_residual(
    ids: np.ndarray,
    xyz: np.ndarray,
    dims: tuple[int, int, int],
    posc: dict[str, np.ndarray],
    box: float,
) -> float:
    n1, n2, n3 = dims
    if ids.min() < 0 or ids.max() >= n1 * n2 * n3:
        return float("inf")
    ix = ids // (n2 * n3)
    iy = (ids // n3) % n2
    iz = ids % n3
    centres = (np.stack([ix, iy, iz], axis=1) + 0.5) / np.array(dims)
    displacements = np.stack(
        [posc["x"][iz, iy, ix], posc["y"][iz, iy, ix], posc["z"][iz, iy, ix]],
        axis=1,
    ) / box
    expected = (centres + displacements) % 1.0
    box_fraction = xyz / (box if float(xyz.max()) > 1.5 else 1.0)
    residual = np.abs(((box_fraction - expected + 0.5) % 1.0) - 0.5)
    return float(residual.max())


def check_roundtrip(case: Path, label: str, box: float) -> None:
    grafic = case / "v1.grafic_128"
    output = case / "output_00001"
    dims, ic_ids = read_grafic_cube(grafic / "ic_particle_ids", "<i8")
    posc = {
        axis: read_grafic_cube(grafic / f"ic_posc{axis}", "<f4")[1]
        for axis in "xyz"
    }
    ids, xyz, nfiles = read_part_dir(output)
    identity = np.array_equal(np.sort(ids), np.sort(ic_ids.ravel().astype("<i8")))
    residual = position_residual(ids, xyz, dims, posc, box)
    if not identity:
        raise AssertionError(f"{label}: RAMSES idp multiset differs from GenetIC IDs")
    if residual >= POS_TOL:
        raise AssertionError(
            f"{label}: ID-indexed position residual {residual:.3e} >= {POS_TOL:.1e}"
        )
    scrambled = ids.copy()
    np.random.default_rng(12345).shuffle(scrambled)
    scrambled_residual = position_residual(scrambled, xyz, dims, posc, box)
    if scrambled_residual <= 0.1:
        raise AssertionError(f"{label}: scrambled-ID negative control has no teeth")
    print(
        f"  {label}: PASS N={dims[0]} npart={ids.size} files={nfiles} "
        f"residual={residual:.3e} scrambled={scrambled_residual:.3e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--box", type=float, default=64.0)
    args = parser.parse_args()
    work = args.workdir.resolve()

    print("== V1 128^3 full-stack validation ==")
    check_white_noise(work / "mono_slab/wn.h5", work / "mono_ksec/wn.h5")
    check_grafic_identity(work / "slab/v1.grafic_128", work / "ksec/v1.grafic_128")
    check_roundtrip(work / "slab", "slab", args.box)
    check_roundtrip(work / "ksec", "k-section", args.box)
    print("V1 FULL-STACK REGRESSION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

