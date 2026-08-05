#!/usr/bin/env python3
"""Verify the level-9 white-noise hand-off and lagRamses particle IDs."""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent.parent / "v2_hop_id_file"
sys.path.insert(0, str(V2_DIR))

from hop_to_genetic_id import find_part_files, read_dmo_ids  # noqa: E402


def check_transfer(path: Path) -> None:
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.lstrip().startswith("#") and line.strip():
                values = [float(value) for value in line.split()]
                if len(values) != 13:
                    raise AssertionError(
                        f"{path}: expected exactly 13 CAMB columns, found {len(values)}"
                    )
                rows.append(values)
    table = np.asarray(rows)
    if table.shape[0] < 100 or not np.all(np.isfinite(table)):
        raise AssertionError(f"{path}: incomplete or non-finite transfer table")
    if np.any(np.diff(table[:, 0]) <= 0) or table[-1, 0] < 25.0:
        raise AssertionError(f"{path}: k coverage does not reach the zoom Nyquist")
    print(
        f"  CAMB: rows={table.shape[0]} k={table[0, 0]:.3e}..{table[-1, 0]:.3e} h/Mpc"
    )


def check_white_noise(h5_path: Path, npy_path: Path, n: int) -> None:
    array = np.load(npy_path, mmap_mode="r")
    if array.shape != (n, n, n) or array.dtype != np.dtype("float64"):
        raise AssertionError(
            f"{npy_path}: expected ({n},{n},{n}) float64, got {array.shape} {array.dtype}"
        )
    total = 0
    sum1 = 0.0
    sum2 = 0.0
    with h5py.File(h5_path, "r") as handle:
        source = handle["WhiteNoise"]
        if source.shape != array.shape:
            raise AssertionError(f"white-noise shape mismatch: {source.shape} != {array.shape}")
        for start in range(0, n, 8):
            stop = min(start + 8, n)
            left = source[start:stop]
            right = np.asarray(array[start:stop])
            if not np.array_equal(left, right):
                raise AssertionError(f"white-noise conversion differs at x={start}:{stop}")
            total += right.size
            sum1 += float(right.sum(dtype=np.float64))
            sum2 += float(np.square(right).sum(dtype=np.float64))
    mean = sum1 / total
    std = np.sqrt(max(0.0, sum2 / total - mean * mean))
    if abs(mean) > 5.0e-4 or not 0.995 < std < 1.005:
        raise AssertionError(f"unexpected white-noise moments mean={mean} std={std}")
    print(f"  white noise: BIT-IDENTICAL shape={array.shape} mean={mean:.3e} std={std:.6f}")


def read_marker(handle, endian: str) -> int:
    payload = handle.read(4)
    if len(payload) != 4:
        raise EOFError("unexpected end of GRAFIC file")
    return struct.unpack(f"{endian}i", payload)[0]


def check_grafic_ids(path: Path, n: int) -> None:
    with path.open("rb") as handle:
        raw = handle.read(4)
        if len(raw) != 4:
            raise AssertionError(f"{path}: missing header")
        little = struct.unpack("<i", raw)[0]
        big = struct.unpack(">i", raw)[0]
        endian = "<" if little == 44 else ">" if big == 44 else None
        if endian is None:
            raise AssertionError(f"{path}: unsupported GRAFIC header marker {raw.hex()}")
        payload = handle.read(44)
        dims = struct.unpack(f"{endian}iii", payload[:12])
        if dims != (n, n, n) or read_marker(handle, endian) != 44:
            raise AssertionError(f"{path}: invalid dimensions or header trailer")
        ix = np.arange(n, dtype=np.int64)
        iy = np.arange(n, dtype=np.int64)[:, None]
        for iz in range(n):
            length = read_marker(handle, endian)
            payload = handle.read(length)
            if read_marker(handle, endian) != length or length not in (4*n*n, 8*n*n):
                raise AssertionError(f"{path}: invalid slab record {iz}")
            dtype = np.dtype(f"{endian}i{length // (n*n)}")
            ids = np.frombuffer(payload, dtype=dtype).reshape(n, n)
            expected = ix[None, :] * n * n + iy * n + iz
            if not np.array_equal(ids, expected):
                raise AssertionError(f"{path}: ID ordering differs in slab z={iz}")
        if handle.read(1):
            raise AssertionError(f"{path}: trailing bytes")
    print(f"  GenetIC GRAFIC IDs: exact Cartesian ordering 0..{n**3 - 1}")


def check_ramses_ids(output: Path, n: int, expected_ranks: int) -> None:
    expected = n**3
    seen = np.zeros(expected, dtype=np.bool_)
    total = 0
    files = find_part_files(output)
    if len(files) != expected_ranks:
        raise AssertionError(f"{output}: found {len(files)} rank files, expected {expected_ranks}")
    for path in files:
        ncpu, ndim, nstar_tot, ids = read_dmo_ids(path)
        if ncpu != expected_ranks or ndim != 3 or nstar_tot != 0:
            raise AssertionError(
                f"{path}: ncpu={ncpu}, ndim={ndim}, nstar_tot={nstar_tot}"
            )
        if np.any(ids < 0) or np.any(ids >= expected):
            raise AssertionError(f"{path}: ID outside 0..{expected - 1}")
        duplicate = seen[ids]
        if np.any(duplicate):
            raise AssertionError(f"{path}: duplicate ID {int(ids[duplicate][0])}")
        seen[ids] = True
        total += ids.size
    if total != expected or not np.all(seen):
        missing = int(np.flatnonzero(~seen)[0]) if not np.all(seen) else -1
        raise AssertionError(f"RAMSES ID permutation incomplete: N={total}, missing={missing}")
    info_files = list(output.glob("info_*.txt"))
    if len(info_files) != 1:
        raise AssertionError(f"{output}: expected one info file")
    info = info_files[0].read_text()
    match = re.search(r"^\s*aexp\s*=\s*([+\-0-9.EeDd]+)", info, re.MULTILINE)
    if match is None or abs(float(match.group(1).replace("D", "E")) - 0.02) > 1.0e-8:
        raise AssertionError(f"{info_files[0]}: expected aexp=0.02")
    print(f"  lagRamses IDs: exact permutation N={total} ranks={len(files)} aexp=0.02")


def check_amplitude_logs(root: Path) -> None:
    mono = (root / "monofonic/monofonic.log").read_text(errors="replace")
    genetic = (root / "genetic/genetic.log").read_text(errors="replace")
    if "CAMB_file: Using A_s=2.1064e-09" not in mono:
        raise AssertionError("monofonIC did not report the requested A_s branch")
    if "A_s=2.1064e-09" not in genetic and "A_s = 2.1064e-09" not in genetic:
        raise AssertionError("GenetIC did not report the requested A_s branch")
    print("  normalization: both IC codes report A_s=2.1064e-9")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workroot", type=Path)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--ranks", type=int, default=16)
    args = parser.parse_args()
    root = args.workroot.resolve()
    n = args.grid_size

    print("== VoidSim V4 level-9 parent verification ==")
    check_transfer(root / "camb/camb_transfer_z49.dat")
    check_white_noise(
        root / "monofonic/wn.h5", root / "genetic/wn_level0.npy", n
    )
    check_grafic_ids(
        root / f"genetic/v4_parent.grafic_{n}/ic_particle_ids", n
    )
    check_ramses_ids(root / "ramses_initial/output_00001", n, args.ranks)
    check_amplitude_logs(root)
    print("V4 LEVEL-9 PARENT GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
