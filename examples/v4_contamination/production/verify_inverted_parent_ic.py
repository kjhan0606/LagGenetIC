#!/usr/bin/env python3
"""Verify the exact V4 normal/inverted level-9 GRAFIC pair."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
import struct

import numpy as np


def read_marker(handle, endian: str) -> int:
    payload = handle.read(4)
    if len(payload) != 4:
        raise EOFError("unexpected end of GRAFIC file")
    return struct.unpack(f"{endian}i", payload)[0]


def grafic_slabs(path: Path) -> tuple[tuple[int, int, int], Iterator[np.ndarray]]:
    handle = path.open("rb")
    raw = handle.read(4)
    if len(raw) != 4:
        handle.close()
        raise AssertionError(f"{path}: missing GRAFIC header")
    little = struct.unpack("<i", raw)[0]
    big = struct.unpack(">i", raw)[0]
    endian = "<" if little == 44 else ">" if big == 44 else None
    if endian is None:
        handle.close()
        raise AssertionError(f"{path}: unsupported GRAFIC record marker")
    header = handle.read(44)
    if len(header) != 44 or read_marker(handle, endian) != 44:
        handle.close()
        raise AssertionError(f"{path}: invalid GRAFIC header")
    dimensions = struct.unpack(f"{endian}iii", header[:12])

    def iterate() -> Iterator[np.ndarray]:
        try:
            n1, n2, n3 = dimensions
            for slab in range(n3):
                length = read_marker(handle, endian)
                payload = handle.read(length)
                if read_marker(handle, endian) != length:
                    raise AssertionError(f"{path}: broken slab {slab}")
                if length not in (4 * n1 * n2, 8 * n1 * n2):
                    raise AssertionError(f"{path}: invalid slab length {length}")
                yield np.frombuffer(
                    payload, dtype=np.dtype(f"{endian}f{length // (n1 * n2)}")
                )
            if handle.read(1):
                raise AssertionError(f"{path}: trailing data")
        finally:
            handle.close()

    return dimensions, iterate()


def check_density_pair(normal: Path, inverted: Path, grid_size: int) -> None:
    normal_dims, normal_slabs = grafic_slabs(normal)
    inverted_dims, inverted_slabs = grafic_slabs(inverted)
    expected = (grid_size, grid_size, grid_size)
    if normal_dims != expected or inverted_dims != expected:
        raise AssertionError(
            f"density dimensions differ: {normal_dims}, {inverted_dims}, expected {expected}"
        )
    maximum_residual = 0.0
    count = 0
    for slab, (left, right) in enumerate(
        zip(normal_slabs, inverted_slabs, strict=True)
    ):
        if not np.array_equal(right, -left):
            residual = float(np.max(np.abs(right + left)))
            raise AssertionError(f"density sign mismatch in slab {slab}: {residual}")
        maximum_residual = max(maximum_residual, float(np.max(np.abs(right + left))))
        count += left.size
    if count != grid_size**3:
        raise AssertionError(f"checked {count} cells, expected {grid_size**3}")
    print(f"  density: exact delta_inverted=-delta_normal over {count} cells")
    print(f"  maximum sign residual: {maximum_residual:.1f}")


def files_identical(left: Path, right: Path, chunk_size: int = 8 << 20) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            left_chunk = first.read(chunk_size)
            right_chunk = second.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("normal", type=Path)
    parser.add_argument("inverted", type=Path)
    parser.add_argument("--grid-size", type=int, default=512)
    args = parser.parse_args()
    normal = args.normal.resolve()
    inverted = args.inverted.resolve()

    print("== VoidSim V4 inverted parent IC verification ==")
    check_density_pair(
        normal / "ic_deltab", inverted / "ic_deltab", args.grid_size
    )
    if not files_identical(
        normal / "ic_particle_ids", inverted / "ic_particle_ids"
    ):
        raise AssertionError("normal/inverted GRAFIC particle IDs differ")
    print("  particle IDs: bit-identical")
    print("V4 INVERTED PARENT IC PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
