#!/usr/bin/env python3
"""Grow a GenetIC Lagrangian id_file by periodic parent-grid shells."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np


def load_id_file(path: Path, grid_size: int) -> np.ndarray:
    if grid_size <= 0:
        raise ValueError("grid size must be positive")
    ids = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if ids.size == 0:
        raise ValueError(f"{path}: id_file is empty")
    count = grid_size**3
    if np.any(ids < 0) or np.any(ids >= count):
        bad = int(ids[(ids < 0) | (ids >= count)][0])
        raise ValueError(
            f"{path}: particle ID {bad} lies outside 0..{count - 1}"
        )
    ordered = np.sort(ids)
    if np.any(ordered[1:] == ordered[:-1]):
        duplicate = int(ordered[:-1][ordered[1:] == ordered[:-1]][0])
        raise ValueError(f"{path}: duplicate particle ID {duplicate}")
    return ordered


def _dilate_axis(ids: np.ndarray, grid_size: int, axis: int) -> np.ndarray:
    plane = grid_size * grid_size
    if axis == 0:
        coordinate = ids // plane
        remainder = ids % plane
        lower = ((coordinate - 1) % grid_size) * plane + remainder
        upper = ((coordinate + 1) % grid_size) * plane + remainder
    elif axis == 1:
        x = ids // plane
        yz = ids % plane
        coordinate = yz // grid_size
        z = yz % grid_size
        lower = x * plane + ((coordinate - 1) % grid_size) * grid_size + z
        upper = x * plane + ((coordinate + 1) % grid_size) * grid_size + z
    elif axis == 2:
        coordinate = ids % grid_size
        base = ids - coordinate
        lower = base + ((coordinate - 1) % grid_size)
        upper = base + ((coordinate + 1) % grid_size)
    else:
        raise ValueError(f"invalid Cartesian axis {axis}")
    return np.unique(np.concatenate((ids, lower, upper)))


def grow_ids(ids: np.ndarray, grid_size: int, shells: int = 1) -> np.ndarray:
    """Return the periodic Chebyshev dilation of a sorted parent-grid ID set."""

    if shells < 0:
        raise ValueError("shell count must be non-negative")
    grown = np.asarray(ids, dtype=np.int64)
    for _ in range(shells):
        for axis in range(3):
            grown = _dilate_axis(grown, grid_size, axis)
    return grown


def write_id_file(path: Path, ids: np.ndarray, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path}: output exists; pass --force to replace it")
    if not path.parent.exists():
        raise FileNotFoundError(f"{path.parent}: output directory not found")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.savetxt(handle, ids, fmt="%d")
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def grow_file(
    input_path: Path,
    output_path: Path,
    grid_size: int,
    shells: int,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    original = load_id_file(input_path, grid_size)
    grown = grow_ids(original, grid_size, shells)
    write_id_file(output_path, grown, force=force)
    return original, grown


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Grow a zero-based GenetIC id_file by periodic Chebyshev shells "
            "on the parent particle grid."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--shells", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    original, grown = grow_file(
        args.input.resolve(),
        args.output.resolve(),
        args.grid_size,
        args.shells,
        force=args.force,
    )
    print(
        f"[ok] periodic buffer: {original.size} -> {grown.size} IDs "
        f"after {args.shells} shell(s)"
    )
    print(f"[ok] wrote GenetIC id_file {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
