#!/usr/bin/env python3
"""Verify the final snapshot of the VoidSim V4 fixed-grid DMO parent."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    from hop_to_genetic_id import find_part_files, read_dmo_ids
except ModuleNotFoundError:
    v2_directory = Path(__file__).resolve().parent.parent.parent / "v2_hop_id_file"
    sys.path.insert(0, str(v2_directory))
    from hop_to_genetic_id import find_part_files, read_dmo_ids


def info_value(text: str, key: str) -> float:
    match = re.search(
        rf"^\s*{re.escape(key)}\s*=\s*([+\-0-9.EeDd]+)", text, re.MULTILINE
    )
    if match is None:
        raise AssertionError(f"info file does not contain {key}")
    return float(match.group(1).replace("D", "E"))


def check_outputs(
    run_dir: Path, expected_ranks: int, expected_outputs: int = 6
) -> Path:
    outputs = sorted(path for path in run_dir.glob("output_[0-9]*") if path.is_dir())
    if len(outputs) != expected_outputs:
        raise AssertionError(
            f"{run_dir}: expected {expected_outputs} snapshots, found {len(outputs)}"
        )

    scale_factors: list[float] = []
    for output in outputs:
        suffix = output.name.removeprefix("output_")
        info_path = output / f"info_{suffix}.txt"
        if not info_path.is_file():
            raise AssertionError(f"{output}: missing {info_path.name}")
        info = info_path.read_text(errors="replace")
        ncpu = int(round(info_value(info, "ncpu")))
        if ncpu != expected_ranks:
            raise AssertionError(f"{info_path}: ncpu={ncpu}, expected {expected_ranks}")
        scale_factors.append(info_value(info, "aexp"))

    if np.any(np.diff(scale_factors) <= 0.0):
        raise AssertionError(f"snapshot scale factors are not increasing: {scale_factors}")
    if not 0.0199 <= scale_factors[0] <= 0.0201:
        raise AssertionError(
            f"initial snapshot scale factor is {scale_factors[0]}, expected a=0.02"
        )
    if not 0.999 <= scale_factors[-1] <= 1.02:
        raise AssertionError(f"final scale factor is {scale_factors[-1]}, expected a=1")
    print("  snapshots: " + ", ".join(f"a={value:.6f}" for value in scale_factors))
    return outputs[-1]


def check_ids(output: Path, grid_size: int, expected_ranks: int) -> None:
    expected = grid_size**3
    seen = np.zeros(expected, dtype=np.bool_)
    files = find_part_files(output)
    if len(files) != expected_ranks:
        raise AssertionError(f"{output}: found {len(files)} particle files")

    total = 0
    for path in files:
        ncpu, ndim, nstar_tot, ids = read_dmo_ids(path)
        if (ncpu, ndim, nstar_tot) != (expected_ranks, 3, 0):
            raise AssertionError(
                f"{path}: ncpu={ncpu}, ndim={ndim}, nstar_tot={nstar_tot}"
            )
        outside = (ids < 0) | (ids >= expected)
        if np.any(outside):
            raise AssertionError(f"{path}: particle ID {int(ids[outside][0])} is invalid")
        duplicate = seen[ids]
        if np.any(duplicate):
            raise AssertionError(f"{path}: duplicate particle ID {int(ids[duplicate][0])}")
        seen[ids] = True
        total += ids.size

    if total != expected or not np.all(seen):
        missing = int(np.flatnonzero(~seen)[0]) if not np.all(seen) else -1
        raise AssertionError(f"incomplete ID permutation: N={total}, missing={missing}")
    print(f"  particle IDs: exact permutation 0..{expected - 1} over {len(files)} ranks")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--outputs", type=int, default=6)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    log = (run_dir / "ramses.log").read_text(errors="replace")
    if "Run completed" not in log:
        raise AssertionError(f"{run_dir / 'ramses.log'}: completion marker is absent")

    print("== VoidSim V4 z=0 DMO verification ==")
    final_output = check_outputs(run_dir, args.ranks, args.outputs)
    check_ids(final_output, args.grid_size, args.ranks)
    print("V4 Z=0 DMO PARENT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
