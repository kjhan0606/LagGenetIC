#!/usr/bin/env python3
"""Verify the compact rank-726 level-14 DMO evolution to zero redshift."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np

try:
    from hop_to_genetic_id import find_part_files, read_dmo_ids
except ModuleNotFoundError:
    v2_directory = Path(__file__).resolve().parent.parent.parent / "v2_hop_id_file"
    sys.path.insert(0, str(v2_directory))
    from hop_to_genetic_id import find_part_files, read_dmo_ids


DEFAULT_ID_CAPACITY = 1_363_148_800
DEFAULT_PARTICLES = 260_317_198


def info_value(text: str, key: str) -> float:
    match = re.search(
        rf"^\s*{re.escape(key)}\s*=\s*([+\-0-9.EeDd]+)", text, re.MULTILINE
    )
    if match is None:
        raise AssertionError(f"info file does not contain {key}")
    return float(match.group(1).replace("D", "E").replace("d", "e"))


def check_outputs(
    run_dir: Path,
    expected_ranks: int,
    expected_outputs: int = 6,
    first_index: int = 2,
) -> tuple[Path, Path]:
    outputs = sorted(path for path in run_dir.glob("output_[0-9]*") if path.is_dir())
    expected_names = [
        f"output_{index:05d}"
        for index in range(first_index, first_index + expected_outputs)
    ]
    if [path.name for path in outputs] != expected_names:
        raise AssertionError(
            f"{run_dir}: expected snapshots {expected_names}, found "
            f"{[path.name for path in outputs]}"
        )

    scale_factors: list[float] = []
    for output in outputs:
        suffix = output.name.removeprefix("output_")
        if not (output / "COMPLETE").is_file():
            raise AssertionError(f"{output}: COMPLETE marker is absent")
        info_path = output / f"info_{suffix}.txt"
        if not info_path.is_file():
            raise AssertionError(f"{output}: missing {info_path.name}")
        info = info_path.read_text(errors="replace")
        ncpu = int(round(info_value(info, "ncpu")))
        levelmin = int(round(info_value(info, "levelmin")))
        levelmax = int(round(info_value(info, "levelmax")))
        if ncpu != expected_ranks:
            raise AssertionError(f"{info_path}: ncpu={ncpu}, expected {expected_ranks}")
        if (levelmin, levelmax) != (9, 14):
            raise AssertionError(
                f"{info_path}: level range {levelmin}..{levelmax}, expected 9..14"
            )
        part_files = find_part_files(output)
        if len(part_files) != expected_ranks:
            raise AssertionError(
                f"{output}: found {len(part_files)} particle files, "
                f"expected {expected_ranks}"
            )
        scale_factors.append(info_value(info, "aexp"))

    if np.any(np.diff(scale_factors) <= 0.0):
        raise AssertionError(f"snapshot scale factors are not increasing: {scale_factors}")
    if not 0.0220 <= scale_factors[0] <= 0.0223:
        raise AssertionError(
            f"restart snapshot scale factor is {scale_factors[0]}, "
            "expected a=0.0221607"
        )
    if not 0.999 <= scale_factors[-1] <= 1.02:
        raise AssertionError(f"final scale factor is {scale_factors[-1]}, expected a=1")
    print("  snapshots: " + ", ".join(f"a={value:.6f}" for value in scale_factors))
    return outputs[0], outputs[-1]


def check_restart_checkpoint(
    handoff_run: Path,
    expected_ranks: int,
    id_capacity: int,
    expected_particles: int,
) -> None:
    log = (handoff_run / "ramses.log").read_text(errors="replace")
    if "Run completed" not in log:
        raise AssertionError(f"{handoff_run / 'ramses.log'}: completion marker is absent")
    if "Problem in check_tree" in log:
        raise AssertionError(f"{handoff_run / 'ramses.log'}: particle-tree failure is present")

    initial = handoff_run / "output_00001"
    restart = handoff_run / "output_00002"
    scale_factors: list[float] = []
    coarse_steps: list[int] = []
    for index, output in ((1, initial), (2, restart)):
        if not (output / "COMPLETE").is_file():
            raise AssertionError(f"{output}: COMPLETE marker is absent")
        info_path = output / f"info_{index:05d}.txt"
        info = info_path.read_text(errors="replace")
        if int(round(info_value(info, "ncpu"))) != expected_ranks:
            raise AssertionError(f"{info_path}: unexpected MPI rank count")
        if (
            int(round(info_value(info, "levelmin"))),
            int(round(info_value(info, "levelmax"))),
        ) != (9, 14):
            raise AssertionError(f"{info_path}: expected the level-9 through 14 mesh")
        if len(find_part_files(output)) != expected_ranks:
            raise AssertionError(f"{output}: unexpected particle-file count")
        scale_factors.append(info_value(info, "aexp"))
        coarse_steps.append(int(round(info_value(info, "nstep_coarse"))))

    if coarse_steps != [0, 1]:
        raise AssertionError(f"handoff coarse-step sequence is {coarse_steps}, expected [0, 1]")
    if not 0.0199 <= scale_factors[0] <= 0.0201:
        raise AssertionError(f"pre-step handoff has a={scale_factors[0]}, expected 0.02")
    if not 0.0220 <= scale_factors[1] <= 0.0223:
        raise AssertionError(
            f"post-step restart has a={scale_factors[1]}, expected 0.0221607"
        )
    check_id_conservation(
        initial,
        restart,
        expected_ranks,
        id_capacity,
        expected_particles,
    )
    print(
        f"  restart checkpoint: nstep_coarse=1, a={scale_factors[1]:.9f}, "
        f"ranks={expected_ranks}"
    )


def _check_local_ids(ids: np.ndarray, path: Path, id_capacity: int) -> None:
    outside = (ids < 0) | (ids >= id_capacity)
    if np.any(outside):
        raise AssertionError(
            f"{path}: particle ID {int(ids[outside][0])} lies outside "
            f"0..{id_capacity - 1}"
        )
    if ids.size > 1:
        ordered = np.sort(ids)
        duplicate = np.flatnonzero(ordered[1:] == ordered[:-1])
        if duplicate.size:
            raise AssertionError(
                f"{path}: duplicate particle ID {int(ordered[duplicate[0]])}"
            )


def check_id_conservation(
    initial_output: Path,
    final_output: Path,
    expected_ranks: int,
    id_capacity: int,
    expected_particles: int,
) -> None:
    if id_capacity <= 0 or expected_particles <= 0:
        raise ValueError("ID capacity and expected particle count must be positive")
    if expected_particles > id_capacity:
        raise ValueError("expected particle count exceeds the ID capacity")

    state = np.zeros(id_capacity, dtype=np.uint8)
    initial_files = find_part_files(initial_output)
    final_files = find_part_files(final_output)
    if len(initial_files) != expected_ranks or len(final_files) != expected_ranks:
        raise AssertionError(
            f"particle-file count changed: initial={len(initial_files)}, "
            f"final={len(final_files)}, expected={expected_ranks}"
        )

    initial_count = 0
    for rank, path in enumerate(initial_files, start=1):
        ncpu, ndim, nstar_tot, ids = read_dmo_ids(path)
        if (ncpu, ndim, nstar_tot) != (expected_ranks, 3, 0):
            raise AssertionError(
                f"{path}: ncpu={ncpu}, ndim={ndim}, nstar_tot={nstar_tot}"
            )
        _check_local_ids(ids, path, id_capacity)
        occupied = state[ids]
        if np.any(occupied):
            index = int(np.flatnonzero(occupied)[0])
            raise AssertionError(
                f"{path}: initial particle ID {int(ids[index])} is globally duplicated"
            )
        state[ids] = 1
        initial_count += ids.size
        if rank % 8 == 0 or rank == expected_ranks:
            print(f"  initial IDs: ranks {rank}/{expected_ranks}", flush=True)

    if initial_count != expected_particles:
        raise AssertionError(
            f"initial snapshot contains {initial_count} particles, "
            f"expected {expected_particles}"
        )

    final_count = 0
    for rank, path in enumerate(final_files, start=1):
        ncpu, ndim, nstar_tot, ids = read_dmo_ids(path)
        if (ncpu, ndim, nstar_tot) != (expected_ranks, 3, 0):
            raise AssertionError(
                f"{path}: ncpu={ncpu}, ndim={ndim}, nstar_tot={nstar_tot}"
            )
        _check_local_ids(ids, path, id_capacity)
        membership = state[ids]
        absent = np.flatnonzero(membership == 0)
        if absent.size:
            raise AssertionError(
                f"{path}: final particle ID {int(ids[absent[0]])} was absent initially"
            )
        repeated = np.flatnonzero(membership == 2)
        if repeated.size:
            raise AssertionError(
                f"{path}: final particle ID {int(ids[repeated[0]])} is globally duplicated"
            )
        state[ids] = 2
        final_count += ids.size
        if rank % 8 == 0 or rank == expected_ranks:
            print(f"  final IDs: ranks {rank}/{expected_ranks}", flush=True)

    if final_count != initial_count:
        raise AssertionError(
            f"particle count changed from {initial_count} to {final_count}"
        )
    print(
        f"  particle IDs: exact initial/final set equality for {final_count} "
        f"particles over {expected_ranks} ranks"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--outputs", type=int, default=6)
    parser.add_argument("--id-capacity", type=int, default=DEFAULT_ID_CAPACITY)
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--handoff-checkpoint", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    if args.handoff_checkpoint:
        print("== VoidSim compact rank-726 restart-checkpoint verification ==")
        check_restart_checkpoint(
            run_dir,
            args.ranks,
            args.id_capacity,
            args.particles,
        )
        print("V4 COMPACT RANK-726 LEVEL-14 RESTART CHECKPOINT PASSED")
        return 0

    log = (run_dir / "ramses.log").read_text(errors="replace")
    if "Run completed" not in log:
        raise AssertionError(f"{run_dir / 'ramses.log'}: completion marker is absent")
    if "Increase ngridmax" in log or "Maximum number of particles incorrect" in log:
        raise AssertionError(f"{run_dir / 'ramses.log'}: capacity failure is present")

    print("== VoidSim compact rank-726 level-14 z=0 DMO verification ==")
    initial_output, final_output = check_outputs(
        run_dir, args.ranks, args.outputs, first_index=2
    )
    check_id_conservation(
        initial_output,
        final_output,
        args.ranks,
        args.id_capacity,
        args.particles,
    )
    print("V4 COMPACT RANK-726 LEVEL-14 Z=0 DMO PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
