#!/usr/bin/env python3
"""Rank production parent haloes and write boundary-safe GenetIC masks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent.parent / "v2_hop_id_file"
sys.path.insert(0, str(V2_DIR))

from hop_to_genetic_id import (  # noqa: E402
    find_part_files,
    inspect_hop_tag,
    read_dmo_ids,
)


@dataclass
class Candidate:
    halo_id: int
    particle_count: int
    hop_mass: float
    contamination: float
    eulerian_centre: tuple[float, float, float]
    mass_rank: int = 0
    lagrangian_start: tuple[float, float, float] | None = None
    lagrangian_width: tuple[float, float, float] | None = None
    lagrangian_centre: tuple[float, float, float] | None = None
    wraps_boundary: tuple[bool, bool, bool] | None = None
    recenter_shift: tuple[float, float, float] | None = None
    boundary_safe: bool = False
    selected: bool = False
    target_rank: int | None = None
    id_file: str | None = None


def parse_catalogue(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[int] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 7:
            raise ValueError(f"{path}:{line_number}: expected at least 7 columns")
        halo_id = int(fields[0])
        if halo_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate halo {halo_id}")
        seen.add(halo_id)
        candidates.append(
            Candidate(
                halo_id=halo_id,
                particle_count=int(fields[1]),
                hop_mass=float(fields[2]),
                contamination=float(fields[3]),
                eulerian_centre=tuple(float(value) for value in fields[4:7]),
            )
        )
    if not candidates:
        raise ValueError(f"{path}: no HOP groups")
    candidates.sort(key=lambda item: (-item.particle_count, item.halo_id))
    for rank, candidate in enumerate(candidates, start=1):
        candidate.mass_rank = rank
    return candidates


def periodic_axis_geometry(
    indices: np.ndarray, grid_size: int
) -> tuple[float, float, float, bool]:
    occupied = np.unique(np.asarray(indices, dtype=np.int64))
    if occupied.size == 0:
        raise ValueError("cannot measure an empty Lagrangian mask")
    if occupied[0] < 0 or occupied[-1] >= grid_size:
        raise ValueError("Lagrangian grid index lies outside the parent grid")
    wrapped = np.concatenate((occupied, occupied[:1] + grid_size))
    gaps = np.diff(wrapped)
    gap_index = int(np.argmax(gaps))
    start_cell = int(wrapped[gap_index + 1] % grid_size)
    width_cells = int(grid_size - gaps[gap_index] + 1)
    start = start_cell / grid_size
    width = width_cells / grid_size
    centre = ((start_cell + 0.5 * width_cells) / grid_size) % 1.0
    wraps = start_cell + width_cells > grid_size
    return start, width, centre, wraps


def attach_geometry(
    candidate: Candidate,
    ids: np.ndarray,
    grid_size: int,
    edge_buffer_cells: int,
) -> None:
    ix = ids // (grid_size * grid_size)
    iy = (ids // grid_size) % grid_size
    iz = ids % grid_size
    axes = [periodic_axis_geometry(axis, grid_size) for axis in (ix, iy, iz)]
    candidate.lagrangian_start = tuple(axis[0] for axis in axes)
    candidate.lagrangian_width = tuple(axis[1] for axis in axes)
    candidate.lagrangian_centre = tuple(axis[2] for axis in axes)
    candidate.wraps_boundary = tuple(axis[3] for axis in axes)
    candidate.recenter_shift = tuple(
        ((0.5 - centre + 0.5) % 1.0) - 0.5 for _, _, centre, _ in axes
    )
    buffer_fraction = edge_buffer_cells / grid_size
    candidate.boundary_safe = all(
        width + 2.0 * buffer_fraction <= 1.0 for _, width, _, _ in axes
    )


def periodic_distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    delta = np.abs(np.asarray(left) - np.asarray(right))
    delta = np.minimum(delta, 1.0 - delta)
    return float(np.sqrt(np.dot(delta, delta)))


def choose_targets(
    candidates: list[Candidate],
    target_count: int,
    min_particles: int,
    max_lagrangian_width: float,
    min_centre_separation: float,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in candidates:
        if candidate.particle_count < min_particles or not candidate.boundary_safe:
            continue
        assert candidate.lagrangian_width is not None
        if max(candidate.lagrangian_width) > max_lagrangian_width:
            continue
        if any(
            periodic_distance(candidate.eulerian_centre, other.eulerian_centre)
            < min_centre_separation
            for other in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == target_count:
            break
    return selected


def collect_candidate_ids(
    output_dir: Path,
    tag_path: Path,
    candidates: list[Candidate],
    grid_size: int,
) -> list[np.ndarray]:
    layout = inspect_hop_tag(tag_path)
    expected = grid_size**3
    if layout.npart != expected:
        raise ValueError(
            f"HOP contains {layout.npart} particles, expected {grid_size}^3={expected}"
        )
    tags = np.memmap(
        tag_path,
        mode="r",
        dtype=np.dtype(f"{layout.endian}i4"),
        offset=layout.payload_offset,
        shape=(layout.npart,),
    )
    lookup = np.full(layout.ngroups, -1, dtype=np.int32)
    for index, candidate in enumerate(candidates):
        if not 1 <= candidate.halo_id <= layout.ngroups:
            raise ValueError(f"halo {candidate.halo_id} lies outside the HOP tag range")
        lookup[candidate.halo_id - 1] = index

    chunks: list[list[np.ndarray]] = [[] for _ in candidates]
    seen = np.zeros(expected, dtype=np.bool_)
    tag_offset = 0
    part_files = find_part_files(output_dir)
    declared_ncpu: int | None = None
    for path in part_files:
        ncpu, ndim, nstar_tot, ids = read_dmo_ids(path)
        if declared_ncpu is None:
            declared_ncpu = ncpu
        if (ncpu, ndim, nstar_tot) != (declared_ncpu, 3, 0):
            raise ValueError(
                f"{path}: ncpu={ncpu}, ndim={ndim}, nstar_tot={nstar_tot}"
            )
        if tag_offset + ids.size > layout.npart:
            raise ValueError("RAMSES particle files contain more entries than HOP")
        if np.any(ids < 0) or np.any(ids >= expected):
            raise ValueError(f"{path}: particle ID lies outside 0..{expected - 1}")
        if np.any(seen[ids]):
            raise ValueError(f"{path}: duplicate parent particle ID")
        seen[ids] = True

        local_tags = np.asarray(tags[tag_offset : tag_offset + ids.size])
        mapped = np.full(local_tags.shape, -1, dtype=np.int32)
        valid = (local_tags >= 0) & (local_tags < layout.ngroups)
        mapped[valid] = lookup[local_tags[valid]]
        for index in np.unique(mapped[mapped >= 0]):
            chunks[int(index)].append(ids[mapped == index])
        tag_offset += ids.size

    if declared_ncpu != len(part_files):
        raise ValueError(
            f"RAMSES declares {declared_ncpu} rank files, found {len(part_files)}"
        )
    if tag_offset != layout.npart or not np.all(seen):
        raise ValueError("RAMSES IDs do not form the complete HOP particle ordering")

    memberships: list[np.ndarray] = []
    for candidate, candidate_chunks in zip(candidates, chunks, strict=True):
        if not candidate_chunks:
            raise ValueError(f"halo {candidate.halo_id} has no matched particle IDs")
        ids = np.sort(np.concatenate(candidate_chunks))
        if ids.size != candidate.particle_count:
            raise ValueError(
                f"halo {candidate.halo_id}: HOP catalogue has "
                f"{candidate.particle_count} particles, matched {ids.size}"
            )
        memberships.append(ids)
    return memberships


def write_reports(
    output_dir: Path, candidates: list[Candidate], parameters: dict[str, object]
) -> None:
    report = {
        "parameters": parameters,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (output_dir / "parent_target_candidates.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (output_dir / "parent_target_candidates.tsv").open("w") as handle:
        handle.write(
            "mass_rank\thalo_id\tnpart\thop_mass\tboundary_safe\tselected\t"
            "xc\tyc\tzc\tlag_width_x\tlag_width_y\tlag_width_z\t"
            "recenter_dx\trecenter_dy\trecenter_dz\n"
        )
        for candidate in candidates:
            assert candidate.lagrangian_width is not None
            handle.write(
                f"{candidate.mass_rank}\t{candidate.halo_id}\t"
                f"{candidate.particle_count}\t{candidate.hop_mass:.9e}\t"
                f"{int(candidate.boundary_safe)}\t{int(candidate.selected)}\t"
                + "\t".join(f"{value:.9e}" for value in candidate.eulerian_centre)
                + "\t"
                + "\t".join(f"{value:.9e}" for value in candidate.lagrangian_width)
                + "\t"
                + "\t".join(f"{value:.9e}" for value in candidate.recenter_shift)
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ramses_output", type=Path)
    parser.add_argument("hop_tag", type=Path)
    parser.add_argument("hop_catalogue", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--target-count", type=int, default=3)
    parser.add_argument("--min-particles", type=int, default=3000)
    parser.add_argument("--max-lagrangian-width", type=float, default=0.5)
    parser.add_argument("--edge-buffer-cells", type=int, default=2)
    parser.add_argument("--min-centre-separation", type=float, default=0.1)
    args = parser.parse_args()

    catalogue = parse_catalogue(args.hop_catalogue.resolve())
    candidates = catalogue[: args.candidate_count]
    memberships = collect_candidate_ids(
        args.ramses_output.resolve(),
        args.hop_tag.resolve(),
        candidates,
        args.grid_size,
    )
    for candidate, ids in zip(candidates, memberships, strict=True):
        attach_geometry(candidate, ids, args.grid_size, args.edge_buffer_cells)

    selected = choose_targets(
        candidates,
        args.target_count,
        args.min_particles,
        args.max_lagrangian_width,
        args.min_centre_separation,
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    membership_by_halo = {
        candidate.halo_id: ids
        for candidate, ids in zip(candidates, memberships, strict=True)
    }
    for target_rank, candidate in enumerate(selected, start=1):
        candidate.selected = True
        candidate.target_rank = target_rank
        path = output_dir / f"target_{target_rank:02d}_halo_{candidate.halo_id:06d}.id"
        np.savetxt(path, membership_by_halo[candidate.halo_id], fmt="%d")
        candidate.id_file = path.name

    parameters = {
        "grid_size": args.grid_size,
        "candidate_count": args.candidate_count,
        "target_count": args.target_count,
        "min_particles": args.min_particles,
        "max_lagrangian_width": args.max_lagrangian_width,
        "edge_buffer_cells": args.edge_buffer_cells,
        "min_centre_separation": args.min_centre_separation,
    }
    write_reports(output_dir, candidates, parameters)
    if len(selected) != args.target_count:
        raise RuntimeError(
            f"only {len(selected)} of {args.target_count} requested targets pass "
            "the predeclared compactness, boundary, and separation cuts"
        )
    print(
        "selected halo IDs: "
        + ", ".join(
            f"{candidate.halo_id} (mass rank {candidate.mass_rank}, "
            f"N={candidate.particle_count})"
            for candidate in selected
        )
    )
    print(f"wrote target masks and diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
