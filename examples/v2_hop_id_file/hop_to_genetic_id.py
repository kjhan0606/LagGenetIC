#!/usr/bin/env python3
"""Convert a RAMSES HOP group selection into a GenetIC id_file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class HopTagLayout:
    npart: int
    ngroups: int
    payload_offset: int
    endian: str
    encoding: str


class FortranRecordReader:
    """Read sequential Fortran records with four-byte record markers."""

    def __init__(self, path: Path):
        self.path = path
        self.handle: BinaryIO = path.open("rb")
        marker = self.handle.read(4)
        self.handle.seek(0)
        if len(marker) != 4:
            raise ValueError(f"{path}: file is too short for a Fortran record")
        little = struct.unpack("<i", marker)[0]
        big = struct.unpack(">i", marker)[0]
        if little == 4:
            self.endian = "<"
        elif big == 4:
            self.endian = ">"
        else:
            raise ValueError(
                f"{path}: unsupported Fortran record marker {marker.hex()}"
            )

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "FortranRecordReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _marker(self) -> int:
        raw = self.handle.read(4)
        if len(raw) != 4:
            raise EOFError(f"{self.path}: unexpected end of file")
        return struct.unpack(f"{self.endian}i", raw)[0]

    def read(self) -> bytes:
        length = self._marker()
        if length < 0:
            raise ValueError(f"{self.path}: negative record length {length}")
        payload = self.handle.read(length)
        if len(payload) != length:
            raise EOFError(f"{self.path}: truncated record payload")
        trailer = self._marker()
        if trailer != length:
            raise ValueError(
                f"{self.path}: record marker mismatch {length} != {trailer}"
            )
        return payload

    def skip(self) -> None:
        length = self._marker()
        if length < 0:
            raise ValueError(f"{self.path}: negative record length {length}")
        self.handle.seek(length, os.SEEK_CUR)
        trailer = self._marker()
        if trailer != length:
            raise ValueError(
                f"{self.path}: record marker mismatch {length} != {trailer}"
            )


def _scalar(payload: bytes, endian: str, name: str) -> int:
    if len(payload) == 4:
        return int(struct.unpack(f"{endian}i", payload)[0])
    if len(payload) == 8:
        return int(struct.unpack(f"{endian}q", payload)[0])
    raise ValueError(f"{name}: expected a four- or eight-byte integer record")


def inspect_hop_tag(path: Path) -> HopTagLayout:
    """Locate the contiguous int32 tag payload in raw or Fortran HOP output."""

    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(20)
    if len(prefix) < 8:
        raise ValueError(f"{path}: HOP tag file is too short")

    for endian in ("<", ">"):
        if len(prefix) >= 20 and struct.unpack(f"{endian}i", prefix[:4])[0] == 8:
            npart, ngroups = struct.unpack(f"{endian}ii", prefix[4:12])
            trailer = struct.unpack(f"{endian}i", prefix[12:16])[0]
            second = struct.unpack(f"{endian}i", prefix[16:20])[0]
            expected = 24 + 4 * npart
            if (
                trailer == 8
                and npart > 0
                and ngroups > 0
                and second == 4 * npart
                and size == expected
            ):
                with path.open("rb") as handle:
                    handle.seek(20 + 4 * npart)
                    final_marker = struct.unpack(
                        f"{endian}i", handle.read(4)
                    )[0]
                if final_marker == 4 * npart:
                    return HopTagLayout(
                        npart, ngroups, 20, endian, "fortran-sequential"
                    )

        npart, ngroups = struct.unpack(f"{endian}ii", prefix[:8])
        if npart > 0 and ngroups > 0 and size == 8 + 4 * npart:
            return HopTagLayout(npart, ngroups, 8, endian, "raw")

    raise ValueError(
        f"{path}: could not identify a raw or Fortran-sequential HOP tag layout"
    )


def _part_rank(path: Path) -> int:
    match = re.search(r"\.out(\d+)$", path.name)
    if match is None:
        raise ValueError(f"{path}: RAMSES particle filename lacks a rank suffix")
    return int(match.group(1))


def find_part_files(output_dir: Path) -> list[Path]:
    files = sorted(output_dir.rglob("part_*.out*"), key=_part_rank)
    if not files:
        raise FileNotFoundError(f"{output_dir}: no part_*.out* files found")
    ranks = [_part_rank(path) for path in files]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{output_dir}: duplicate RAMSES particle rank suffixes")
    return files


def read_dmo_ids(path: Path) -> tuple[int, int, int, np.ndarray]:
    """Return ncpu, ndim, nstar_tot, and IDs from one DMO RAMSES part file."""

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
        for _ in range(2 * ndim + 1):
            records.skip()
        payload = records.read()

    if ndim != 3:
        raise ValueError(f"{path}: expected ndim=3, found {ndim}")
    if npart < 0:
        raise ValueError(f"{path}: negative particle count {npart}")
    if len(payload) == 8 * npart:
        dtype = np.dtype(f"{endian}i8")
    elif len(payload) == 4 * npart:
        dtype = np.dtype(f"{endian}i4")
    else:
        raise ValueError(
            f"{path}: identity record has {len(payload)} bytes for {npart} particles"
        )
    ids = np.frombuffer(payload, dtype=dtype).astype(np.int64, copy=True)
    return ncpu, ndim, nstar_tot, ids


def convert(
    output_dir: Path,
    tag_path: Path,
    output_path: Path,
    halo_ids: list[int],
    grid_size: int,
    force: bool = False,
) -> np.ndarray:
    if grid_size <= 0:
        raise ValueError("grid size must be positive")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path}: output exists; pass --force to replace it"
        )
    if not output_path.parent.exists():
        raise FileNotFoundError(f"{output_path.parent}: output directory not found")

    layout = inspect_hop_tag(tag_path)
    requested = sorted(set(halo_ids))
    if not requested:
        raise ValueError("at least one halo ID is required")
    if requested[0] < 1 or requested[-1] > layout.ngroups:
        raise ValueError(
            f"halo IDs must lie in the HOP catalogue range 1..{layout.ngroups}"
        )

    expected_particles = grid_size**3
    if layout.npart != expected_particles:
        raise ValueError(
            f"HOP contains {layout.npart} particles but a {grid_size}^3 "
            f"parent requires {expected_particles}"
        )

    tags = np.memmap(
        tag_path,
        mode="r",
        dtype=np.dtype(f"{layout.endian}i4"),
        offset=layout.payload_offset,
        shape=(layout.npart,),
    )
    tag_min = int(tags.min())
    tag_max = int(tags.max())
    if tag_min < -1 or tag_max >= layout.ngroups:
        raise ValueError(
            f"{tag_path}: tag range {tag_min}..{tag_max} is inconsistent "
            f"with {layout.ngroups} groups"
        )

    target_tags = np.asarray([halo_id - 1 for halo_id in requested], dtype=np.int64)
    seen = np.zeros(expected_particles, dtype=np.bool_)
    selected_chunks: list[np.ndarray] = []
    tag_offset = 0
    part_files = find_part_files(output_dir)
    declared_ncpu: int | None = None

    for path in part_files:
        ncpu, _, nstar_tot, ids = read_dmo_ids(path)
        if declared_ncpu is None:
            declared_ncpu = ncpu
        elif ncpu != declared_ncpu:
            raise ValueError(f"{path}: inconsistent ncpu={ncpu}, expected {declared_ncpu}")
        if nstar_tot != 0:
            raise ValueError(
                f"{path}: nstar_tot={nstar_tot}; the bundled HOP ordering is "
                "supported only for the dark-matter-only parent run"
            )
        if tag_offset + ids.size > layout.npart:
            raise ValueError("RAMSES particle files contain more entries than the HOP tag")
        if np.any(ids < 0) or np.any(ids >= expected_particles):
            bad = ids[(ids < 0) | (ids >= expected_particles)][0]
            raise ValueError(
                f"{path}: particle ID {int(bad)} lies outside 0.."
                f"{expected_particles - 1}"
            )
        if np.any(seen[ids]):
            duplicate = int(ids[seen[ids]][0])
            raise ValueError(f"{path}: duplicate particle ID {duplicate}")
        seen[ids] = True

        local_tags = np.asarray(tags[tag_offset : tag_offset + ids.size])
        mask = np.isin(local_tags, target_tags)
        if np.any(mask):
            selected_chunks.append(ids[mask])
        tag_offset += ids.size

    if declared_ncpu is not None and declared_ncpu != len(part_files):
        raise ValueError(
            f"RAMSES header declares {declared_ncpu} files but found {len(part_files)}"
        )
    if tag_offset != layout.npart:
        raise ValueError(
            f"RAMSES particle files contain {tag_offset} entries but HOP has "
            f"{layout.npart} tags"
        )
    if not np.all(seen):
        missing = int(np.flatnonzero(~seen)[0])
        raise ValueError(f"RAMSES parent is missing Lagrangian particle ID {missing}")
    if not selected_chunks:
        raise ValueError(f"no particles belong to requested halo IDs {requested}")

    selected = np.sort(np.concatenate(selected_chunks))
    if np.any(selected[1:] == selected[:-1]):
        raise ValueError("selected halo membership contains duplicate particle IDs")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.savetxt(handle, selected, fmt="%d")
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)

    print(
        f"[ok] HOP {layout.encoding}: {layout.npart} particles, "
        f"{layout.ngroups} groups"
    )
    print(
        f"[ok] selected {selected.size} particle IDs from halo IDs "
        f"{','.join(map(str, requested))}"
    )
    print(
        f"[ok] wrote GenetIC id_file {output_path} "
        f"(range {int(selected[0])}..{int(selected[-1])})"
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Join a DMO RAMSES particle output with a HOP .tag catalogue and "
            "write the selected zero-based IDs in GenetIC id_file format."
        )
    )
    parser.add_argument("ramses_output", type=Path)
    parser.add_argument("hop_tag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--halo-id",
        type=int,
        nargs="+",
        action="append",
        required=True,
        help=(
            "one-based halo IDs reported by the HOP .pos catalogue; "
            "the option may be repeated"
        ),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        required=True,
        help="parent particles per dimension; the converter requires N^3 exact IDs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output file",
    )
    args = parser.parse_args()

    convert(
        args.ramses_output.resolve(),
        args.hop_tag.resolve(),
        args.output.resolve(),
        [halo_id for group in args.halo_id for halo_id in group],
        args.grid_size,
        args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
