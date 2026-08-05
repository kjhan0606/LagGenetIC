#!/usr/bin/env python3
"""Measure low-resolution particle contamination around a periodic void."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent / "v2_hop_id_file"
sys.path.insert(0, str(V2_DIR))

from hop_to_genetic_id import (  # noqa: E402
    FortranRecordReader,
    _scalar,
    find_part_files,
)
from buffer_mask import grow_file  # noqa: E402


@dataclass(frozen=True)
class ParticleChunk:
    ncpu: int
    positions: np.ndarray | None
    masses: np.ndarray
    type_codes: np.ndarray
    layout: str


def _real_array(payload: bytes, endian: str, count: int, name: str) -> np.ndarray:
    for size in (8, 4):
        if len(payload) == size * count:
            return np.frombuffer(
                payload, dtype=np.dtype(f"{endian}f{size}")
            ).astype(np.float64, copy=False)
    raise ValueError(
        f"{name}: record has {len(payload)} bytes for {count} real values"
    )


def _optional_record(records: FortranRecordReader) -> bytes | None:
    position = records.handle.tell()
    marker = records.handle.read(1)
    records.handle.seek(position)
    if not marker:
        return None
    return records.read()


def read_particle_chunk(
    path: Path,
    with_positions: bool,
    requested_layout: str = "auto",
) -> ParticleChunk:
    """Read fields common to legacy lagRamses and current RAMSES outputs."""

    with FortranRecordReader(path) as records:
        endian = records.endian
        ncpu = _scalar(records.read(), endian, f"{path}: ncpu")
        ndim = _scalar(records.read(), endian, f"{path}: ndim")
        npart = _scalar(records.read(), endian, f"{path}: npart")
        records.skip()
        records.skip()
        records.skip()
        records.skip()
        records.skip()

        if ndim != 3:
            raise ValueError(f"{path}: expected ndim=3, found {ndim}")
        if npart < 0:
            raise ValueError(f"{path}: negative particle count {npart}")

        if with_positions:
            coordinates = [
                _real_array(
                    records.read(), endian, npart, f"{path}: position {axis}"
                )
                for axis in range(ndim)
            ]
            positions = np.column_stack(coordinates)
        else:
            for _ in range(ndim):
                records.skip()
            positions = None

        for _ in range(ndim):
            records.skip()
        masses = _real_array(records.read(), endian, npart, f"{path}: mass")
        records.skip()
        records.skip()
        type_payload = records.read()
        if len(type_payload) != npart:
            raise ValueError(
                f"{path}: particle type record has {len(type_payload)} bytes "
                f"for {npart} particles"
            )
        type_codes = np.frombuffer(type_payload, dtype=np.int8)
        following = _optional_record(records)

    if npart == 0:
        inferred_layout = "unknown"
    elif following is not None and len(following) == npart:
        inferred_layout = "modern"
    else:
        inferred_layout = "legacy"
    if requested_layout not in ("auto", "legacy", "modern"):
        raise ValueError(f"unsupported particle layout {requested_layout}")
    layout = inferred_layout if requested_layout == "auto" else requested_layout
    if layout == "modern" and (
        following is None or (npart > 0 and len(following) != npart)
    ):
        raise ValueError(f"{path}: modern family/tag records were not found")
    if layout == "legacy" and (
        npart > 0 and following is not None and len(following) == npart
    ):
        raise ValueError(f"{path}: file contains modern family/tag records")

    return ParticleChunk(ncpu, positions, masses, type_codes, layout)


def read_box_size(output_dir: Path) -> float:
    info_files = sorted(output_dir.glob("info_*.txt"))
    if len(info_files) != 1:
        raise ValueError(
            f"{output_dir}: expected one info_*.txt file, found {len(info_files)}"
        )
    match = re.search(
        r"^\s*boxlen\s*=\s*([+\-0-9.EeDd]+)",
        info_files[0].read_text(),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"{info_files[0]}: boxlen is absent")
    value = float(match.group(1).replace("D", "E").replace("d", "e"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{info_files[0]}: invalid boxlen {value}")
    return value


def _dm_codes(layout: str, requested: tuple[int, ...] | None) -> tuple[int, ...]:
    if requested is not None:
        return requested
    return (0,) if layout == "legacy" else (1,)


def _periodic_distances(
    positions: np.ndarray, center: np.ndarray, box_size: float
) -> np.ndarray:
    wrapped = np.mod(positions, box_size)
    delta = np.abs(wrapped - center)
    delta = np.minimum(delta, box_size - delta)
    return np.sqrt(np.einsum("ij,ij->i", delta, delta))


def measure(
    output_dir: Path,
    center: tuple[float, float, float],
    void_radius: float,
    box_size: float | None = None,
    inner_multiple: float = 2.0,
    outer_multiple: float = 5.0,
    fraction_threshold: float = 1.0e-3,
    fine_mass: float | None = None,
    mass_rtol: float = 1.0e-6,
    layout: str = "auto",
    dm_codes: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if void_radius <= 0 or not math.isfinite(void_radius):
        raise ValueError("void radius must be finite and positive")
    if not 0 < inner_multiple <= outer_multiple:
        raise ValueError("aperture multiples require 0 < inner <= outer")
    if not 0 <= fraction_threshold < 1:
        raise ValueError("mass-fraction threshold must lie in [0,1)")
    if mass_rtol < 0:
        raise ValueError("mass tolerance must be non-negative")

    output_dir = output_dir.resolve()
    if box_size is None:
        box_size = read_box_size(output_dir)
    if box_size <= 0 or not math.isfinite(box_size):
        raise ValueError("box size must be finite and positive")
    if outer_multiple * void_radius > 0.5 * box_size:
        raise ValueError(
            f"outer aperture {outer_multiple * void_radius:g} exceeds half "
            f"the periodic box size {0.5 * box_size:g}"
        )
    center_array = np.mod(np.asarray(center, dtype=np.float64), box_size)
    if center_array.shape != (3,) or not np.all(np.isfinite(center_array)):
        raise ValueError("center must contain three finite coordinates")

    files = find_part_files(output_dir)
    declared_ncpu: int | None = None
    detected_layout: str | None = None
    total_dm_particles = 0
    mass_histogram: dict[float, int] = {}
    minimum_mass = math.inf

    for path in files:
        chunk = read_particle_chunk(path, False, requested_layout=layout)
        if declared_ncpu is None:
            declared_ncpu = chunk.ncpu
        elif chunk.ncpu != declared_ncpu:
            raise ValueError(
                f"{path}: ncpu={chunk.ncpu}, expected {declared_ncpu}"
            )
        if detected_layout is None and chunk.layout != "unknown":
            detected_layout = chunk.layout
        elif chunk.layout not in ("unknown", detected_layout):
            raise ValueError(
                f"{path}: particle layout {chunk.layout} differs from "
                f"{detected_layout}"
            )
        local_dm = np.isin(
            chunk.type_codes, _dm_codes(chunk.layout, dm_codes)
        )
        local_masses = chunk.masses[local_dm]
        if np.any(~np.isfinite(local_masses)) or np.any(local_masses <= 0):
            raise ValueError(f"{path}: dark matter masses must be finite and positive")
        total_dm_particles += int(local_masses.size)
        if local_masses.size:
            minimum_mass = min(minimum_mass, float(local_masses.min()))
            values, counts = np.unique(local_masses, return_counts=True)
            for value, count in zip(values, counts, strict=True):
                key = float(value)
                mass_histogram[key] = mass_histogram.get(key, 0) + int(count)

    if declared_ncpu != len(files):
        raise ValueError(
            f"headers declare {declared_ncpu} files but found {len(files)}"
        )
    if total_dm_particles == 0:
        raise ValueError("snapshot contains no selected dark matter particles")
    if fine_mass is None:
        fine_mass = minimum_mass
    if fine_mass <= 0 or not math.isfinite(fine_mass):
        raise ValueError("fine particle mass must be finite and positive")
    low_mass_cut = fine_mass * (1.0 + mass_rtol)

    apertures = {
        "inner": {
            "multiple": inner_multiple,
            "radius": inner_multiple * void_radius,
            "particle_count": 0,
            "fine_particle_count": 0,
            "low_resolution_particle_count": 0,
            "total_mass": 0.0,
            "low_resolution_mass": 0.0,
        },
        "outer": {
            "multiple": outer_multiple,
            "radius": outer_multiple * void_radius,
            "particle_count": 0,
            "fine_particle_count": 0,
            "low_resolution_particle_count": 0,
            "total_mass": 0.0,
            "low_resolution_mass": 0.0,
        },
    }

    assert detected_layout is not None
    for path in files:
        chunk = read_particle_chunk(path, True, requested_layout=detected_layout)
        local_dm = np.isin(
            chunk.type_codes, _dm_codes(chunk.layout, dm_codes)
        )
        assert chunk.positions is not None
        positions = chunk.positions[local_dm]
        masses = chunk.masses[local_dm]
        if np.any(~np.isfinite(positions)):
            raise ValueError(f"{path}: non-finite dark matter position")
        distances = _periodic_distances(positions, center_array, box_size)
        low_resolution = masses > low_mass_cut
        for aperture in apertures.values():
            inside = distances <= float(aperture["radius"])
            low_inside = inside & low_resolution
            aperture["particle_count"] += int(np.count_nonzero(inside))
            aperture["fine_particle_count"] += int(
                np.count_nonzero(inside & ~low_resolution)
            )
            aperture["low_resolution_particle_count"] += int(
                np.count_nonzero(low_inside)
            )
            aperture["total_mass"] += float(masses[inside].sum())
            aperture["low_resolution_mass"] += float(masses[low_inside].sum())

    for aperture in apertures.values():
        total_mass = float(aperture["total_mass"])
        low_mass = float(aperture["low_resolution_mass"])
        aperture["low_resolution_mass_fraction"] = (
            low_mass / total_mass if total_mass > 0 else None
        )

    inner = apertures["inner"]
    outer = apertures["outer"]
    reasons: list[str] = []
    if int(inner["low_resolution_particle_count"]) != 0:
        reasons.append("low-resolution particles lie inside the inner aperture")
    outer_fraction = outer["low_resolution_mass_fraction"]
    if outer_fraction is None:
        reasons.append("the outer aperture contains no dark matter particles")
    elif float(outer_fraction) >= fraction_threshold:
        reasons.append("the outer low-resolution mass fraction is too large")
    if int(outer["fine_particle_count"]) == 0:
        reasons.append("the outer aperture contains no finest-mass particles")

    return {
        "schema_version": 1,
        "passed": not reasons,
        "failure_reasons": reasons,
        "output_directory": str(output_dir),
        "particle_layout": detected_layout,
        "particle_files": len(files),
        "total_dark_matter_particles": total_dm_particles,
        "box_size": box_size,
        "center": center_array.tolist(),
        "void_radius": void_radius,
        "fine_particle_mass": fine_mass,
        "low_resolution_mass_cut": low_mass_cut,
        "mass_relative_tolerance": mass_rtol,
        "mass_spectrum": [
            {"mass": mass, "particle_count": mass_histogram[mass]}
            for mass in sorted(mass_histogram)
        ],
        "criteria": {
            "inner_multiple": inner_multiple,
            "inner_low_resolution_particle_limit": 0,
            "outer_multiple": outer_multiple,
            "outer_low_resolution_mass_fraction_limit": fraction_threshold,
        },
        "apertures": apertures,
    }


def write_json(path: Path, result: dict[str, object], force: bool = False) -> None:
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
            json.dump(result, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def report(result: dict[str, object]) -> None:
    apertures = result["apertures"]
    assert isinstance(apertures, dict)
    inner = apertures["inner"]
    outer = apertures["outer"]
    assert isinstance(inner, dict) and isinstance(outer, dict)
    fraction = outer["low_resolution_mass_fraction"]
    fraction_text = "undefined" if fraction is None else f"{float(fraction):.6e}"
    status = "PASS" if result["passed"] else "FAIL"
    print(f"V4 CONTAMINATION {status}")
    print(
        f"layout={result['particle_layout']} files={result['particle_files']} "
        f"DM={result['total_dark_matter_particles']} "
        f"m_fine={float(result['fine_particle_mass']):.12e}"
    )
    print(
        f"{float(inner['multiple']):g} R_v: "
        f"{inner['low_resolution_particle_count']} low-resolution particles "
        f"of {inner['particle_count']}"
    )
    print(
        f"{float(outer['multiple']):g} R_v: "
        f"low-resolution mass fraction={fraction_text} "
        f"({outer['low_resolution_particle_count']} particles)"
    )
    for reason in result["failure_reasons"]:
        print(f"  - {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the VoidSim zoom-contamination criteria to a RAMSES "
            "particle snapshot and optionally grow the next GenetIC mask."
        )
    )
    parser.add_argument("ramses_output", type=Path)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--void-radius", type=float, required=True)
    parser.add_argument("--box-size", type=float)
    parser.add_argument("--inner-multiple", type=float, default=2.0)
    parser.add_argument("--outer-multiple", type=float, default=5.0)
    parser.add_argument("--fraction-threshold", type=float, default=1.0e-3)
    parser.add_argument("--fine-mass", type=float)
    parser.add_argument("--mass-rtol", type=float, default=1.0e-6)
    parser.add_argument(
        "--layout", choices=("auto", "legacy", "modern"), default="auto"
    )
    parser.add_argument(
        "--dm-code",
        type=int,
        nargs="+",
        help="particle type/family codes treated as dark matter",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--next-mask", type=Path)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--grow-shells", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    growth_arguments = (args.mask, args.next_mask, args.grid_size)
    if any(value is not None for value in growth_arguments) and not all(
        value is not None for value in growth_arguments
    ):
        parser.error("--mask, --next-mask, and --grid-size must be given together")
    if args.next_mask is not None and args.next_mask.exists() and not args.force:
        parser.error(
            f"{args.next_mask}: output exists; pass --force to replace it"
        )
    if args.json is not None and args.json.exists() and not args.force:
        parser.error(f"{args.json}: output exists; pass --force to replace it")

    result = measure(
        args.ramses_output,
        tuple(args.center),
        args.void_radius,
        box_size=args.box_size,
        inner_multiple=args.inner_multiple,
        outer_multiple=args.outer_multiple,
        fraction_threshold=args.fraction_threshold,
        fine_mass=args.fine_mass,
        mass_rtol=args.mass_rtol,
        layout=args.layout,
        dm_codes=tuple(args.dm_code) if args.dm_code is not None else None,
    )
    report(result)

    if not result["passed"] and args.next_mask is not None:
        assert args.mask is not None and args.grid_size is not None
        original, grown = grow_file(
            args.mask.resolve(),
            args.next_mask.resolve(),
            args.grid_size,
            args.grow_shells,
            force=args.force,
        )
        result["buffer_growth"] = {
            "input_mask": str(args.mask.resolve()),
            "output_mask": str(args.next_mask.resolve()),
            "shells": args.grow_shells,
            "input_particle_ids": int(original.size),
            "output_particle_ids": int(grown.size),
        }
        print(
            f"[next] contamination failed; grew mask "
            f"{original.size} -> {grown.size} IDs"
        )
        print(f"[next] rerun with {args.next_mask.resolve()}")

    if args.json is not None:
        write_json(args.json.resolve(), result, force=args.force)
        print(f"[ok] wrote metric JSON {args.json.resolve()}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
