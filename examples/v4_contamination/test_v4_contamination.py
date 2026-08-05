#!/usr/bin/env python3
"""Regression tests for the V4 contamination and mask-growth tools."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
MEASURE = HERE / "measure_contamination.py"
sys.path.insert(0, str(HERE))

from buffer_mask import grow_ids, load_id_file  # noqa: E402
from measure_contamination import measure  # noqa: E402


def record(payload: bytes) -> bytes:
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def write_part(
    path: Path,
    ncpu: int,
    positions: np.ndarray,
    masses: np.ndarray,
    ids: np.ndarray,
    layout: str = "legacy",
) -> None:
    count = masses.size
    chunks = [
        struct.pack("<i", ncpu),
        struct.pack("<i", 3),
        struct.pack("<i", count),
        struct.pack("<4i", 1, 2, 3, 4),
        struct.pack("<q", 0),
        struct.pack("<d", 0.0),
        struct.pack("<d", 0.0),
        struct.pack("<i", 0),
    ]
    chunks.extend(
        np.asarray(positions[:, axis], dtype="<f8").tobytes()
        for axis in range(3)
    )
    chunks.extend(np.zeros(count, dtype="<f8").tobytes() for _ in range(3))
    chunks.append(np.asarray(masses, dtype="<f8").tobytes())
    chunks.append(np.asarray(ids, dtype="<i8").tobytes())
    chunks.append(np.full(count, 7, dtype="<i4").tobytes())
    if layout == "legacy":
        chunks.append(np.zeros(count, dtype=np.int8).tobytes())
        chunks.append(np.zeros(count, dtype="<f8").tobytes())
    elif layout == "modern":
        chunks.append(np.ones(count, dtype=np.int8).tobytes())
        chunks.append(np.zeros(count, dtype=np.int8).tobytes())
    else:
        raise ValueError(layout)
    path.write_bytes(b"".join(record(chunk) for chunk in chunks))


def write_snapshot(root: Path, contaminated: bool, layout: str = "legacy") -> Path:
    output = root / "output_00001"
    output.mkdir(parents=True)
    (output / "info_00001.txt").write_text("boxlen = 1.0\n")
    positions = np.asarray(
        [
            [0.99, 0.00, 0.00],
            [0.15, 0.00, 0.00],
            [0.075 if contaminated else 0.40, 0.00, 0.00],
            [0.20 if contaminated else 0.45, 0.00, 0.00],
        ]
    )
    masses = np.asarray([1.0, 1.0, 8.0, 8.0])
    write_part(
        output / "part_00001.out00001",
        2,
        positions[:2],
        masses[:2],
        np.asarray([0, 1]),
        layout=layout,
    )
    write_part(
        output / "part_00001.out00002",
        2,
        positions[2:],
        masses[2:],
        np.asarray([2, 3]),
        layout=layout,
    )
    return output


def check_mask_growth(root: Path) -> None:
    seed = np.asarray([0], dtype=np.int64)
    grown = grow_ids(seed, 4, shells=1)
    expected = {
        x * 16 + y * 4 + z
        for x in (0, 1, 3)
        for y in (0, 1, 3)
        for z in (0, 1, 3)
    }
    if set(map(int, grown)) != expected:
        raise AssertionError("periodic corner dilation does not contain 27 cells")
    twice = grow_ids(seed, 4, shells=2)
    if twice.size != 64:
        raise AssertionError("two shells on a 4^3 grid must cover the full grid")

    duplicate = root / "duplicate.id"
    duplicate.write_text("0\n0\n")
    try:
        load_id_file(duplicate, 4)
    except ValueError as error:
        if "duplicate particle ID" not in str(error):
            raise
    else:
        raise AssertionError("duplicate id_file entry was not rejected")


def check_measurement(root: Path) -> None:
    contaminated = write_snapshot(root / "bad", contaminated=True)
    failed = measure(contaminated, (0.0, 0.0, 0.0), 0.05)
    if failed["passed"]:
        raise AssertionError("contaminated synthetic snapshot passed")
    inner = failed["apertures"]["inner"]
    outer = failed["apertures"]["outer"]
    if inner["low_resolution_particle_count"] != 1:
        raise AssertionError("inner low-resolution count is incorrect")
    if not np.isclose(outer["low_resolution_mass_fraction"], 16.0 / 18.0):
        raise AssertionError("outer low-resolution mass fraction is incorrect")

    clean = write_snapshot(root / "good", contaminated=False)
    passed = measure(clean, (0.0, 0.0, 0.0), 0.05)
    if not passed["passed"]:
        raise AssertionError(f"clean synthetic snapshot failed: {passed}")
    if passed["apertures"]["inner"]["particle_count"] != 1:
        raise AssertionError("periodic minimum-image distance was not applied")

    modern = write_snapshot(root / "modern", contaminated=False, layout="modern")
    modern_result = measure(modern, (0.0, 0.0, 0.0), 0.05)
    if modern_result["particle_layout"] != "modern" or not modern_result["passed"]:
        raise AssertionError("modern family/tag snapshot was not detected")


def check_cli_growth(root: Path) -> None:
    contaminated = root / "bad" / "output_00001"
    mask = root / "mask.id"
    next_mask = root / "mask_shell01.id"
    metric = root / "metric.json"
    mask.write_text("0\n")
    result = subprocess.run(
        [
            sys.executable,
            str(MEASURE),
            str(contaminated),
            "--center",
            "0",
            "0",
            "0",
            "--void-radius",
            "0.05",
            "--json",
            str(metric),
            "--mask",
            str(mask),
            "--next-mask",
            str(next_mask),
            "--grid-size",
            "4",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 1:
        raise RuntimeError(result.stdout)
    if not metric.exists() or not next_mask.exists():
        raise AssertionError("failed metric did not write JSON and the next mask")
    payload = json.loads(metric.read_text())
    if payload["passed"]:
        raise AssertionError("failed CLI metric JSON reports a pass")
    if payload["buffer_growth"]["output_particle_ids"] != 27:
        raise AssertionError("metric JSON omits the generated buffer shell")
    if np.loadtxt(next_mask, dtype=np.int64, ndmin=1).size != 27:
        raise AssertionError("CLI did not add one periodic mask shell")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="voidsim_v4_") as directory:
        root = Path(directory)
        check_mask_growth(root)
        check_measurement(root)
        check_cli_growth(root)
    print("V4 CONTAMINATION AND BUFFER-GROWTH REGRESSION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
