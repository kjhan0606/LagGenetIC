#!/usr/bin/env python3
"""Regression test for the HOP-to-GenetIC particle-ID conversion."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
CONVERTER = HERE / "hop_to_genetic_id.py"


def record(payload: bytes) -> bytes:
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def write_part(path: Path, ncpu: int, ids: list[int]) -> None:
    count = len(ids)
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
    chunks.extend(np.zeros(count, dtype="<f8").tobytes() for _ in range(7))
    chunks.append(np.asarray(ids, dtype="<i8").tobytes())
    chunks.append(np.ones(count, dtype="<i4").tobytes())
    chunks.append(np.zeros(count, dtype="i1").tobytes())
    chunks.append(np.zeros(count, dtype="i1").tobytes())
    path.write_bytes(b"".join(record(chunk) for chunk in chunks))


def write_tag(path: Path, tags: list[int], ngroups: int, fortran: bool) -> None:
    header = struct.pack("<ii", len(tags), ngroups)
    payload = np.asarray(tags, dtype="<i4").tobytes()
    path.write_bytes(record(header) + record(payload) if fortran else header + payload)


def run_converter(
    root: Path, tag: Path, output: Path, extra: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(CONVERTER),
        str(root / "output_00001"),
        str(tag),
        str(output),
        "--halo-id",
        "2",
        "--halo-id",
        "3",
        "--halo-id",
        "2",
        "--grid-size",
        "2",
    ]
    if extra:
        command.extend(extra)
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def check_genetic(binary: Path, root: Path, ids: Path) -> None:
    roundtrip = root / "roundtrip.id"
    parameter_file = root / "genetic.txt"
    parameter_file.write_text(
        "\n".join(
            [
                "base_grid 1.0 2",
                f"id_file {ids}",
                f"dump_id_file {roundtrip}",
                "",
            ]
        )
    )
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [str(binary.resolve()), str(parameter_file)],
        cwd=root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GenetIC id_file round trip failed:\n{result.stdout}")
    if ids.read_text() != roundtrip.read_text():
        raise AssertionError("GenetIC changed the selected particle-ID set")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genetic-binary", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="voidsim_v2_hop_") as directory:
        root = Path(directory)
        output_dir = root / "output_00001"
        output_dir.mkdir()
        write_part(output_dir / "part_00001.out00001", 2, [5, 0, 7])
        write_part(output_dir / "part_00001.out00002", 2, [3, 1, 6, 2, 4])
        tags = [-1, 0, 1, 1, 2, 0, 2, 1]
        expected = np.asarray([1, 2, 3, 4, 7], dtype=np.int64)

        fortran_tag = root / "groups_fortran.tag"
        write_tag(fortran_tag, tags, 3, fortran=True)
        output = root / "selected.id"
        result = run_converter(root, fortran_tag, output)
        if result.returncode != 0:
            raise RuntimeError(result.stdout)
        actual = np.loadtxt(output, dtype=np.int64, ndmin=1)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"Fortran HOP selection differs: {actual}")

        raw_tag = root / "groups_raw.tag"
        write_tag(raw_tag, tags, 3, fortran=False)
        raw_output = root / "selected_raw.id"
        result = run_converter(root, raw_tag, raw_output)
        if result.returncode != 0:
            raise RuntimeError(result.stdout)
        raw_actual = np.loadtxt(raw_output, dtype=np.int64, ndmin=1)
        if not np.array_equal(raw_actual, expected):
            raise AssertionError(f"raw HOP selection differs: {raw_actual}")

        overwrite = run_converter(root, raw_tag, raw_output)
        if overwrite.returncode == 0 or "output exists" not in overwrite.stdout:
            raise AssertionError("existing output was not rejected")

        bad_tag = root / "bad_count.tag"
        write_tag(bad_tag, tags[:-1], 3, fortran=True)
        bad = run_converter(root, bad_tag, root / "bad.id")
        if bad.returncode == 0 or "parent requires 8" not in bad.stdout:
            raise AssertionError("HOP/RAMSES particle-count mismatch was not rejected")

        if args.genetic_binary is not None:
            check_genetic(args.genetic_binary, root, output)

    suffix = " and GenetIC round trip" if args.genetic_binary is not None else ""
    print(f"PASS: raw and Fortran HOP layouts{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
