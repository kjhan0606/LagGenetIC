#!/usr/bin/env python3
"""Extract the standard 13 CAMB columns from an extended lagCAMB table."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


STANDARD_COLUMNS = 13


def convert(source: Path, destination: Path) -> tuple[int, int]:
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    rows = 0
    maximum_columns = 0
    temporary: str | None = None
    try:
        with source.open() as reader, tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as writer:
            temporary = writer.name
            for line_number, line in enumerate(reader, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    names = stripped[1:].split()
                    writer.write("# " + " ".join(names[:STANDARD_COLUMNS]) + "\n")
                    continue
                values = stripped.split()
                maximum_columns = max(maximum_columns, len(values))
                if len(values) < STANDARD_COLUMNS:
                    raise ValueError(
                        f"{source}:{line_number}: found {len(values)} columns; "
                        f"need at least {STANDARD_COLUMNS}"
                    )
                for value in values:
                    float(value.replace("D", "E").replace("d", "e"))
                writer.write(" ".join(values[:STANDARD_COLUMNS]) + "\n")
                rows += 1
        if rows < 2:
            raise ValueError(f"{source}: transfer table contains only {rows} data rows")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return rows, maximum_columns


def check_k_coverage(path: Path, minimum_kmax: float) -> tuple[float, float]:
    wave_numbers = []
    with path.open() as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                wave_numbers.append(float(line.split()[0].replace("D", "E")))
    if len(wave_numbers) < 2 or any(
        right <= left for left, right in zip(wave_numbers, wave_numbers[1:])
    ):
        raise ValueError(f"{path}: transfer wave numbers are not increasing")
    if wave_numbers[-1] < minimum_kmax:
        raise ValueError(
            f"{path}: kmax={wave_numbers[-1]} does not reach {minimum_kmax} h/Mpc"
        )
    return wave_numbers[0], wave_numbers[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--minimum-kmax", type=float)
    args = parser.parse_args()
    rows, source_columns = convert(args.source, args.destination)
    coverage = ""
    if args.minimum_kmax is not None:
        minimum_k, maximum_k = check_k_coverage(
            args.destination, args.minimum_kmax
        )
        coverage = f", k={minimum_k:.6e}..{maximum_k:.6e} h/Mpc"
    print(
        f"[ok] {args.source}: {rows} rows, up to {source_columns} columns; "
        f"wrote standard {STANDARD_COLUMNS}-column table {args.destination}"
        f"{coverage}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
