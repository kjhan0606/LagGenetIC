#!/usr/bin/env python3
"""
Convert a monofonIC white-noise dump to a NumPy .npy file that
GenetIC's `import_level` command can ingest.

Pipeline:
    monofonIC (MPI, parent box)
        --dump_noise= true (HDF5)   or   --output_format= grafic
        \\
         \\---> this script ---> wn_level<N>.npy
                                       \\
                                        \\---> GenetIC `import_level N wn_level<N>.npy`
                                              (reads via Field::loadGridData ->
                                               io::numpy::LoadArrayFromNumpy)

Supported inputs:
  * HDF5  : single 3-D dataset called WhiteNoise (or selectable with --dset)
  * GRAFIC: a `ic_white` or `ic_deltab` file emitted by monofonIC's
            grafic writer (3-D real field on a uniform cube)

Notes:
  - GenetIC's `import_level <lvl> <file>` (bindings.hpp:142 -> ic.hpp:810)
    calls Field::loadGridData(file) -> io::numpy::LoadArrayFromNumpy,
    which checks that the array is N x N x N where N = level grid size.
  - GenetIC was built with -DDOUBLEPRECISION, so it expects float64.
    We always cast to float64 here regardless of source precision.
  - Real-space convention: GenetIC loads the field, calls toFourier(),
    then applies the dm whitenoise transfer ratio. Pass the *unscaled*
    real-space white noise (the standard normal field), NOT the density.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def load_hdf5(path: Path, dset: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        if dset not in f:
            raise KeyError(
                f"dataset '{dset}' not in {path}. "
                f"Available top-level keys: {list(f.keys())}"
            )
        arr = f[dset][...]
    if arr.ndim != 3:
        raise ValueError(f"expected 3-D array, got shape {arr.shape}")
    if arr.shape[0] != arr.shape[1] or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"expected cubic grid, got shape {arr.shape}")
    return arr


def load_grafic(path: Path) -> np.ndarray:
    """
    GRAFIC binary format (as written by monofonIC / RAMSES):
      record 1: 4*int + 4*float (header)
        np1, np2, np3 (int32 each), dx (float32),
        x1o, x2o, x3o (float32), astart, omegam, omegav, h0 (float32)
      then np3 records, each a (np1 * np2) float32 slab.
    Each Fortran record is bracketed by 4-byte length markers.
    """
    with open(path, "rb") as f:
        marker = struct.unpack("<i", f.read(4))[0]
        # header is 11 * 4 = 44 bytes (but monofonIC writes 8*int+4 floats=44)
        if marker != 44:
            raise ValueError(
                f"unexpected GRAFIC header marker {marker} in {path} "
                "(expected 44)"
            )
        hdr = struct.unpack("<3i8f", f.read(44))
        np1, np2, np3 = hdr[:3]
        end = struct.unpack("<i", f.read(4))[0]
        if end != marker:
            raise ValueError("GRAFIC header record not closed properly")
        if np1 != np2 or np2 != np3:
            raise ValueError(
                f"expected cubic grid, got ({np1}, {np2}, {np3})"
            )
        arr = np.empty((np3, np2, np1), dtype=np.float32)
        for k in range(np3):
            rec_len = struct.unpack("<i", f.read(4))[0]
            expected = np1 * np2 * 4
            if rec_len != expected:
                raise ValueError(
                    f"slab {k}: rec length {rec_len} != expected {expected}"
                )
            slab = np.frombuffer(f.read(expected), dtype=np.float32)
            arr[k] = slab.reshape((np2, np1))
            end = struct.unpack("<i", f.read(4))[0]
            if end != rec_len:
                raise ValueError(f"slab {k} record not closed properly")
    # GRAFIC orders axes (np3, np2, np1) which corresponds to (z, y, x).
    # GenetIC's loadGridData stores a flat row-major array of size N^3;
    # the on-disk numpy ordering matches what GenetIC writes via dumpGridData
    # (also row-major C-order). Transpose to (x, y, z) for GenetIC parity.
    return np.transpose(arr, (2, 1, 0))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("input", type=Path,
                   help="monofonIC white-noise file (HDF5 or GRAFIC)")
    p.add_argument("output", type=Path,
                   help="output .npy path (GenetIC import_level argument)")
    p.add_argument("--format", choices=("auto", "hdf5", "grafic"),
                   default="auto",
                   help="input format (default: infer from extension)")
    p.add_argument("--dset", default="WhiteNoise",
                   help="HDF5 dataset name (default: WhiteNoise)")
    p.add_argument("--expect-n", type=int, default=0,
                   help="if >0, assert grid side length equals this value "
                        "(set to the GenetIC level grid size for safety)")
    args = p.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 1

    fmt = args.format
    if fmt == "auto":
        suffix = args.input.suffix.lower()
        if suffix in (".h5", ".hdf5"):
            fmt = "hdf5"
        elif suffix == ".npy":
            print("error: input is already .npy; nothing to do",
                  file=sys.stderr)
            return 1
        else:
            fmt = "grafic"
        print(f"[info] auto-detected format: {fmt}")

    if fmt == "hdf5":
        arr = load_hdf5(args.input, args.dset)
    else:
        arr = load_grafic(args.input)

    n = arr.shape[0]
    if args.expect_n and n != args.expect_n:
        print(f"error: grid size {n} != --expect-n {args.expect_n}",
              file=sys.stderr)
        return 2

    # GenetIC built with -DDOUBLEPRECISION expects float64.
    out = np.ascontiguousarray(arr, dtype=np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out, allow_pickle=False)

    print(f"[ok] wrote {args.output}  shape={out.shape}  dtype={out.dtype}")
    print(f"     mean={out.mean():.4e}  std={out.std():.4e}  "
          f"min={out.min():.4e}  max={out.max():.4e}")
    print(f"     -> GenetIC paramfile:  import_level <lvl> "
          f"{os.path.basename(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
