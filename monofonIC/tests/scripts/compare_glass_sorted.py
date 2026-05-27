#!/usr/bin/env python3
"""Compare two glass-output HDF5 files (or multi-file gadget sets) by sorting
particles on initial position. Required because the ksec glass path assigns
IDs by post-domain-decomp order, so the slab and ksec runs produce identical
particles in different orders.

Usage:
  compare_glass_sorted.py <ref> <test> [--rtol R] [--atol A]

Either argument may be:
  - a single-file output:    foo.hdf5
  - a multi-file gadget set: foo.0.hdf5, foo.1.hdf5, ... (pass foo.0.hdf5;
    siblings are auto-detected)

Exit codes:
  0 - Match within tolerance
  1 - Differ
  2 - Error
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np


def expand_multifile(path):
    """Return a list of HDF5 files: just [path] if single-file, or all siblings
    foo.0.hdf5, foo.1.hdf5, ... if the given path looks like an indexed file."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m = re.match(r"^(.*)\.(\d+)\.hdf5$", path)
    if not m:
        return [path]
    base = m.group(1)
    siblings = sorted(
        glob.glob(f"{base}.*.hdf5"),
        key=lambda p: int(re.match(rf"^{re.escape(base)}\.(\d+)\.hdf5$", p).group(1)),
    )
    return siblings or [path]


def load_partset(files, part):
    pos_chunks, vel_chunks = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            grp = h[f"PartType{part}"]
            pos_chunks.append(np.asarray(grp["Coordinates"]))
            vel_chunks.append(np.asarray(grp["Velocities"]))
    pos = np.concatenate(pos_chunks)
    vel = np.concatenate(vel_chunks)
    return pos, vel


def sort_by_pos(pos, vel, ndigits=8):
    key = np.round(pos, ndigits)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    return pos[order], vel[order]


def cmp(name, a, b, rtol, atol):
    if a.shape != b.shape:
        print(f"  {name}: shape mismatch {a.shape} vs {b.shape}")
        return False
    abs_diff = np.abs(a - b)
    rel_diff = abs_diff / (np.abs(a) + atol)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        print(f"  {name}: max|abs|={abs_diff.max():.3e} max|rel|={rel_diff.max():.3e}  FAIL")
        return False
    print(f"  {name}: max|abs|={abs_diff.max():.3e} max|rel|={rel_diff.max():.3e}  OK")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--atol", type=float, default=1e-14)
    args = ap.parse_args(argv)

    try:
        ref_files = expand_multifile(args.ref)
        test_files = expand_multifile(args.test)
    except FileNotFoundError as e:
        print(f"file not found: {e}", file=sys.stderr)
        return 2

    print(f"ref:  {ref_files}")
    print(f"test: {test_files}")

    all_ok = True
    for part in (0, 1):
        try:
            rp, rv = load_partset(ref_files, part)
            tp, tv = load_partset(test_files, part)
        except KeyError:
            continue
        print(f"PartType{part}: N_ref={rp.shape[0]} N_test={tp.shape[0]}")
        rp, rv = sort_by_pos(rp, rv)
        tp, tv = sort_by_pos(tp, tv)
        ok_p = cmp(f"PartType{part}/Coordinates", rp, tp, args.rtol, args.atol)
        ok_v = cmp(f"PartType{part}/Velocities", rv, tv, args.rtol, args.atol)
        all_ok = all_ok and ok_p and ok_v

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
