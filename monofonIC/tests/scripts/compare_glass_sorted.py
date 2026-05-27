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
    """Return (pos, vel, mass_or_None). Masses dataset is optional — the
    Gadget plugin only emits it when particles have individual masses
    (e.g. masked-SC). For uniform-mass lattices it lives in
    /Header/MassTable and there is no /PartType*/Masses dataset."""
    pos_chunks, vel_chunks, mass_chunks = [], [], []
    has_masses = None
    for f in files:
        with h5py.File(f, "r") as h:
            grp = h[f"PartType{part}"]
            pos_chunks.append(np.asarray(grp["Coordinates"]))
            vel_chunks.append(np.asarray(grp["Velocities"]))
            this_has = "Masses" in grp
            if has_masses is None:
                has_masses = this_has
            elif has_masses != this_has:
                raise RuntimeError(
                    f"PartType{part}: Masses dataset present in some files but "
                    f"not others — file {f} disagrees"
                )
            if this_has:
                mass_chunks.append(np.asarray(grp["Masses"]))
    pos = np.concatenate(pos_chunks)
    vel = np.concatenate(vel_chunks)
    mass = np.concatenate(mass_chunks) if has_masses else None
    return pos, vel, mass


def sort_by_pos(pos, vel, mass=None, ndigits=8):
    key = np.round(pos, ndigits)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    if mass is None:
        return pos[order], vel[order], None
    return pos[order], vel[order], mass[order]


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
            rp, rv, rm = load_partset(ref_files, part)
            tp, tv, tm = load_partset(test_files, part)
        except KeyError:
            continue
        print(f"PartType{part}: N_ref={rp.shape[0]} N_test={tp.shape[0]}"
              f"  Masses={'yes' if rm is not None else 'no'}")
        rp, rv, rm = sort_by_pos(rp, rv, rm)
        tp, tv, tm = sort_by_pos(tp, tv, tm)
        ok_p = cmp(f"PartType{part}/Coordinates", rp, tp, args.rtol, args.atol)
        ok_v = cmp(f"PartType{part}/Velocities", rv, tv, args.rtol, args.atol)
        all_ok = all_ok and ok_p and ok_v
        if (rm is None) != (tm is None):
            print(f"  PartType{part}/Masses: present in one but not the other"
                  f" (ref={rm is not None}, test={tm is not None})  FAIL")
            all_ok = False
        elif rm is not None:
            ok_m = cmp(f"PartType{part}/Masses", rm, tm, args.rtol, args.atol)
            all_ok = all_ok and ok_m

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
