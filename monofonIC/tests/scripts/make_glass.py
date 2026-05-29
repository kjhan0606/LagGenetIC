#!/usr/bin/env python3
"""Generate a glass HDF5 file for monofonIC.

Format (matches glass_loader.hh):
  /Header        group with attribute BoxSize (float64, scalar)
  /PartType1/Coordinates  shape (n_side**3, 3), float64, in [0, BoxSize)

Default (no args): writes glass64.hdf5 with n_side=4 (64 particles),
matching the ksection consistency tests.
"""
import argparse
import h5py
import numpy as np


def make(n_side: int, out: str, boxsize: float = 1.0, seed: int = 42) -> int:
    rng = np.random.default_rng(seed=seed)
    xs = (np.arange(n_side) + 0.5) / n_side
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing='ij')
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    coords += (rng.random(coords.shape) - 0.5) * (0.2 / n_side)
    coords = np.mod(coords, boxsize).astype(np.float64)

    assert coords.shape == (n_side ** 3, 3)
    assert (coords >= 0).all() and (coords < boxsize).all()

    with h5py.File(out, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs.create("BoxSize", boxsize, dtype=np.float64)
        pt = f.create_group("PartType1")
        pt.create_dataset("Coordinates", data=coords, dtype=np.float64)

    return n_side ** 3


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-side", type=int, default=4,
                    help="lattice resolution per axis (default 4 = 64 particles)")
    ap.add_argument("--out", default="glass64.hdf5",
                    help="output HDF5 filename (default glass64.hdf5)")
    ap.add_argument("--boxsize", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = make(args.n_side, args.out, args.boxsize, args.seed)
    print(f"Wrote {args.out}: {n} particles, BoxSize={args.boxsize}")
