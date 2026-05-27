#!/usr/bin/env python3
"""Generate a tiny glass HDF5 file for monofonIC smoke testing.

Format (matches glass_loader.hh):
  /Header        attribute BoxSize = 1.0 (scalar)
  /PartType1/Coordinates  shape (N, 3), float64, in [0, BoxSize)
"""
import h5py
import numpy as np

# 4x4x4 SC lattice perturbed slightly so it is non-degenerate
n_side = 4
np_total = n_side ** 3
boxsize = 1.0

rng = np.random.default_rng(seed=42)

xs = (np.arange(n_side) + 0.5) / n_side
gx, gy, gz = np.meshgrid(xs, xs, xs, indexing='ij')
coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
# Perturb by up to ~0.1 cell
coords += (rng.random(coords.shape) - 0.5) * (0.2 / n_side)
coords = np.mod(coords, boxsize).astype(np.float64)

assert coords.shape == (np_total, 3)
assert coords.dtype == np.float64
assert (coords >= 0).all() and (coords < boxsize).all()

out = "glass64.hdf5"
with h5py.File(out, "w") as f:
    hdr = f.create_group("Header")
    hdr.attrs.create("BoxSize", boxsize, dtype=np.float64)
    pt = f.create_group("PartType1")
    pt.create_dataset("Coordinates", data=coords, dtype=np.float64)

print(f"Wrote {out}: {np_total} particles, BoxSize={boxsize}")
