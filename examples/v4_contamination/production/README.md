# V4 level-9 parent production gate

This directory scales the validated hand-off to a `512^3` parent in a
`512 Mpc/h` periodic box.  All large files and simulation work are placed under
`/gpfs/kjhan/VoidSim/v4_parent_n512`; the source tree holds only inputs and
validation code.

The sequence is:

1. lagCAMB writes its extended transfer table at `z=49`, and the preparation
   step preserves the raw file while extracting the standard 13 CAMB columns
   accepted by both IC codes;
2. monofonIC draws the distributed parent white noise with 16 MPI ranks and
   four OpenMP threads per rank;
3. the converter makes an exact float64 NumPy hand-off;
4. GenetIC imports that white noise and writes the level-9 DMO parent GRAFIC IC;
5. lagRamses ingests the IC on 16 MPI ranks and writes the initial snapshot;
6. the verifier checks the transfer coverage, exact HDF5/NumPy white-noise
   equality, Cartesian GRAFIC IDs, the complete RAMSES ID permutation, and the
   `A_s` branch reported by both IC codes.
7. the production DMO run evolves that validated fixed level-9 parent from
   `a=0.02` to `a=1` on one Grammar node with 64 MPI ranks.  It uses the FFTW3
   base-grid Poisson solver and writes the initial state followed by five
   scheduled snapshots.  The final gate verifies the complete particle-ID
   permutation before declaring success.

Run all stages, or one named resumable stage, with:

```bash
./run_parent.sh
./run_parent.sh camb
./run_parent.sh monofonic
./run_parent.sh convert
./run_parent.sh genetic
./run_parent.sh ramses
./run_parent.sh verify
```

After the level-9 gate passes, stage and submit the production evolution with:

```bash
./submit_dmo_z0.sh
```

After the z=0 verifier passes, build the parent HOP catalogue and select the
three most massive compact, non-wrapping Lagrangian patches on LagEunha with:

```bash
./run_parent_hop_lageunha.sh
```

The production selector examines the 32 most populated HOP groups in one pass
over the complete `512^3` RAMSES ID permutation.  A target must contain at
least 3000 parent particles, have a periodic Lagrangian envelope narrower than
half the box on every axis, fit after recentering with a two-cell buffer, and
lie at least 0.1 box lengths from an already selected target.  The measured
recentring shift prevents an arbitrary parent-box origin from biasing the
sample.  The selector writes the three GenetIC `id_file` masks and
machine-readable JSON/TSV diagnostics under `dmo_z0/parent_targets`.

The job is named `void_dmo512`; its run directory is
`/gpfs/kjhan/VoidSim/v4_parent_n512/dmo_z0`.  The submitter refuses to replace
simulation output and records the exact binary checksum, its build revision,
and the current source revisions in `provenance.txt`.  An environment-only
retry reuses the validated staged executable even if the developer worktree
has since rebuilt its default binary.

Successful stages leave `.STAGE.complete` markers in the GPFS work root.  Set
`V4_WORKROOT` only to choose another dedicated GPFS directory; the runner
accepts only a dedicated directory below `/gpfs/kjhan`.
