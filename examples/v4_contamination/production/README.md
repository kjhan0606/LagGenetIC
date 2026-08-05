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
   base-grid Poisson solver, writes five snapshots, and verifies the final
   particle-ID permutation before declaring success.

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

The job is named `void_dmo512`; its run directory is
`/gpfs/kjhan/VoidSim/v4_parent_n512/dmo_z0`.  The submitter refuses to replace
an existing run, rejects a changed lagRamses executable, and records the exact
binary checksum and source revisions in `provenance.txt`.

Successful stages leave `.STAGE.complete` markers in the GPFS work root.  Set
`V4_WORKROOT` only to choose another dedicated GPFS directory; the runner
accepts only a dedicated directory below `/gpfs/kjhan`.
