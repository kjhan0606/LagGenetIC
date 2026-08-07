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

The next matched control globally reverses the same imported white noise and
evolves a second fixed level-9 parent on LagEunha:

```bash
./run_inverted_parent_lageunha.sh
```

Before evolution, the runner checks `delta_inverted=-delta_normal` exactly in
all `512^3` cells and requires bit-identical Cartesian particle IDs.  The
inverted z=0 parent supplies the Eulerian void centers, radii, and density
profiles needed before any high-resolution hydrodynamic run is staged.

The first post-processing pass tracks the selected particle IDs into the
inverted final snapshot, locates a local minimum in a 4 Mpc/h Gaussian-smoothed
`256^3` density field, and measures spherical shell and enclosed-density
profiles:

```bash
python3 measure_inverted_voids.py \
  /gpfs/kjhan/VoidSim/v4_parent_n512/inverted_parent_z0/evolution/output_00006 \
  /gpfs/kjhan/VoidSim/v4_parent_n512/dmo_z0/parent_targets \
  /gpfs/kjhan/VoidSim/v4_parent_n512/inverted_parent_z0/void_analysis
```

The reported outermost radius with enclosed `delta <= -0.8` is explicitly a
pre-watershed diagnostic, not the final catalogue effective radius.

After the inverted parent and preliminary profile gates pass, the target-1
hand-off regression creates a matched normal/inverted two-level GRAFIC pair.
The selected Lagrangian span is 48 Mpc/h, so a centered 128-cell fine grid
spans 64 Mpc/h at effective `1024^3` resolution without allocating a full-box
fine cube.  The launcher checks exact sign reversal on both levels and then
uses 16 MPI ranks for a one-step lagRamses ingestion test:

```bash
ssh lageunha \
  /home/kjhan/BACKUP/VoidSim/code/LagGenetIC/examples/v4_contamination/production/run_target1_zoom_validation_lageunha.sh
```

This is a format, mask, and particle-ID regression only.  It is not a
production zoom evolution and does not set the final nested hierarchy.
Because `nexpand=1`, lagRamses is expected to add a one-oct boundary layer;
the level-10 grid count can therefore exceed the exact target refmap count.
The one-step check retains the standard eight-particle refinement criterion.
The namelist repeats the value through every active level because
`m_refine` is a level array.  A single scalar would set only its first entry
and leave the zoom levels disabled.  Setting the array to zero would request
whole-box refinement after the first step and is not a valid way to disable
dynamic refinement.

The next scale gate preserves the same 64 Mpc/h patch while adding an
effective `2048^3` level.  The second `zoom_grid 1 256` command is relative to
the preceding 64 Mpc/h grid.  A factor of one is therefore required to retain
the physical target volume.  The runner records wall and CPU time for both
GenetIC cases, the IC verifier, and the 16-rank RAMSES ingestion:

```bash
ssh lageunha \
  /home/kjhan/BACKUP/VoidSim/code/LagGenetIC/examples/v4_contamination/production/run_target1_level11_validation_lageunha.sh
```

This level-11 run is a measured resource and hierarchy gate.  It verifies all
three GRAFIC levels, both AMR mesh counts, the expected hierarchical particle
count, and global ID uniqueness.  It is not a final level-14 production IC.
The full 64 Mpc/h hierarchy would require dense grids of 512 cells at level
12, 1024 cells at level 13, and 2048 cells at level 14.  Those allocations are
deferred until the target-resolution and patch-size trade-off is fixed.

The job is named `void_dmo512`; its run directory is
`/gpfs/kjhan/VoidSim/v4_parent_n512/dmo_z0`.  The submitter refuses to replace
simulation output and records the exact binary checksum, its build revision,
and the current source revisions in `provenance.txt`.  An environment-only
retry reuses the validated staged executable even if the developer worktree
has since rebuilt its default binary.

Successful stages leave `.STAGE.complete` markers in the GPFS work root.  Set
`V4_WORKROOT` only to choose another dedicated GPFS directory; the runner
accepts only a dedicated directory below `/gpfs/kjhan`.
