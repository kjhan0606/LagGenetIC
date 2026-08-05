# VoidSim V3: sign-inverted closed loop

This regression applies GenetIC's global `reverse` operation to a real
HOP-selected Lagrangian mask from V2. It generates matched normal and inverted
two-level GRAFIC initial conditions, ingests the inverted pair with lagRamses,
and verifies the coarse-to-fine particle replacement.

The largest V2 halo is unsuitable for this test because its Lagrangian envelope
crosses the periodic `z` boundary. The current GRAFIC union exporter requires a
non-wrapping zoom box. The fixed-seed reference test therefore uses HOP halo 21,
the most populated halo whose 2:1 zoom box remains inside all three parent
boundaries. It contains 554 parent particles.

Run V2 with `KEEP_V2_WORKDIR=1`, then pass that directory to V3:

```bash
RAMSES_BIN=/path/to/lagRamses/bin/ramses_final3d \
KEEP_V3_WORKDIR=1 \
./run_v3.sh /tmp/voidsim-v2-example
```

The reference result is:

- The base and zoom density fields satisfy
  `delta_inverted = -delta_normal` cell by cell with zero residual.
- The selected parent peak has mean
  `delta = 2.741336077452e-2`. The inverted value is its exact negative.
- The coarse refinement map contains exactly the 554 HOP member cells.
- lagRamses creates 1,532 level-7 grids after its one-cell buffer.
- The 1,532 replaced parent cells become 12,256 fine particles.
- The initial lagRamses snapshot contains 272,868 unique IDs in four MPI files.

The lagRamses namelist sets both `m_refine=0` and `ivar_refine=0`.
`m_refine` enables refinement evaluation, while `ivar_refine=0` selects the
GRAFIC `ic_refmap`. Omitting the latter refines the entire level instead of the
masked region.
