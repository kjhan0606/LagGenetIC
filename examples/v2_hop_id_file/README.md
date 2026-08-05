# VoidSim V2: HOP halo membership to GenetIC `id_file`

This directory contains the V2 bridge from a dark-matter-only RAMSES parent
simulation to the Lagrangian mask consumed by GenetIC.

The workflow is:

1. Run HOP on the RAMSES parent snapshot.
2. Choose one or more one-based halo identifiers from the HOP `.pos` catalogue.
3. Join the particle-order HOP `.tag` array to the RAMSES particle IDs.
4. Write the selected initial-grid IDs as a sorted GenetIC `id_file`.

The converter accepts both raw and Fortran-sequential HOP tag files. It reads
RAMSES particle files in rank order and supports 32-bit and 64-bit particle IDs.
It refuses baryonic snapshots because the bundled RAMSES HOP reader predates the
current particle-family layout. The intended parent run for this protocol is DMO.

## Example

Run the bundled RAMSES HOP utilities on a DMO output. The converter consumes
the regrouped catalogue, not the intermediate `.hop` file:

```bash
hop -in output_00050/part_00050.out -p 1. -o void_parent_hop
regroup -root void_parent_hop -douter 80. -dsaddle 200. -dpeak 240. \
  -f77 -o void_parent_groups
poshalo -inp output_00050 -pre void_parent_groups
```

Convert HOP halo 27 to a GenetIC mask for a 128-cubed parent grid:

```bash
python3 hop_to_genetic_id.py \
  /path/to/output_00050 \
  /path/to/void_parent_groups.tag \
  halo_27.id \
  --halo-id 27 \
  --grid-size 128
```

Several HOP groups can be unioned into one mask by repeating `--halo-id`:

```bash
python3 hop_to_genetic_id.py \
  /path/to/output_00050 \
  /path/to/void_parent_groups.tag \
  halo_complex.id \
  --halo-id 27 --halo-id 31 --halo-id 42 \
  --grid-size 128
```

The output contains one zero-based initial-grid particle ID per line. GenetIC
loads it with:

```text
id_file halo_27.id
```

The converter verifies that the parent snapshot contains exactly `N^3` distinct
IDs spanning `0` through `N^3-1`. This check prevents a HOP particle-order mismatch
from silently producing an invalid zoom mask.

## Synthetic test

The synthetic regression exercises raw and Fortran HOP tags, multiple RAMSES
rank files, scrambled particle order, duplicate halo selection, overwrite
protection, and GenetIC's `id_file` parser:

```bash
python3 test_hop_to_genetic_id.py \
  --genetic-binary ../../genetIC/genetIC/genetIC
```

The test creates all inputs in a temporary directory and leaves the source tree
unchanged.

## Evolved-parent integration

`run_v2_evolved_parent.sh` closes the V2 loop with a nonlinear DMO parent. It
generates a 64-cubed white-noise field, writes GRAFIC initial conditions,
evolves the fixed-grid parent beyond `a=1`, builds the bundled HOP utilities in
the temporary work directory, selects the most populated halo, and traces its
member IDs to the initial snapshot. The final check also sends the mask through
GenetIC's `id_file` reader and writer.

Set the lagRamses source root before running:

```bash
RAMSES_ROOT=/path/to/lagRamses \
KEEP_V2_WORKDIR=1 \
./run_v2_evolved_parent.sh
```

The default decomposition uses 16 MPI ranks for monofonIC and four MPI ranks
for lagRamses. The 64-cubed evolution is too small to amortize lagRamses
communication over 16 ranks. Override these choices with `NP` and `RAMSES_NP`.
In the reference environment, a coarse lagRamses step took about 30 seconds on
16 ranks and 8 seconds on four ranks. Four ranks were about 3.8 times faster.

The reference run on 2026-08-05 found 138 HOP haloes. Its largest halo contains
2,528 of 262,144 particles. The selected mask passed an exact GenetIC round
trip. Its periodic Lagrangian envelope widths, in units of the parent box, are
`(0.279315, 0.263687, 0.355837)`.
