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

## Test

The synthetic regression exercises raw and Fortran HOP tags, multiple RAMSES
rank files, scrambled particle order, duplicate halo selection, overwrite
protection, and GenetIC's `id_file` parser:

```bash
python3 test_hop_to_genetic_id.py \
  --genetic-binary ../../genetIC/genetIC/genetIC
```

The test creates all inputs in a temporary directory and leaves the source tree
unchanged.
