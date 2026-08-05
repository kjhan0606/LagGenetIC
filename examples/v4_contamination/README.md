# VoidSim V4: contamination and buffer growth

The V4 tools apply the zoom-contamination gate to a RAMSES particle snapshot.
The default gate requires:

- zero low-resolution dark matter particles within `2 R_v`;
- a low-resolution dark matter mass fraction below `1e-3` within `5 R_v`;
- at least one finest-mass particle within the outer aperture.

`R_v` denotes the measured Eulerian void radius. The center and radius must use
the same coordinate units as the RAMSES particle positions. The box size is
read from `info_*.txt` unless `--box-size` is supplied. Distances use the
periodic minimum image.

The tool reads one rank file at a time. A first pass finds the smallest dark
matter particle mass. A second pass measures both apertures, which keeps memory
use bounded by the largest individual rank file. Legacy lagRamses particle
types and current RAMSES family/tag records are detected automatically.

Run the measurement with:

```bash
python3 measure_contamination.py output_00042 \
  --center 0.51 0.73 0.57 \
  --void-radius 0.03 \
  --json metric_shell00.json
```

A passing measurement exits with status 0. A failed gate exits with status 1.
The JSON file records the particle mass spectrum, aperture counts, mass
fractions, thresholds, and failure reasons.

The same command can prepare the next Lagrangian mask after a failed gate:

```bash
python3 measure_contamination.py output_00042 \
  --center 0.51 0.73 0.57 \
  --void-radius 0.03 \
  --json metric_shell00.json \
  --mask halo.id \
  --next-mask halo_shell01.id \
  --grid-size 512
```

One failure adds one periodic Chebyshev shell on the parent grid. The output is
a sorted, unique, zero-based GenetIC `id_file`. Rerun GenetIC and lagRamses
with the new mask and repeat the measurement until the command exits with
status 0. More than one shell can be added with `--grow-shells`.

Mask growth can also be run independently:

```bash
python3 buffer_mask.py halo.id halo_shell01.id --grid-size 512
```

The focused synthetic regression covers legacy and modern particle layouts,
periodic aperture distances, both contamination thresholds, atomic JSON
output, and mask growth across all three periodic boundaries:

```bash
python3 test_v4_contamination.py
```
