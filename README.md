# LagGenetIC

Combined repository containing local modifications to two cosmological
IC-generation codes used by the VoidSim project:

- `monofonIC/` — fork of [cosmo-sims/monofonIC](https://github.com/cosmo-sims/monofonIC).
  Adds a cuRamses-style **k-section recursive domain decomposition** for
  particle generation (gated by `setup/UseKSectionParticles=yes`),
  covering SC/BCC/FCC/RSC, masked-SC, and glass lattices with or
  without baryons (glass: no per-particle masses, matching the slab
  path). Supersedes the slab-only FFTW3-MPI particle path for these
  lattice types; the slab path remains active for grid-only outputs
  (`grafic2`, `field_lagrangian`, `field_eulerian`). See
  [`monofonIC/README.md`](monofonIC/README.md#k-section-particle-decomposition-local-extension)
  for usage. Slab vs k-section bit-identity is exercised in CI via
  the `test_glass_consistency` ctest target (and analogous smoke
  tests for the Bravais and masked-SC paths).
- `genetIC/` — fork of [pynbody/genetIC](https://github.com/pynbody/genetIC).
  Local modifications to bindings, dummyic, ic, and particle species
  bookkeeping (see `git diff` against upstream).

Each subdirectory has its own build system and documentation; see the
README inside each for build/usage instructions.

## Power-spectrum normalisation

The bundled monofonIC and GenetIC can normalise a tabulated CAMB transfer
function with either the late-time amplitude `sigma_8` or the primordial
amplitude `A_s`. Explicit `A_s` and `sigma_8` values are mutually exclusive.
The pivot `k_p` is given in physical Mpc^-1.

For monofonIC, select the primordial-amplitude path in the cosmology section:

```ini
[cosmology]
A_s = 2.1005e-9
k_p = 0.05
transfer = CAMB_file
transfer_file = camb_transfer.dat
```

Use `sigma_8 = ...` instead of `A_s` for numerical late-time normalisation.
A named `ParameterSet` supplies its tabulated `A_s` unless the configuration
explicitly overrides the normalisation.

For GenetIC, set exactly one amplitude before the `camb` command:

```text
A_s 2.1005e-9
k_p 0.05
camb camb_transfer.dat
```

The aliases `As` and `kpivot` are also accepted. The `A_s` path uses the
redshift already represented by the imported CAMB table, so the table must
correspond to the redshift required by the IC workflow. The legacy syntax
uses `s8 <value>` in place of `A_s`.

Focused regression tests live in
`genetIC/tests/test_As_normalization/` and in the monofonIC CTest target
`test_transfer_CAMB_file_As_normalization`.

## VoidSim full-stack validation

`examples/v1_id_roundtrip/` contains the reproducible 128^3 V1 regression for
the monofonIC -> GenetIC -> lagRamses particle-ID contract. It runs both the
classical slab and k-section monofonIC particle paths, checks that their parent
white-noise dumps and GenetIC grafic fields are bit-identical, then verifies
exact ID and initial-position preservation through four-rank lagRamses runs.

`examples/v2_hop_id_file/` provides the next bridge in the zoom workflow. It
joins HOP group tags to the particle ordering of a DMO RAMSES parent snapshot
and writes the selected zero-based Lagrangian IDs in GenetIC `id_file` format.
The converter validates the full `0..N^3-1` ID permutation before it writes a
mask. Its regression covers raw and Fortran-sequential HOP tags and an actual
GenetIC `id_file` round trip. The evolved-parent integration advances a
64-cubed DMO parent beyond `a=1`, finds the most populated HOP halo, and checks
its compact Lagrangian envelope in the initial snapshot.

`examples/v3_inverted_closed_loop/` applies `reverse` to a non-wrapping V2
halo mask and generates matched normal and inverted two-level GRAFIC ICs. The
regression verifies exact cellwise sign inversion, masked lagRamses refinement,
coarse-particle replacement, and unique Lagrangian IDs.

`examples/v4_contamination/` measures low-resolution dark matter contamination
within periodic `2 R_v` and `5 R_v` apertures. A failed gate can atomically
write the next GenetIC `id_file` after adding a periodic parent-grid shell.
The reader streams rank files and supports both legacy lagRamses particle types
and current RAMSES family/tag records.

Upstream commits at the time of fresh-start:
- monofonIC: based on `bca7fa4` (with KSectionHalo + masked support);
  subsequent local commits add glass-on-k-section (Phase 4) and the
  `test_glass_consistency` regression test.
- genetIC: based on `b3622eb` (with the 4 local working-copy modifications)
