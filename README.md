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

Upstream commits at the time of fresh-start:
- monofonIC: based on `bca7fa4` (with KSectionHalo + masked support);
  subsequent local commits add glass-on-k-section (Phase 4) and the
  `test_glass_consistency` regression test.
- genetIC: based on `b3622eb` (with the 4 local working-copy modifications)
