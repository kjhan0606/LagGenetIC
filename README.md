# LagGenetIC

Combined repository containing local modifications to two cosmological
IC-generation codes used by the VoidSim project:

- `monofonIC/` — fork of [cosmo-sims/monofonIC](https://github.com/cosmo-sims/monofonIC).
  Adds a cuRamses-style **k-section recursive domain decomposition** for
  particle generation (gated by `setup/UseKSectionParticles=yes`),
  covering SC/BCC/FCC/RSC and masked-SC lattices with or without baryons.
  Supersedes the slab-only FFTW3-MPI particle path for these lattice
  types; the slab path remains active for glass and grid-only outputs.
- `genetIC/` — fork of [pynbody/genetIC](https://github.com/pynbody/genetIC).
  Local modifications to bindings, dummyic, ic, and particle species
  bookkeeping (see `git diff` against upstream).

Each subdirectory has its own build system and documentation; see the
README inside each for build/usage instructions.

Upstream commits at the time of fresh-start:
- monofonIC: based on `bca7fa4` (with KSectionHalo + masked support)
- genetIC: based on `b3622eb` (with the 4 local working-copy modifications)
