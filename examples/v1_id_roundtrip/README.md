# VoidSim V1: 128^3 full-stack particle-ID round trip

This regression exercises the production IC contract at the validation size:

1. monofonIC generates a 128^3 parent white-noise field twice, once with the
   classical slab particle path and once with `UseKSectionParticles=yes`.
2. The HDF5 white-noise dumps are converted to NumPy and imported by GenetIC
   with `import_level_as ... whitenoise`.
3. GenetIC writes single-level grafic ICs, including deterministic
   `ic_particle_ids`.
4. lagRamses reads each IC with four MPI ranks and writes an initial snapshot.
5. `verify_v1.py` checks white-noise and grafic bit identity, exact particle-ID
   multiset preservation, and ID-indexed particle positions.

The k-section switch applies to monofonIC's particle generation/output stage;
white-noise and other grid fields intentionally remain on the FFT slab path.
The test therefore also asserts that activating the particle redistribution
does not perturb the parent white noise handed to GenetIC.

Build both bundled IC codes, then run:

```bash
RAMSES_BIN=/path/to/ramses_final3d ./run_v1.sh
```

Useful overrides are `MONOFONIC_BIN`, `GENETIC_BIN`, `MPIEXEC`, `NP`,
`V1_WORKDIR`, and `KEEP_V1_WORKDIR=1`. The default temporary work directory is
deleted only after a successful or failed run and only when it was created by
`mktemp` inside the script.

