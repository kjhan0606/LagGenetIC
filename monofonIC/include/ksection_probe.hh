// This file is part of monofonIC (MUSIC2)
// Configurable end-to-end probe for the k-section redistribution machinery.
// When `setup/EnableKSectionProbe = yes` in the IC config, runs a
// slab->ksec->slab roundtrip on a ramp field of the same shape as `tmp`,
// verifies bit-exact recovery, and reports per-rank cell counts + timing.
#pragma once

#include "grid_fft.hh"
#include "config_file.hh"

namespace ksection {

//! Run the probe iff `setup/EnableKSectionProbe` is true. No-op otherwise.
//! `tmp` is used only as a layout template; its data is not modified on exit
//! (probe restores zero).
void run_probe(Grid_FFT<real_t>& tmp, config_file& the_config);

} // namespace ksection
