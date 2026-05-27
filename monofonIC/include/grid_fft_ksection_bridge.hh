// This file is part of monofonIC (MUSIC2)
// Bridge: Grid_FFT<T,true> (FFTW3-MPI slab, real-space) <-> Grid_KSection<T>.
//
// Wraps SlabRedistributor with awareness of the in-place real-FFT padding
// (stride_z = n_[2] + 2 for real grids in real space).
#pragma once

#include <array>
#include <stdexcept>

#include "grid_fft.hh"
#include "grid_ksection.hh"
#include "mpi_ksection.hh"
#include "mpi_ksection_redistribute.hh"

namespace ksection {

//! Build a SlabDescriptor matching a Grid_FFT<T,true> currently in real space.
template <typename T, bool D>
SlabDescriptor make_slab_descriptor(const Grid_FFT<T, D>& g) {
    if (g.space_ != rspace_id)
        throw std::logic_error(
            "make_slab_descriptor: Grid_FFT must be in real space");
    SlabDescriptor d;
    d.n_global    = {static_cast<int>(g.n_[0]),
                     static_cast<int>(g.n_[1]),
                     static_cast<int>(g.n_[2])};
    d.local_0_start = static_cast<int>(g.local_0_start_);
    d.local_0_size  = static_cast<int>(g.local_0_size_);
    d.stride_z      = static_cast<int>(g.sizes_[3]); // npr_ for real grids
    return d;
}

//! Forward: pull the real-space slab data out of `src` into `dst` (Grid_KSection).
//! Cell indices remain owned by `redist` (recv_gix/giy/giz); only values are
//! written into `dst`, so callers must read indices via the redistributor.
template <typename T, bool D>
void redistribute_slab_to_ksec(const Grid_FFT<T, D>& src,
                               const SlabRedistributor& redist,
                               Grid_KSection<T>& dst) {
    dst.set_meta({static_cast<int>(src.n_[0]),
                  static_cast<int>(src.n_[1]),
                  static_cast<int>(src.n_[2])},
                 {static_cast<double>(src.length_[0]),
                  static_cast<double>(src.length_[1]),
                  static_cast<double>(src.length_[2])});
    redist.slab_to_ksec(src.data_, dst.data_vec());
}

//! Backward: push ksec cells back into the matching real-space slab `dst`.
template <typename T, bool D>
void redistribute_ksec_to_slab(const Grid_KSection<T>& src,
                               const SlabRedistributor& redist,
                               Grid_FFT<T, D>& dst) {
    if (dst.space_ != rspace_id)
        throw std::logic_error(
            "redistribute_ksec_to_slab: dst Grid_FFT must be in real space");
    redist.ksec_to_slab(src.data(), dst.data_);
}

} // namespace ksection
