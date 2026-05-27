// This file is part of monofonIC (MUSIC2)
// One-cell positive-direction halo exchange on top of a k-section
// redistribute plan. Each rank's ksec block (axis-aligned cuboid) is
// extended by +1 cell along +x/+y/+z so callers can read 2x2x2 stencils
// or CIC kernels that span the block boundary without per-cell MPI.
//
// Halo cells map to up to 7 distinct extended regions (3 faces, 3 edges,
// 1 corner). Each halo cell's owner is queried via KSectionTree, so the
// plan handles non-aligned k-section partitions.
#pragma once

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(USE_MPI)
#include <mpi.h>
#endif

#include "mpi_ksection.hh"
#include "mpi_ksection_redistribute.hh"

namespace ksection {

class KSectionHalo {
public:
#if defined(USE_MPI)
    KSectionHalo(const SlabRedistributor& redist,
                 const KSectionTree& tree,
                 MPI_Comm comm = MPI_COMM_WORLD);
#else
    KSectionHalo(const SlabRedistributor& redist,
                 const KSectionTree& tree);
#endif

    int Lx() const { return Lx_; }
    int Ly() const { return Ly_; }
    int Lz() const { return Lz_; }
    int Xmin() const { return Xmin_; }
    int Ymin() const { return Ymin_; }
    int Zmin() const { return Zmin_; }

    //! Extended-buffer size: (Lx+1)*(Ly+1)*(Lz+1).
    std::size_t extended_size() const {
        return static_cast<std::size_t>(Lx_ + 1) * (Ly_ + 1) * (Lz_ + 1);
    }

    //! Flat index in the extended buffer for extended-local (ei, ej, ek).
    //! Valid for ei in [0,Lx], ej in [0,Ly], ek in [0,Lz].
    std::size_t ext_index(int ei, int ej, int ek) const {
        return (static_cast<std::size_t>(ei) * (Ly_ + 1) + ej) * (Lz_ + 1) + ek;
    }

    //! Fill the extended buffer from ksec_core (in SlabRedistributor recv order):
    //! first packs the core (ext cells with ei<Lx,ej<Ly,ek<Lz), then exchanges
    //! halo cells from neighbor ranks. ext_buf is resized to extended_size().
    template <typename T>
    void update_halo(const T* ksec_core, std::vector<T>& ext_buf) const;

    //! Reverse mapping: extended-local (ei, ej, ek) -> ksec recv slot, valid
    //! only for core cells (ei<Lx,ej<Ly,ek<Lz). Useful for direct lookups.
    int recv_slot_of_core(int ei, int ej, int ek) const {
        return block_to_recv_perm_[(ei * Ly_ + ej) * Lz_ + ek];
    }

private:
    int Lx_ = 0, Ly_ = 0, Lz_ = 0;
    int Xmin_ = 0, Ymin_ = 0, Zmin_ = 0;
    int Nx_ = 0, Ny_ = 0, Nz_ = 0;
    int myrank_ = 0;
    int nranks_ = 1;
#if defined(USE_MPI)
    MPI_Comm comm_ = MPI_COMM_NULL;
#endif

    // Maps block-local (ei*Ly+ej)*Lz+ek -> SlabRedistributor recv slot.
    std::vector<std::int32_t> block_to_recv_perm_;

    // Send plan: which of MY recv-slot cells to ship to each peer.
    std::vector<std::size_t> send_counts_, send_displs_;
    std::vector<std::int32_t> send_perm_;  // [N_total_send] -> redist recv slot

    // Recv plan: where to deposit each incoming cell in the extended buffer.
    std::vector<std::size_t> recv_counts_, recv_displs_;
    std::vector<std::int32_t> recv_perm_;  // [N_total_recv] -> ext_index

    std::size_t n_send_total_ = 0;
    std::size_t n_recv_total_ = 0;

#if defined(USE_MPI)
    template <typename U>
    void halo_alltoallv(const U* send_full,
                        const std::vector<std::size_t>& send_cnts,
                        const std::vector<std::size_t>& send_disps,
                        U* recv_full,
                        const std::vector<std::size_t>& recv_cnts,
                        const std::vector<std::size_t>& recv_disps) const;
#endif
};

// ----------------------- templated method definitions ----------------------

#if defined(USE_MPI)
template <typename U>
void KSectionHalo::halo_alltoallv(
    const U* send_full,
    const std::vector<std::size_t>& send_cnts,
    const std::vector<std::size_t>& send_disps,
    U* recv_full,
    const std::vector<std::size_t>& recv_cnts,
    const std::vector<std::size_t>& recv_disps) const
{
    // Halo volumes are O(face area), so they always fit in int per-rank in
    // realistic configurations. We still convert defensively.
    std::vector<int> scounts(nranks_), rcounts(nranks_);
    std::vector<int> sdispls(nranks_), rdispls(nranks_);
    for (int d = 0; d < nranks_; ++d) {
        if (send_cnts[d] > static_cast<std::size_t>(INT_MAX) ||
            recv_cnts[d] > static_cast<std::size_t>(INT_MAX))
            throw std::runtime_error("KSectionHalo: per-peer halo count exceeds int");
        scounts[d] = static_cast<int>(send_cnts[d]);
        rcounts[d] = static_cast<int>(recv_cnts[d]);
        sdispls[d] = static_cast<int>(send_disps[d]);
        rdispls[d] = static_cast<int>(recv_disps[d]);
    }
    MPI_Alltoallv(const_cast<U*>(send_full), scounts.data(), sdispls.data(),
                  detail::mpi_type<U>(),
                  recv_full, rcounts.data(), rdispls.data(),
                  detail::mpi_type<U>(), comm_);
}
#endif

template <typename T>
void KSectionHalo::update_halo(const T* ksec_core, std::vector<T>& ext_buf) const {
    ext_buf.assign(extended_size(), T{});

    // 1. Pack core cells into the extended buffer.
    for (int ei = 0; ei < Lx_; ++ei) {
        for (int ej = 0; ej < Ly_; ++ej) {
            for (int ek = 0; ek < Lz_; ++ek) {
                const int recv_slot = block_to_recv_perm_[(ei * Ly_ + ej) * Lz_ + ek];
                ext_buf[ext_index(ei, ej, ek)] = ksec_core[recv_slot];
            }
        }
    }

    if (n_send_total_ == 0 && n_recv_total_ == 0) return;

    // 2. Build send buffer from my recv-order data.
    std::vector<T> sendbuf(std::max<std::size_t>(1, n_send_total_));
    for (std::size_t s = 0; s < n_send_total_; ++s) {
        sendbuf[s] = ksec_core[send_perm_[s]];
    }
    std::vector<T> recvbuf(std::max<std::size_t>(1, n_recv_total_), T{});

#if defined(USE_MPI)
    halo_alltoallv(sendbuf.data(), send_counts_, send_displs_,
                   recvbuf.data(), recv_counts_, recv_displs_);
#else
    for (std::size_t s = 0; s < n_send_total_; ++s) recvbuf[s] = sendbuf[s];
#endif

    // 3. Unpack received cells into the halo region of ext_buf.
    for (std::size_t r = 0; r < n_recv_total_; ++r) {
        ext_buf[recv_perm_[r]] = recvbuf[r];
    }
}

} // namespace ksection
