// This file is part of monofonIC (MUSIC2)
// Build the slab->ksec redistribution plan.
#include "mpi_ksection_redistribute.hh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ksection {

#if defined(USE_MPI)
SlabRedistributor::SlabRedistributor(const SlabDescriptor& slab,
                                     const KSectionTree& tree,
                                     MPI_Comm comm)
    : slab_(slab), comm_(comm)
{
    MPI_Comm_rank(comm_, &myrank_);
    MPI_Comm_size(comm_, &nranks_);
#else
SlabRedistributor::SlabRedistributor(const SlabDescriptor& slab,
                                     const KSectionTree& tree)
    : slab_(slab)
{
    myrank_ = 0; nranks_ = 1;
#endif
    if (tree.nproc() != nranks_) {
        throw std::invalid_argument(
            "SlabRedistributor: KSectionTree::nproc must equal MPI comm size");
    }
    if (slab_.stride_z < slab_.n_global[2])
        throw std::invalid_argument("SlabRedistributor: stride_z < n_global[2]");

    const int Ny = slab_.n_global[1];
    const int Nz = slab_.n_global[2];

    // Largest local slab linear index = (local_0_size-1)*Ny*stride_z + ...
    // <= local_0_size * Ny * stride_z. Must fit uint32_t for send_perm_.
    {
        const std::size_t max_idx =
            static_cast<std::size_t>(slab_.local_0_size)
          * static_cast<std::size_t>(slab_.n_global[1])
          * static_cast<std::size_t>(slab_.stride_z);
        if (max_idx > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error(
                "SlabRedistributor: local slab index exceeds uint32_t; "
                "increase rank count or rebuild with wider send_perm_");
        }
    }

    // Pass 1: count per-dest sends.
    send_counts_.assign(nranks_, 0);
    for (int i = 0; i < slab_.local_0_size; ++i) {
        const int ix = i + slab_.local_0_start;
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const int r = tree.cpu_for_cell(ix, j, k);
                send_counts_[r] += 1;
            }
        }
    }

    send_displs_.assign(nranks_, 0);
    for (int r = 1; r < nranks_; ++r)
        send_displs_[r] = send_displs_[r - 1] + send_counts_[r - 1];
    n_send_total_ = (nranks_ > 0)
        ? send_displs_[nranks_ - 1] + send_counts_[nranks_ - 1]
        : 0;

    // Pass 2: fill send_perm_ and (in parallel) per-rank global-index arrays
    // that will be Alltoallv'd to populate recv_g{ix,iy,iz}_ on the destination.
    send_perm_.assign(n_send_total_, 0u);
    std::vector<std::size_t> cursor(nranks_, 0);
    std::vector<int> send_gix(n_send_total_, 0);
    std::vector<int> send_giy(n_send_total_, 0);
    std::vector<int> send_giz(n_send_total_, 0);

    for (int i = 0; i < slab_.local_0_size; ++i) {
        const int ix = i + slab_.local_0_start;
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const int r = tree.cpu_for_cell(ix, j, k);
                const std::size_t slot = send_displs_[r] + cursor[r];
                cursor[r] += 1;
                send_perm_[slot] = static_cast<std::uint32_t>(slab_.slab_index(i, j, k));
                send_gix[slot] = ix;
                send_giy[slot] = j;
                send_giz[slot] = k;
            }
        }
    }

    // Exchange counts -> recv_counts_. MPI_Alltoall on plain ints is fine
    // because per-peer count fits int by construction (it's a partition of
    // local_0_size*Ny*Nz which itself <= uint32_t per the check above).
    recv_counts_.assign(nranks_, 0);
#if defined(USE_MPI)
    {
        std::vector<int> sc(nranks_), rc(nranks_);
        for (int r = 0; r < nranks_; ++r) sc[r] = static_cast<int>(send_counts_[r]);
        MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, comm_);
        for (int r = 0; r < nranks_; ++r) recv_counts_[r] = static_cast<std::size_t>(rc[r]);
    }
#else
    recv_counts_[0] = send_counts_[0];
#endif
    recv_displs_.assign(nranks_, 0);
    for (int r = 1; r < nranks_; ++r)
        recv_displs_[r] = recv_displs_[r - 1] + recv_counts_[r - 1];
    n_recv_total_ = (nranks_ > 0)
        ? recv_displs_[nranks_ - 1] + recv_counts_[nranks_ - 1]
        : 0;

    // Decide chunk count: agreed across all ranks so per-round per-rank
    // bytes fit in int (MPI-3 Alltoallv counts/displs are int).
    {
        std::size_t local_max = std::max(n_send_total_, n_recv_total_);
        std::size_t global_max = local_max;
#if defined(USE_MPI)
        MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, comm_);
#endif
        n_chunks_ = static_cast<int>(
            (global_max + detail::kAlltoallvSafeMax - 1) / detail::kAlltoallvSafeMax);
        if (n_chunks_ < 1) n_chunks_ = 1;
        // Test hook: bump K to exercise the chunked path at sizes where
        // the K=1 fast path would otherwise dominate. Honored only if the
        // env value is strictly greater than the natural K.
        if (const char* env = std::getenv("KSECTION_FORCE_CHUNKS")) {
            const int forced = std::atoi(env);
            if (forced > n_chunks_) n_chunks_ = forced;
        }
    }

    // Exchange global indices once so recv side knows what it's receiving.
    recv_gix_.assign(n_recv_total_, 0);
    recv_giy_.assign(n_recv_total_, 0);
    recv_giz_.assign(n_recv_total_, 0);

#if defined(USE_MPI)
    chunked_alltoallv(send_gix.data(), send_counts_, send_displs_,
                      recv_gix_.data(), recv_counts_, recv_displs_);
    chunked_alltoallv(send_giy.data(), send_counts_, send_displs_,
                      recv_giy_.data(), recv_counts_, recv_displs_);
    chunked_alltoallv(send_giz.data(), send_counts_, send_displs_,
                      recv_giz_.data(), recv_counts_, recv_displs_);
#else
    recv_gix_ = send_gix;
    recv_giy_ = send_giy;
    recv_giz_ = send_giz;
#endif
}

} // namespace ksection
