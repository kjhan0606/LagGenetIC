// This file is part of monofonIC (MUSIC2)
// Slab <-> KSection cell redistributor via MPI_Alltoallv.
//
// Scope: real-space grid only. The plan (send/recv counts + perms) is
// computed once at construction; payload transfers are O(local cells) +
// one (or K, when per-rank totals exceed INT_MAX) Alltoallv per call.
// Roundtrip slab->ksec->slab is bit-exact.
#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(USE_MPI)
#include <mpi.h>
#endif

#include "mpi_ksection.hh"

namespace ksection {

//! Description of one rank's slab in the FFTW3-MPI real-space layout.
//!   stride_z: contiguous stride in the innermost dim. For tightly-packed
//!   buffers this equals n_global[2]. For Grid_FFT<real_t,true> in real
//!   space the in-place FFT padding makes it n_global[2] + 2.
struct SlabDescriptor {
    std::array<int, 3> n_global; //!< {Nx, Ny, Nz}
    int local_0_start = 0;       //!< global x-index of first slab row on this rank
    int local_0_size  = 0;       //!< number of x-slabs on this rank
    int stride_z      = 0;       //!< inner-dim stride (>= n_global[2])

    int n_y() const { return n_global[1]; }
    int n_z() const { return n_global[2]; }
    long long local_cells() const {
        return static_cast<long long>(local_0_size) * n_global[1] * n_global[2];
    }
    long long slab_index(int i, int j, int k) const {
        // i is local (0..local_0_size-1), j,k are global within (0..Ny-1, 0..Nz-1)
        return (static_cast<long long>(i) * n_global[1] + j) * stride_z + k;
    }
};

namespace detail {
#if defined(USE_MPI)
template <typename T> inline MPI_Datatype mpi_type();
template <> inline MPI_Datatype mpi_type<float>()       { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_type<double>()      { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_type<long double>() { return MPI_LONG_DOUBLE; }
template <> inline MPI_Datatype mpi_type<int>()         { return MPI_INT; }
#endif

//! Safety margin below INT_MAX for per-rank Alltoallv buffer size. Using
//! INT_MAX/2 leaves room for displs+counts arithmetic without wraparound.
inline constexpr std::size_t kAlltoallvSafeMax =
    static_cast<std::size_t>(INT_MAX) / 2;
} // namespace detail

class SlabRedistributor {
public:
#if defined(USE_MPI)
    SlabRedistributor(const SlabDescriptor& slab,
                      const KSectionTree& tree,
                      MPI_Comm comm = MPI_COMM_WORLD);
#else
    SlabRedistributor(const SlabDescriptor& slab,
                      const KSectionTree& tree);
#endif

    //! Move slab-layout cells to ksec layout (this rank's owned cells).
    //! Output buffer is resized to `n_recv()`. Optional index buffers receive
    //! the global (ix,iy,iz) of each delivered cell, in recv order.
    template <typename T>
    void slab_to_ksec(const T* slab_data,
                      std::vector<T>& out_data,
                      std::vector<int>* out_gix = nullptr,
                      std::vector<int>* out_giy = nullptr,
                      std::vector<int>* out_giz = nullptr) const;

    //! Reverse direction. ksec_data must be sized n_recv() and in the same
    //! order as slab_to_ksec delivered it (i.e. recv-order). slab_data must
    //! be sized for the slab (local_0_size * Ny * stride_z).
    template <typename T>
    void ksec_to_slab(const T* ksec_data, T* slab_data) const;

    std::size_t n_recv() const { return n_recv_total_; }
    std::size_t n_send() const { return n_send_total_; }
    const SlabDescriptor& slab() const { return slab_; }
    const std::vector<int>& recv_gix() const { return recv_gix_; }
    const std::vector<int>& recv_giy() const { return recv_giy_; }
    const std::vector<int>& recv_giz() const { return recv_giz_; }
    //! Number of Alltoallv rounds needed (1 unless per-rank totals exceed INT_MAX).
    int num_chunks() const { return n_chunks_; }

private:
    SlabDescriptor slab_;
    int myrank_ = 0;
    int nranks_ = 1;
#if defined(USE_MPI)
    MPI_Comm comm_ = MPI_COMM_NULL;
#endif

    // Plan: per-peer counts in items; cumulative displs are recomputed
    // per chunk inside the transfer helpers and so are not stored.
    std::vector<std::size_t> send_counts_;  // per dest rank, in items
    std::vector<std::size_t> recv_counts_;
    std::vector<std::size_t> send_displs_;  // cumulative offsets into the full peer-major sendbuf
    std::vector<std::size_t> recv_displs_;
    std::vector<std::uint32_t> send_perm_;  // [N_total_send] -> local slab linear index
    std::size_t n_send_total_ = 0;
    std::size_t n_recv_total_ = 0;
    int n_chunks_ = 1;                      // determined at construction (>=1)
    std::vector<int> recv_gix_, recv_giy_, recv_giz_;

#if defined(USE_MPI)
    //! Chunked MPI_Alltoallv on a peer-major buffer. `send_full` is sized
    //! to sum(send_cnts) and `recv_full` is resized to sum(recv_cnts).
    //! Internally splits into n_chunks_ rounds so per-round per-rank items
    //! fit in int (counts/displs are int in the MPI-3 interface).
    template <typename U>
    void chunked_alltoallv(const U* send_full,
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
void SlabRedistributor::chunked_alltoallv(
    const U* send_full,
    const std::vector<std::size_t>& send_cnts,
    const std::vector<std::size_t>& send_disps,
    U* recv_full,
    const std::vector<std::size_t>& recv_cnts,
    const std::vector<std::size_t>& recv_disps) const
{
    const int K = n_chunks_;
    // Fast path: single chunk — call MPI_Alltoallv directly with no extra
    // packing buffers. This is the common case (per-rank totals < INT_MAX/2).
    if (K == 1) {
        std::vector<int> scounts(nranks_), rcounts(nranks_);
        std::vector<int> sdispls(nranks_), rdispls(nranks_);
        for (int d = 0; d < nranks_; ++d) {
            scounts[d] = static_cast<int>(send_cnts[d]);
            rcounts[d] = static_cast<int>(recv_cnts[d]);
            sdispls[d] = static_cast<int>(send_disps[d]);
            rdispls[d] = static_cast<int>(recv_disps[d]);
        }
        MPI_Alltoallv(const_cast<U*>(send_full), scounts.data(), sdispls.data(),
                      detail::mpi_type<U>(),
                      recv_full, rcounts.data(), rdispls.data(),
                      detail::mpi_type<U>(), comm_);
        return;
    }

    // Per-round per-peer slice size (ceil-divide so any extra goes early).
    std::vector<std::size_t> send_chunk(nranks_), recv_chunk(nranks_);
    for (int d = 0; d < nranks_; ++d) {
        send_chunk[d] = (send_cnts[d] + K - 1) / K;
        recv_chunk[d] = (recv_cnts[d] + K - 1) / K;
    }
    std::vector<int> rcounts(nranks_, 0), rdispls(nranks_, 0);
    std::vector<int> scounts(nranks_, 0), sdispls(nranks_, 0);
    std::vector<U> sbuf, rbuf;

    for (int k = 0; k < K; ++k) {
        // Compute this round's per-peer counts and pack sendbuf peer-major.
        std::size_t s_round_total = 0, r_round_total = 0;
        for (int d = 0; d < nranks_; ++d) {
            const std::size_t s_off = k * send_chunk[d];
            const std::size_t s_cnt = (s_off >= send_cnts[d])
                ? 0 : std::min(send_chunk[d], send_cnts[d] - s_off);
            const std::size_t r_off = k * recv_chunk[d];
            const std::size_t r_cnt = (r_off >= recv_cnts[d])
                ? 0 : std::min(recv_chunk[d], recv_cnts[d] - r_off);
            scounts[d] = static_cast<int>(s_cnt);
            rcounts[d] = static_cast<int>(r_cnt);
            sdispls[d] = static_cast<int>(s_round_total);
            rdispls[d] = static_cast<int>(r_round_total);
            s_round_total += s_cnt;
            r_round_total += r_cnt;
        }
        sbuf.assign(std::max<std::size_t>(1, s_round_total), U{});
        rbuf.assign(std::max<std::size_t>(1, r_round_total), U{});
        // Pack send slices.
        for (int d = 0; d < nranks_; ++d) {
            const std::size_t s_off = k * send_chunk[d];
            const std::size_t s_cnt = static_cast<std::size_t>(scounts[d]);
            const U* src = send_full + send_disps[d] + s_off;
            std::copy(src, src + s_cnt, sbuf.data() + sdispls[d]);
        }
        MPI_Alltoallv(sbuf.data(), scounts.data(), sdispls.data(),
                      detail::mpi_type<U>(),
                      rbuf.data(), rcounts.data(), rdispls.data(),
                      detail::mpi_type<U>(), comm_);
        // Unpack recv slices.
        for (int d = 0; d < nranks_; ++d) {
            const std::size_t r_off = k * recv_chunk[d];
            const std::size_t r_cnt = static_cast<std::size_t>(rcounts[d]);
            U* dst = recv_full + recv_disps[d] + r_off;
            const U* src = rbuf.data() + rdispls[d];
            std::copy(src, src + r_cnt, dst);
        }
    }
}
#endif

template <typename T>
void SlabRedistributor::slab_to_ksec(const T* slab_data,
                                     std::vector<T>& out_data,
                                     std::vector<int>* out_gix,
                                     std::vector<int>* out_giy,
                                     std::vector<int>* out_giz) const {
    std::vector<T> sendbuf(std::max<std::size_t>(1, n_send_total_));
    for (std::size_t s = 0; s < n_send_total_; ++s) {
        sendbuf[s] = slab_data[send_perm_[s]];
    }
    out_data.assign(std::max<std::size_t>(1, n_recv_total_), T{});

#if defined(USE_MPI)
    chunked_alltoallv(sendbuf.data(), send_counts_, send_displs_,
                      out_data.data(), recv_counts_, recv_displs_);
#else
    // nranks_==1: send buffer equals recv buffer in order.
    for (std::size_t s = 0; s < n_send_total_; ++s) out_data[s] = sendbuf[s];
#endif

    if (n_recv_total_ == 0) out_data.clear();
    else                    out_data.resize(n_recv_total_);

    if (out_gix) *out_gix = recv_gix_;
    if (out_giy) *out_giy = recv_giy_;
    if (out_giz) *out_giz = recv_giz_;
}

template <typename T>
void SlabRedistributor::ksec_to_slab(const T* ksec_data, T* slab_data) const {
    // Send back: counts/displs roles swap.
    std::vector<T> sendbuf;
    if (n_recv_total_ > 0)
        sendbuf.assign(ksec_data, ksec_data + n_recv_total_);
    else
        sendbuf.assign(1, T{});

    std::vector<T> recvbuf(std::max<std::size_t>(1, n_send_total_), T{});
#if defined(USE_MPI)
    chunked_alltoallv(sendbuf.data(), recv_counts_, recv_displs_,
                      recvbuf.data(), send_counts_, send_displs_);
#else
    for (std::size_t s = 0; s < n_send_total_; ++s) recvbuf[s] = sendbuf[s];
#endif

    for (std::size_t s = 0; s < n_send_total_; ++s) {
        slab_data[send_perm_[s]] = recvbuf[s];
    }
}

} // namespace ksection
