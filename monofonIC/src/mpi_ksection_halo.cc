// This file is part of monofonIC (MUSIC2)
// Build the +xyz 1-cell halo plan for a k-section block.
#include "mpi_ksection_halo.hh"

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>

namespace ksection {

#if defined(USE_MPI)
KSectionHalo::KSectionHalo(const SlabRedistributor& redist,
                           const KSectionTree& tree,
                           MPI_Comm comm)
    : comm_(comm)
{
    MPI_Comm_rank(comm_, &myrank_);
    MPI_Comm_size(comm_, &nranks_);
#else
KSectionHalo::KSectionHalo(const SlabRedistributor& redist,
                           const KSectionTree& tree)
{
    myrank_ = 0; nranks_ = 1;
#endif
    if (tree.nproc() != nranks_)
        throw std::invalid_argument(
            "KSectionHalo: KSectionTree::nproc must equal MPI comm size");

    const auto& gix = redist.recv_gix();
    const auto& giy = redist.recv_giy();
    const auto& giz = redist.recv_giz();
    const std::size_t n_core = gix.size();

    Nx_ = redist.slab().n_global[0];
    Ny_ = redist.slab().n_global[1];
    Nz_ = redist.slab().n_global[2];

    // 1. Derive my block bbox from the recv'd global indices.
    if (n_core == 0) {
        // Empty rank: extended buffer is 1x1x1, no exchanges. Pick a degenerate
        // base coord (0) and rely on extended_size()=1 reading harmlessly.
        Lx_ = Ly_ = Lz_ = 0;
        Xmin_ = Ymin_ = Zmin_ = 0;
        block_to_recv_perm_.clear();
        send_counts_.assign(nranks_, 0);
        send_displs_.assign(nranks_, 0);
        recv_counts_.assign(nranks_, 0);
        recv_displs_.assign(nranks_, 0);
        return;
    }
    int xmin = gix[0], xmax = gix[0];
    int ymin = giy[0], ymax = giy[0];
    int zmin = giz[0], zmax = giz[0];
    for (std::size_t s = 1; s < n_core; ++s) {
        if (gix[s] < xmin) xmin = gix[s]; else if (gix[s] > xmax) xmax = gix[s];
        if (giy[s] < ymin) ymin = giy[s]; else if (giy[s] > ymax) ymax = giy[s];
        if (giz[s] < zmin) zmin = giz[s]; else if (giz[s] > zmax) zmax = giz[s];
    }
    Xmin_ = xmin; Ymin_ = ymin; Zmin_ = zmin;
    Lx_ = xmax - xmin + 1;
    Ly_ = ymax - ymin + 1;
    Lz_ = zmax - zmin + 1;
    if (static_cast<std::size_t>(Lx_) * Ly_ * Lz_ != n_core)
        throw std::runtime_error(
            "KSectionHalo: ksec block is not axis-aligned rectangular "
            "(Lx*Ly*Lz != n_recv); halo plan requires k-section walls");

    // 2. Build block -> recv-slot permutation.
    block_to_recv_perm_.assign(static_cast<std::size_t>(Lx_) * Ly_ * Lz_, -1);
    for (std::size_t s = 0; s < n_core; ++s) {
        const int ei = gix[s] - Xmin_;
        const int ej = giy[s] - Ymin_;
        const int ek = giz[s] - Zmin_;
        block_to_recv_perm_[(ei * Ly_ + ej) * Lz_ + ek] = static_cast<std::int32_t>(s);
    }

    // 3. Enumerate halo extended cells and look up their owners.
    //    For each halo ext (ei,ej,ek), compute global (gix_h,giy_h,giz_h)
    //    with wraparound and ask the tree who owns it.
    struct Want { int owner; std::int32_t ext_idx; int gix, giy, giz; };
    std::vector<Want> wants;
    wants.reserve(static_cast<std::size_t>(Ly_ + 1) * (Lz_ + 1) +
                  static_cast<std::size_t>(Lx_ + 1) * (Lz_ + 1) +
                  static_cast<std::size_t>(Lx_ + 1) * (Ly_ + 1));
    for (int ei = 0; ei <= Lx_; ++ei) {
        for (int ej = 0; ej <= Ly_; ++ej) {
            for (int ek = 0; ek <= Lz_; ++ek) {
                if (ei < Lx_ && ej < Ly_ && ek < Lz_) continue; // core
                const int gx = (Xmin_ + ei) % Nx_;
                const int gy = (Ymin_ + ej) % Ny_;
                const int gz = (Zmin_ + ek) % Nz_;
                const int gx_w = (gx + Nx_) % Nx_;
                const int gy_w = (gy + Ny_) % Ny_;
                const int gz_w = (gz + Nz_) % Nz_;
                const int owner = tree.cpu_for_cell(gx_w, gy_w, gz_w);
                const std::int32_t flat =
                    static_cast<std::int32_t>(ext_index(ei, ej, ek));
                wants.push_back({owner, flat, gx_w, gy_w, gz_w});
            }
        }
    }

    // 4. Group wants by owner.
    recv_counts_.assign(nranks_, 0);
    for (const auto& w : wants) recv_counts_[w.owner] += 1;
    recv_displs_.assign(nranks_, 0);
    for (int r = 1; r < nranks_; ++r)
        recv_displs_[r] = recv_displs_[r - 1] + recv_counts_[r - 1];
    n_recv_total_ = recv_displs_[nranks_ - 1] + recv_counts_[nranks_ - 1];

    // Order wants peer-major and remember recv_perm (where to deposit) per slot.
    recv_perm_.assign(n_recv_total_, 0);
    std::vector<int> req_gix(n_recv_total_), req_giy(n_recv_total_), req_giz(n_recv_total_);
    {
        std::vector<std::size_t> cursor(nranks_, 0);
        for (const auto& w : wants) {
            const std::size_t slot = recv_displs_[w.owner] + cursor[w.owner]++;
            recv_perm_[slot] = w.ext_idx;
            req_gix[slot] = w.gix;
            req_giy[slot] = w.giy;
            req_giz[slot] = w.giz;
        }
    }

    // 5. Exchange counts (recv_counts -> peer's send_counts).
    send_counts_.assign(nranks_, 0);
#if defined(USE_MPI)
    {
        std::vector<int> rc(nranks_), sc(nranks_);
        for (int r = 0; r < nranks_; ++r) rc[r] = static_cast<int>(recv_counts_[r]);
        MPI_Alltoall(rc.data(), 1, MPI_INT, sc.data(), 1, MPI_INT, comm_);
        for (int r = 0; r < nranks_; ++r) send_counts_[r] = static_cast<std::size_t>(sc[r]);
    }
#else
    send_counts_[0] = recv_counts_[0];
#endif
    send_displs_.assign(nranks_, 0);
    for (int r = 1; r < nranks_; ++r)
        send_displs_[r] = send_displs_[r - 1] + send_counts_[r - 1];
    n_send_total_ = send_displs_[nranks_ - 1] + send_counts_[nranks_ - 1];

    // 6. Exchange requested global coords so peers know which of their cells
    //    we want. Three int Alltoallvs.
    std::vector<int> in_gix(std::max<std::size_t>(1, n_send_total_), 0);
    std::vector<int> in_giy(std::max<std::size_t>(1, n_send_total_), 0);
    std::vector<int> in_giz(std::max<std::size_t>(1, n_send_total_), 0);
#if defined(USE_MPI)
    halo_alltoallv(req_gix.data(), recv_counts_, recv_displs_,
                   in_gix.data(), send_counts_, send_displs_);
    halo_alltoallv(req_giy.data(), recv_counts_, recv_displs_,
                   in_giy.data(), send_counts_, send_displs_);
    halo_alltoallv(req_giz.data(), recv_counts_, recv_displs_,
                   in_giz.data(), send_counts_, send_displs_);
#else
    for (std::size_t s = 0; s < n_send_total_; ++s) {
        in_gix[s] = req_gix[s];
        in_giy[s] = req_giy[s];
        in_giz[s] = req_giz[s];
    }
#endif

    // 7. For each cell my peers want from me, look up the redist recv slot.
    send_perm_.assign(n_send_total_, -1);
    for (std::size_t s = 0; s < n_send_total_; ++s) {
        const int ei = in_gix[s] - Xmin_;
        const int ej = in_giy[s] - Ymin_;
        const int ek = in_giz[s] - Zmin_;
        if (ei < 0 || ei >= Lx_ || ej < 0 || ej >= Ly_ || ek < 0 || ek >= Lz_)
            throw std::runtime_error(
                "KSectionHalo: peer requested a cell I don't own; "
                "ksec partition is inconsistent with halo plan");
        send_perm_[s] = block_to_recv_perm_[(ei * Ly_ + ej) * Lz_ + ek];
    }
}

} // namespace ksection
