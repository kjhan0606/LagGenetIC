// This file is part of monofonIC (MUSIC2)
// Glass-particle loader for the k-section path. Mirrors particle::glass_loader
// but:
//   - domain-decomposes particles by full 3D ksec bbox (KSectionTree::cpu_for_point),
//     not just x-slab;
//   - reads CIC values from an extended halo buffer (KSectionHalo extended grid),
//     using positions in cell-space relative to the local block's (Xmin,Ymin,Zmin).
//
// The CIC kernel needs only +xyz halo of 1 cell (already provided by
// KSectionHalo), since positions are domain-decomposed into
// [Xmin,Xmax) x [Ymin,Ymax) x [Zmin,Zmax) and the 8 corners
// {ix,ix+1} x {iy,iy+1} x {iz,iz+1} all lie in [Xmin,Xmax+1] x ... x [Zmin,Zmax+1].
//
// Cell-space coordinates are in [0, N) along each axis. Wrap-around is handled
// by KSectionHalo (its +xyz neighbors wrap modulo N).
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#if defined(USE_MPI)
#include <mpi.h>
#endif

#include "general.hh"
#include "config_file.hh"
#include "logger.hh"
#include "mpi_ksection.hh"
#include "mpi_ksection_halo.hh"

#if defined(USE_HDF5)
#include "HDF_IO.hh"
#endif

namespace ksection {

template <typename T>
class GlassLoader {
public:
    using vec3 = std::array<real_t, 3>;

    //! Read /PartType1/Coordinates from `GlassFileName`, tile by
    //! `GlassTiles^3`, scale to cell-space `[0, N)`, and shuffle to the rank
    //! owning each point under `tree`. After construction, `glass_posr` holds
    //! the local rank's particle positions in cell-space, all within
    //! [Xmin, Xmin+Lx) x [Ymin, Ymin+Ly) x [Zmin, Zmin+Lz).
    GlassLoader(config_file& cf,
                const KSectionTree& tree,
                const KSectionHalo& halo,
                int ngrid,
                double boxlen
#if defined(USE_MPI)
                , MPI_Comm comm = MPI_COMM_WORLD
#endif
                )
        : tree_(&tree), halo_(&halo), ngrid_(ngrid), boxlen_(boxlen)
    {
#if defined(USE_MPI)
        comm_ = comm;
        MPI_Comm_rank(comm_, &myrank_);
        MPI_Comm_size(comm_, &nranks_);
#endif
        if (tree.nproc() != nranks_)
            throw std::invalid_argument(
                "ksection::GlassLoader: tree.nproc() != MPI comm size");

        const std::string glass_fname =
            cf.get_value<std::string>("setup", "GlassFileName");
        const std::size_t ntiles =
            cf.get_value<std::size_t>("setup", "GlassTiles");
        if (ntiles < 1)
            throw std::invalid_argument("ksection::GlassLoader: GlassTiles must be >= 1");

        real_t lglassbox = 1.0;
        std::size_t np_in_file = 0;
#if defined(USE_HDF5)
        HDFReadGroupAttribute(glass_fname, "Header", "BoxSize", lglassbox);
        std::vector<int> glass_extent;
        HDFGetDatasetExtent(glass_fname, "/PartType1/Coordinates", glass_extent);
        np_in_file = static_cast<std::size_t>(glass_extent[0]);
#else
        throw std::runtime_error(
            "ksection::GlassLoader: Glass lattice requires HDF5 support");
#endif

#if defined(USE_MPI)
        MPI_Bcast(&np_in_file, 1, MPI_UNSIGNED_LONG_LONG, 0, comm_);
#endif

        const std::size_t ntiles3 = ntiles * ntiles * ntiles;
        const std::size_t total_glass_particles = np_in_file * ntiles3;
        const std::size_t fft_grid_size =
            static_cast<std::size_t>(ngrid_) * ngrid_ * ngrid_;
        const real_t fft_oversampling =
            real_t(total_glass_particles) / real_t(fft_grid_size);

        music::ilog << "KSec glass: " << np_in_file
                    << " base particles, " << ntiles << "^3 tiles = "
                    << total_glass_particles << " total" << std::endl;
        music::ilog << "KSec glass: FFT grid oversampling = "
                    << fft_oversampling << "x (vs. " << ngrid_
                    << "^3 grid)" << std::endl;

        // ---- Read and chunk-scatter the base glass coords ----
        // Same chunked Scatter-by-Send/Recv pattern as the slab glass_loader.
        std::vector<real_t> local_base;
#if defined(USE_MPI)
        const std::size_t base_chunk = np_in_file / nranks_;
        const std::size_t remainder  = np_in_file % nranks_;
        const std::size_t my_base_count =
            base_chunk + (static_cast<std::size_t>(myrank_) < remainder ? 1 : 0);
        local_base.assign(my_base_count * 3, real_t(0));

        for (int ichunk = 0; ichunk < nranks_; ++ichunk) {
            const std::size_t chunk_count =
                base_chunk + (static_cast<std::size_t>(ichunk) < remainder ? 1 : 0);
            const std::size_t chunk_offset =
                static_cast<std::size_t>(ichunk) * base_chunk +
                std::min<std::size_t>(static_cast<std::size_t>(ichunk), remainder);
            std::vector<real_t> chunk_data;
            if (myrank_ == 0) {
#if defined(USE_HDF5)
                HDFReadVectorSlab(glass_fname, "/PartType1/Coordinates",
                                  chunk_offset, chunk_count, chunk_data);
#endif
            }
            if (ichunk == myrank_) {
                if (myrank_ == 0) local_base = std::move(chunk_data);
                else
                    MPI_Recv(local_base.data(),
                             static_cast<int>(chunk_count * 3),
                             MPI::get_datatype<real_t>(),
                             0, ichunk, comm_, MPI_STATUS_IGNORE);
            } else if (myrank_ == 0) {
                MPI_Send(chunk_data.data(),
                         static_cast<int>(chunk_count * 3),
                         MPI::get_datatype<real_t>(),
                         ichunk, ichunk, comm_);
            }
        }
        const std::size_t my_base = my_base_count;
#else
        // Serial: read everything on rank 0.
#if defined(USE_HDF5)
        HDFReadDataset(glass_fname, "/PartType1/Coordinates", local_base);
#endif
        const std::size_t my_base = local_base.size() / 3;
#endif

        // ---- Tile and convert to cell-space [0, N) ----
        std::vector<vec3> tiled(my_base * ntiles3, vec3{0, 0, 0});
        const real_t ngf = real_t(ngrid_);
        #pragma omp parallel for
        for (std::size_t ib = 0; ib < my_base; ++ib) {
            for (std::size_t it = 0; it < ntiles3; ++it) {
                const std::size_t tx = it / (ntiles * ntiles);
                const std::size_t ty = (it / ntiles) % ntiles;
                const std::size_t tz = it % ntiles;
                const std::size_t idx = ib * ntiles3 + it;
                tiled[idx][0] = std::fmod(
                    (local_base[3 * ib + 0] / lglassbox + real_t(tx)) /
                        real_t(ntiles) * ngf + ngf, ngf);
                tiled[idx][1] = std::fmod(
                    (local_base[3 * ib + 1] / lglassbox + real_t(ty)) /
                        real_t(ntiles) * ngf + ngf, ngf);
                tiled[idx][2] = std::fmod(
                    (local_base[3 * ib + 2] / lglassbox + real_t(tz)) /
                        real_t(ntiles) * ngf + ngf, ngf);
            }
        }

        // ---- 3D domain decomposition via KSectionTree ----
        // Convert cell coord -> real-space coord for cpu_for_point lookup.
        const real_t dx = static_cast<real_t>(boxlen_) / ngf;
        auto owner = [&](const vec3& p) {
            return tree_->cpu_for_point(
                static_cast<double>(p[0]) * dx,
                static_cast<double>(p[1]) * dx,
                static_cast<double>(p[2]) * dx);
        };

#if defined(USE_MPI)
        std::vector<int> sendcounts(nranks_, 0), recvcounts(nranks_, 0);
        std::vector<int> sendoffsets(nranks_, 0), recvoffsets(nranks_, 0);

        // Sort by destination rank (in place, by owner key) so we can pack
        // contiguous Alltoallv send buckets.
        std::sort(tiled.begin(), tiled.end(),
                  [&](const vec3& a, const vec3& b) {
                      return owner(a) < owner(b);
                  });
        for (const auto& p : tiled) sendcounts[owner(p)] += 3;

        MPI_Alltoall(sendcounts.data(), 1, MPI_INT,
                     recvcounts.data(), 1, MPI_INT, comm_);

        std::size_t tot_recv = 0;
        for (int r = 0; r < nranks_; ++r) {
            if (r > 0) {
                sendoffsets[r] = sendoffsets[r - 1] + sendcounts[r - 1];
                recvoffsets[r] = recvoffsets[r - 1] + recvcounts[r - 1];
            }
            tot_recv += recvcounts[r];
        }

        glass_posr_.assign(tot_recv / 3, vec3{0, 0, 0});
        MPI_Alltoallv(tiled.data(), sendcounts.data(), sendoffsets.data(),
                      MPI::get_datatype<real_t>(),
                      glass_posr_.data(), recvcounts.data(), recvoffsets.data(),
                      MPI::get_datatype<real_t>(), comm_);
#else
        glass_posr_ = std::move(tiled);
#endif

        num_p_ = glass_posr_.size();

        // Validate every local particle lies inside [Xmin,Xmin+Lx) x ...
        // After cpu_for_point shuffle, this should always hold; assert it so
        // a bad partition is caught loudly instead of producing silent
        // halo-buffer aliasing.
        {
            const int Xmin = halo_->Xmin();
            const int Ymin = halo_->Ymin();
            const int Zmin = halo_->Zmin();
            const int Xmax = Xmin + halo_->Lx();
            const int Ymax = Ymin + halo_->Ly();
            const int Zmax = Zmin + halo_->Lz();
            std::size_t out_of_box = 0;
            for (const auto& p : glass_posr_) {
                if (p[0] < real_t(Xmin) || p[0] >= real_t(Xmax) ||
                    p[1] < real_t(Ymin) || p[1] >= real_t(Ymax) ||
                    p[2] < real_t(Zmin) || p[2] >= real_t(Zmax))
                    ++out_of_box;
            }
            if (out_of_box > 0)
                throw std::runtime_error(
                    "ksection::GlassLoader: " + std::to_string(out_of_box) +
                    " particle(s) outside local block after domain decomp; "
                    "ksec tree and halo are inconsistent");
        }

        // Compute global offset (running sum of n_local across ranks).
        off_p_ = 0;
#if defined(USE_MPI)
        std::vector<std::size_t> all_n(nranks_, 0);
        MPI_Allgather(&num_p_, 1, MPI_UNSIGNED_LONG_LONG,
                      all_n.data(), 1, MPI_UNSIGNED_LONG_LONG, comm_);
        for (int r = 0; r < myrank_; ++r) off_p_ += all_n[r];
        std::size_t global = 0;
        for (int r = 0; r < nranks_; ++r) global += all_n[r];
        global_num_particles_ = global;
#else
        global_num_particles_ = num_p_;
#endif
    }

    //! CIC interpolation at `pos` (cell-space [0,N)). Reads from the extended
    //! buffer produced by KSectionHalo::update_halo (size
    //! (Lx+1)*(Ly+1)*(Lz+1)). pos must lie in [Xmin,Xmin+Lx) x ...
    T get_cic_at_ext(const vec3& pos, const T* ext) const noexcept {
        const int Xmin = halo_->Xmin();
        const int Ymin = halo_->Ymin();
        const int Zmin = halo_->Zmin();

        const int ix = static_cast<int>(pos[0]);
        const int iy = static_cast<int>(pos[1]);
        const int iz = static_cast<int>(pos[2]);
        const real_t dx = pos[0] - real_t(ix), tx = real_t(1) - dx;
        const real_t dy = pos[1] - real_t(iy), ty = real_t(1) - dy;
        const real_t dz = pos[2] - real_t(iz), tz = real_t(1) - dz;

        const int ei = ix - Xmin;
        const int ej = iy - Ymin;
        const int ek = iz - Zmin;

        T v{0};
        v += ext[halo_->ext_index(ei,     ej,     ek    )] * tx * ty * tz;
        v += ext[halo_->ext_index(ei,     ej,     ek + 1)] * tx * ty * dz;
        v += ext[halo_->ext_index(ei,     ej + 1, ek    )] * tx * dy * tz;
        v += ext[halo_->ext_index(ei,     ej + 1, ek + 1)] * tx * dy * dz;
        v += ext[halo_->ext_index(ei + 1, ej,     ek    )] * dx * ty * tz;
        v += ext[halo_->ext_index(ei + 1, ej,     ek + 1)] * dx * ty * dz;
        v += ext[halo_->ext_index(ei + 1, ej + 1, ek    )] * dx * dy * tz;
        v += ext[halo_->ext_index(ei + 1, ej + 1, ek + 1)] * dx * dy * dz;
        return v;
    }

    //! CIC compensation kernel in Fourier space. Same as
    //! grid_interpolate<1,...>::compensation_kernel: 1 / sinc^2(...).
    static ccomplex_t compensation_kernel_static(
        const vec3_t<real_t>& k,
        const std::array<real_t, 3>& kny) noexcept {
        auto sinc = [](real_t x) {
            return (std::fabs(x) > real_t(1e-10)) ? std::sin(x) / x : real_t(1);
        };
        const real_t dfx = sinc(real_t(0.5) * real_t(M_PI) * k[0] / kny[0]);
        const real_t dfy = sinc(real_t(0.5) * real_t(M_PI) * k[1] / kny[1]);
        const real_t dfz = sinc(real_t(0.5) * real_t(M_PI) * k[2] / kny[2]);
        const real_t del = (dfx * dfy * dfz) * (dfx * dfy * dfz);
        return ccomplex_t(real_t(1) / del, real_t(0));
    }

    const std::vector<vec3>& glass_posr() const noexcept { return glass_posr_; }
    std::size_t size() const noexcept { return num_p_; }
    std::size_t offset() const noexcept { return off_p_; }
    std::size_t global_num_particles() const noexcept { return global_num_particles_; }

private:
    const KSectionTree* tree_ = nullptr;
    const KSectionHalo* halo_ = nullptr;
    int ngrid_ = 0;
    double boxlen_ = 1.0;
    int myrank_ = 0;
    int nranks_ = 1;
#if defined(USE_MPI)
    MPI_Comm comm_ = MPI_COMM_NULL;
#endif

    std::vector<vec3> glass_posr_;
    std::size_t num_p_ = 0;
    std::size_t off_p_ = 0;
    std::size_t global_num_particles_ = 0;
};

} // namespace ksection
