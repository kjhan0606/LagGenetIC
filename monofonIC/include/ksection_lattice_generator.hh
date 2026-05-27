// This file is part of monofonIC (MUSIC2)
// Bravais-lattice particle generator that consumes Grid_KSection
// (cuRamses-style recursive k-section partition) instead of FFTW3-MPI
// slabs. Supports SC/BCC/FCC/RSC (overload = 1 << lattice_type), the
// secondary-lattice baryon shift, and per-particle masses.
//
// Masked-SC (lattice_type = -2) is supported for positions and velocities;
// per-cell masses for masked require a 1-cell halo and are handled by
// set_masses_masked() (KSectionHalo wired by the caller).
//
// Glass (lattice_type = -1) is supported via the glass_tag constructor; the
// caller wires a ksection::GlassLoader (which performs 3D domain-decomp via
// KSectionTree::cpu_for_point) and provides a KSectionHalo-extended buffer to
// set_positions_glass / set_velocities_glass. Glass does not support per-
// particle masses (matches the slab path).
//
// The caller is responsible for applying shift_field() to the slab before
// each redistribute; this class is index-only and never touches the slab.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(USE_MPI)
#include <mpi.h>
#endif

#include "general.hh"
#include "grid_ksection.hh"
#include "ksection_glass_loader.hh"
#include "logger.hh"
#include "mpi_ksection_halo.hh"
#include "mpi_ksection_redistribute.hh"
#include "particle_container.hh"

namespace ksection {

template <typename T>
class ksection_lattice_generator {
public:
    //! Bravais constructor: allocate particles + set IDs.
    //!   lattice_type: 0=SC, 1=BCC, 2=FCC, 3=RSC. Overload = 1 << lattice_type.
    //!   shifted_lattice: false=primary species, true=secondary (baryon, etc.).
    //! Per-particle indexing: particles_[ishift * n_recv + s].
    ksection_lattice_generator(int lattice_type,
                               bool shifted_lattice,
                               const SlabRedistributor& redist,
                               int ngrid,
                               std::size_t IDoffset,
                               bool b64reals, bool b64ids,
                               bool bwithmasses)
    : lattice_type_(lattice_type),
      shifted_lattice_(shifted_lattice),
      ngrid_(ngrid),
      n_recv_(static_cast<std::size_t>(redist.n_recv())),
      redist_(&redist),
      b64reals_(b64reals), b64ids_(b64ids)
    {
        if (lattice_type < 0)
            throw std::logic_error(
                "ksection_lattice_generator: Bravais constructor called with "
                "non-Bravais lattice_type; use the masked constructor instead");
        overload_ = std::size_t{1} << lattice_type;

        const std::size_t nlocal = overload_ * n_recv_;
        particles_.allocate(nlocal, b64reals, b64ids, bwithmasses);

        const std::uint64_t Nu = static_cast<std::uint64_t>(ngrid);
        global_num_particles_ = Nu * Nu * Nu * overload_;

        // ID encoding mirrors Bravais lattice_generator:
        //   off  = IDoffset * overload * N^3
        //   ID   = off + overload * cell_idx_1d + iload
        const std::uint64_t off =
            static_cast<std::uint64_t>(IDoffset) * overload_ * Nu * Nu * Nu;

        const auto& gix = redist.recv_gix();
        const auto& giy = redist.recv_giy();
        const auto& giz = redist.recv_giz();
        for (std::size_t s = 0; s < n_recv_; ++s) {
            const std::uint64_t cell_idx_1d =
                (static_cast<std::uint64_t>(gix[s]) * Nu +
                 static_cast<std::uint64_t>(giy[s])) * Nu +
                static_cast<std::uint64_t>(giz[s]);
            for (std::size_t iload = 0; iload < overload_; ++iload) {
                const std::uint64_t id = off + overload_ * cell_idx_1d + iload;
                const std::size_t slot = iload * n_recv_ + s;
                if (b64ids) particles_.set_id64(slot, id);
                else        particles_.set_id32(slot,
                                static_cast<std::uint32_t>(id));
            }
        }

        music::ilog << "Created KSec Particles [" << (shifted_lattice ? 1 : 0)
                    << "] : " << global_num_particles_ << std::endl;
    }

    //! Masked-SC constructor (lattice_type = -2 in the slab nomenclature).
    //! Mirrors particle_generator::lattice_masked: filters cells against the
    //! 2x2x2 mask, allocates only the selected slots, and stores
    //!   ID = IDoffset * N^3 + (gix*N + giy)*N + giz
    //! Particle indexing: particles_[ip] where ip = 0..n_particles-1 in the
    //! recv-slot order; selected_slots_[ip] = redist recv slot for cell ip.
    //!
    //!   lattice_index: 0 = primary (CDM), 1 = secondary (baryon).
    //!   mask: 8-entry 2x2x2 mask, value at ((gx%2)*2+gy%2)*2+gz%2 selects the
    //!         lattice_index that cell contributes to. Caller builds it from
    //!         DoBaryons + ParticleMaskType using the same rules as the slab
    //!         path.
    struct masked_tag {};
    ksection_lattice_generator(masked_tag,
                               int lattice_index,
                               const std::array<int, 8>& mask,
                               const SlabRedistributor& redist,
                               int ngrid,
                               std::size_t IDoffset,
                               bool b64reals, bool b64ids,
                               bool bwithmasses
#if defined(USE_MPI)
                               , MPI_Comm comm = MPI_COMM_WORLD
#endif
                               )
    : lattice_type_(-2),
      shifted_lattice_(lattice_index == 1),
      ngrid_(ngrid),
      n_recv_(static_cast<std::size_t>(redist.n_recv())),
      redist_(&redist),
      b64reals_(b64reals), b64ids_(b64ids),
      lattice_index_(lattice_index),
      mask_(mask)
    {
        if (lattice_index < 0 || lattice_index > 1)
            throw std::invalid_argument(
                "ksection_lattice_generator (masked): lattice_index must be 0 or 1");
        if (ngrid % 2 != 0)
            throw std::invalid_argument(
                "ksection_lattice_generator (masked): linear field resolution "
                "must be even (N % 2 == 0)");
        overload_ = 1;

        // Filter recv slots by mask, preserve recv-slot order.
        const auto& gix = redist.recv_gix();
        const auto& giy = redist.recv_giy();
        const auto& giz = redist.recv_giz();
        selected_slots_.clear();
        selected_slots_.reserve(n_recv_);
        for (std::size_t s = 0; s < n_recv_; ++s) {
            const int sig = ((gix[s] & 1) * 2 + (giy[s] & 1)) * 2 + (giz[s] & 1);
            if (mask_[sig] == lattice_index)
                selected_slots_.push_back(static_cast<std::uint32_t>(s));
        }
        const std::size_t nlocal = selected_slots_.size();

        particles_.allocate(nlocal, b64reals, b64ids, bwithmasses);

        // Global particle count for this masked species.
#if defined(USE_MPI)
        {
            std::uint64_t local = static_cast<std::uint64_t>(nlocal);
            std::uint64_t global = 0;
            MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG,
                          MPI_SUM, comm);
            global_num_particles_ = static_cast<std::size_t>(global);
        }
#else
        global_num_particles_ = nlocal;
#endif

        // ID encoding mirrors slab masked path:
        //   off = IDoffset * N^3
        //   ID  = off + cell_idx_1d, cell_idx_1d = (gix*N + giy)*N + giz
        const std::uint64_t Nu = static_cast<std::uint64_t>(ngrid);
        const std::uint64_t off =
            static_cast<std::uint64_t>(IDoffset) * Nu * Nu * Nu;
        for (std::size_t ip = 0; ip < nlocal; ++ip) {
            const std::size_t s = selected_slots_[ip];
            const std::uint64_t cell_idx_1d =
                (static_cast<std::uint64_t>(gix[s]) * Nu +
                 static_cast<std::uint64_t>(giy[s])) * Nu +
                static_cast<std::uint64_t>(giz[s]);
            const std::uint64_t id = off + cell_idx_1d;
            if (b64ids) particles_.set_id64(ip, id);
            else        particles_.set_id32(ip,
                            static_cast<std::uint32_t>(id));
        }

        music::ilog << "Created KSec Masked Particles ["
                    << lattice_index << "] : "
                    << global_num_particles_ << std::endl;
    }

    //! Store positions for one sub-shift block (ishift) and one dim.
    //! base_shift_unit = lattice_shifts[lt][ishift] + (shifted ? second_shift : 0).
    //! Position[idim] = ((g_idim + base_shift_unit[idim]) / N) * lunit + disp[s].
    void set_positions(int ishift, int idim, T lunit,
                       const std::array<T, 3>& base_shift_unit,
                       const Grid_KSection<T>& ksec) {
        if (lattice_type_ < 0)
            throw std::logic_error(
                "ksection_lattice_generator::set_positions called on a "
                "non-Bravais generator; use set_positions_masked instead");
        check_ishift(ishift);
        const auto& gix = redist_->recv_gix();
        const auto& giy = redist_->recv_giy();
        const auto& giz = redist_->recv_giz();
        const T invN = T(1.0) / T(ngrid_);
        const T* d = ksec.data();
        const std::size_t base = static_cast<std::size_t>(ishift) * n_recv_;
        const T sh = base_shift_unit[idim];
        for (std::size_t s = 0; s < n_recv_; ++s) {
            const int g = (idim == 0) ? gix[s] : (idim == 1) ? giy[s] : giz[s];
            const T pos = (T(g) + sh) * invN * lunit + d[s];
            const std::size_t slot = base + s;
            if (b64reals_) particles_.set_pos64(slot, idim, pos);
            else           particles_.set_pos32(slot, idim, pos);
        }
    }

    void set_velocities(int ishift, int idim, const Grid_KSection<T>& ksec) {
        if (lattice_type_ < 0)
            throw std::logic_error(
                "ksection_lattice_generator::set_velocities called on a "
                "non-Bravais generator; use set_velocities_masked instead");
        check_ishift(ishift);
        const T* d = ksec.data();
        const std::size_t base = static_cast<std::size_t>(ishift) * n_recv_;
        for (std::size_t s = 0; s < n_recv_; ++s) {
            const std::size_t slot = base + s;
            if (b64reals_) particles_.set_vel64(slot, idim, d[s]);
            else           particles_.set_vel32(slot, idim, d[s]);
        }
    }

    //! Mass at slot = pmeanmass * rho[s]. Bravais branch only; pmeanmass =
    //! munit / (N^3 * overload).
    void set_masses(int ishift, T pmeanmass, const Grid_KSection<T>& ksec) {
        if (lattice_type_ < 0)
            throw std::logic_error(
                "ksection_lattice_generator::set_masses called on a "
                "non-Bravais generator; use set_masses_masked instead");
        check_ishift(ishift);
        const T* d = ksec.data();
        const std::size_t base = static_cast<std::size_t>(ishift) * n_recv_;
        for (std::size_t s = 0; s < n_recv_; ++s) {
            const std::size_t slot = base + s;
            const T m = pmeanmass * d[s];
            if (b64reals_) particles_.set_mass64(slot, m);
            else           particles_.set_mass32(slot, m);
        }
    }

    //! Masked-SC positions: pos[idim] = (g[idim] / N) * lunit + disp[s].
    //! No Bravais sub-shift — the mask itself selects which cell holds the
    //! particle, so the per-cell offset stays at zero.
    void set_positions_masked(int idim, T lunit,
                              const Grid_KSection<T>& ksec) {
        require_masked();
        const auto& gix = redist_->recv_gix();
        const auto& giy = redist_->recv_giy();
        const auto& giz = redist_->recv_giz();
        const T invN = T(1.0) / T(ngrid_);
        const T* d = ksec.data();
        const std::size_t n_part = selected_slots_.size();
        for (std::size_t ip = 0; ip < n_part; ++ip) {
            const std::size_t s = selected_slots_[ip];
            const int g = (idim == 0) ? gix[s] : (idim == 1) ? giy[s] : giz[s];
            const T pos = T(g) * invN * lunit + d[s];
            if (b64reals_) particles_.set_pos64(ip, idim, pos);
            else           particles_.set_pos32(ip, idim, pos);
        }
    }

    //! Masked-SC masses. Mirrors the slab masked path:
    //!   pmass = pmeanmass * (rho[s] - (mean_masked - mean_full))
    //! over a 2x2x2 stencil aligned to the cell-2 group containing the
    //! particle's cell. Requires the ksec block to be 2-aligned (built via
    //! KSectionTree::set_cell_alignment(2)) so the stencil stays inside the
    //! core block and no halo exchange is needed.
    //!
    //! pmeanmass = munit / global_num_particles().
    void set_masses_masked(T pmeanmass,
                           const Grid_KSection<T>& ksec,
                           const KSectionHalo& halo) {
        require_masked();
        const int Xmin = halo.Xmin(), Ymin = halo.Ymin(), Zmin = halo.Zmin();
        const int Lx   = halo.Lx(),   Ly   = halo.Ly(),   Lz   = halo.Lz();
        if ((Xmin & 1) || (Ymin & 1) || (Zmin & 1) ||
            (Lx & 1)   || (Ly & 1)   || (Lz & 1)) {
            throw std::runtime_error(
                "ksection_lattice_generator::set_masses_masked: ksec block is "
                "not 2-aligned in cell space; rebuild the KSectionTree with "
                "set_cell_alignment(2)");
        }
        const auto& gix = redist_->recv_gix();
        const auto& giy = redist_->recv_giy();
        const auto& giz = redist_->recv_giz();
        const T* d = ksec.data();
        const std::size_t n_part = selected_slots_.size();
        for (std::size_t ip = 0; ip < n_part; ++ip) {
            const std::size_t s = selected_slots_[ip];
            const int ei = gix[s] - Xmin;
            const int ej = giy[s] - Ymin;
            const int ek = giz[s] - Zmin;
            const int ei0 = ei & ~1;
            const int ej0 = ej & ~1;
            const int ek0 = ek & ~1;
            T sum_full   = T(0);
            T sum_masked = T(0);
            int cnt_full = 0, cnt_masked = 0;
            for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c) {
                const int recv_slot =
                    halo.recv_slot_of_core(ei0 + a, ej0 + b, ek0 + c);
                const T v = d[recv_slot];
                sum_full += v; ++cnt_full;
                if (mask_[(a * 2 + b) * 2 + c] == lattice_index_) {
                    sum_masked += v; ++cnt_masked;
                }
            }
            const T mean_mask = sum_masked / T(cnt_masked) -
                                sum_full   / T(cnt_full);
            const T pmass = pmeanmass * (d[s] - mean_mask);
            if (b64reals_) particles_.set_mass64(ip, pmass);
            else           particles_.set_mass32(ip, pmass);
        }
    }

    //! Glass constructor. Allocates particles for the local glass count and
    //! sets sequential IDs `IDoffset + i + glass.offset()`. After this you
    //! must call set_positions_glass / set_velocities_glass with the
    //! KSectionHalo-extended buffer of the displacement field.
    //!
    //! Glass does not support per-particle masses (matches slab path).
    struct glass_tag {};
    ksection_lattice_generator(glass_tag,
                               const GlassLoader<T>& glass,
                               const SlabRedistributor& redist,
                               int ngrid,
                               std::size_t IDoffset,
                               bool b64reals, bool b64ids)
    : lattice_type_(-1),
      shifted_lattice_(false),
      ngrid_(ngrid),
      n_recv_(static_cast<std::size_t>(redist.n_recv())),
      redist_(&redist),
      b64reals_(b64reals), b64ids_(b64ids),
      glass_(&glass)
    {
        overload_ = 1;

        const std::size_t nlocal = glass.size();
        particles_.allocate(nlocal, b64reals, b64ids, false);
        global_num_particles_ = glass.global_num_particles();

        const std::uint64_t off =
            static_cast<std::uint64_t>(IDoffset) *
            static_cast<std::uint64_t>(global_num_particles_);
        const std::uint64_t glass_off = static_cast<std::uint64_t>(glass.offset());
        for (std::size_t i = 0; i < nlocal; ++i) {
            const std::uint64_t id = off + glass_off + static_cast<std::uint64_t>(i);
            if (b64ids) particles_.set_id64(i, id);
            else        particles_.set_id32(i, static_cast<std::uint32_t>(id));
        }

        music::ilog << "Created KSec Glass Particles : "
                    << global_num_particles_ << std::endl;
    }

    //! Glass positions: pos[idim] = pos_cell[idim] / N * lunit + cic(disp).
    //! `ext` is the KSectionHalo extended buffer (size (Lx+1)*(Ly+1)*(Lz+1))
    //! holding the idim-component of the displacement field after a
    //! redistribute + halo update.
    void set_positions_glass(int idim, T lunit, const T* ext) {
        require_glass();
        const auto& positions = glass_->glass_posr();
        const T invN = T(1) / T(ngrid_);
        const std::size_t n = positions.size();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& p = positions[i];
            const T disp = glass_->get_cic_at_ext(p, ext);
            const T pos_out = T(p[idim]) * invN * lunit + disp;
            if (b64reals_) particles_.set_pos64(i, idim, pos_out);
            else           particles_.set_pos32(i, idim, pos_out);
        }
    }

    //! Glass velocities: vel[idim] = cic(vel_field).
    void set_velocities_glass(int idim, const T* ext) {
        require_glass();
        const auto& positions = glass_->glass_posr();
        const std::size_t n = positions.size();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& p = positions[i];
            const T v = glass_->get_cic_at_ext(p, ext);
            if (b64reals_) particles_.set_vel64(i, idim, v);
            else           particles_.set_vel32(i, idim, v);
        }
    }

    //! Masked-SC velocities: vel[idim] = disp[s] (no per-cell offset).
    void set_velocities_masked(int idim, const Grid_KSection<T>& ksec) {
        require_masked();
        const T* d = ksec.data();
        const std::size_t n_part = selected_slots_.size();
        for (std::size_t ip = 0; ip < n_part; ++ip) {
            const std::size_t s = selected_slots_[ip];
            if (b64reals_) particles_.set_vel64(ip, idim, d[s]);
            else           particles_.set_vel32(ip, idim, d[s]);
        }
    }

    const particle::container& get_particles() const noexcept {
        return particles_;
    }

    std::size_t global_num_particles() const noexcept {
        return global_num_particles_;
    }

    std::size_t overload() const noexcept { return overload_; }
    int lattice_type() const noexcept { return lattice_type_; }
    int lattice_index() const noexcept { return lattice_index_; }

    //! Recv-slot ordering of the selected particles (masked path only).
    //! Empty for Bravais. Used by set_masses_masked (Phase 3) to drive
    //! the halo-based 2x2x2 stencil over the same particles.
    const std::vector<std::uint32_t>& selected_slots() const noexcept {
        return selected_slots_;
    }

    const std::array<int, 8>& mask() const noexcept { return mask_; }

private:
    void check_ishift(int ishift) const {
        if (ishift < 0 || static_cast<std::size_t>(ishift) >= overload_)
            throw std::out_of_range(
                "ksection_lattice_generator: ishift out of range");
    }
    void require_masked() const {
        if (lattice_type_ != -2)
            throw std::logic_error(
                "ksection_lattice_generator: masked API called on a "
                "non-masked generator");
    }
    void require_glass() const {
        if (lattice_type_ != -1)
            throw std::logic_error(
                "ksection_lattice_generator: glass API called on a "
                "non-glass generator");
    }

    int lattice_type_;
    bool shifted_lattice_;
    int ngrid_;
    std::size_t n_recv_;
    const SlabRedistributor* redist_ = nullptr;
    std::size_t overload_ = 1;
    bool b64reals_;
    bool b64ids_;
    particle::container particles_;
    std::size_t global_num_particles_ = 0;

    // Masked-path state (empty for Bravais).
    int lattice_index_ = 0;
    std::array<int, 8> mask_ = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<std::uint32_t> selected_slots_;

    // Glass-path state (null for Bravais/masked).
    const GlassLoader<T>* glass_ = nullptr;
};

} // namespace ksection
