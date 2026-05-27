// This file is part of monofonIC (MUSIC2)
// Implementation of the k-section roundtrip probe.
#include "ksection_probe.hh"

#include <cstddef>
#include <cstdio>
#include <vector>

#include "general.hh"
#include "grid_fft_ksection_bridge.hh"
#include "grid_ksection.hh"
#include "logger.hh"
#include "mpi_ksection.hh"
#include "mpi_ksection_redistribute.hh"

namespace ksection {

void run_probe(Grid_FFT<real_t>& tmp, config_file& the_config) {
    const bool enabled =
        the_config.get_value_safe<bool>("setup", "EnableKSectionProbe", false);
    if (!enabled) return;

    int myrank = 0, nranks = 1;
#if defined(USE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif

    if (myrank == 0) {
        music::ilog << "\n" << colors::BOLD << colors::HEADER
                    << colors::SYM_DIAMOND << " K-section probe: roundtrip slab->ksec->slab"
                    << colors::RESET << std::endl;
    }

    if (tmp.space_ != rspace_id) {
        if (myrank == 0)
            music::elog << "K-section probe: tmp must be in real space" << std::endl;
        return;
    }

    const int Nx = static_cast<int>(tmp.n_[0]);
    const int Ny = static_cast<int>(tmp.n_[1]);
    const int Nz = static_cast<int>(tmp.n_[2]);
    const SlabDescriptor slab = make_slab_descriptor(tmp);

    const double t0 = get_wtime();
    KSectionTree tree(nranks,
                      {static_cast<double>(tmp.length_[0]),
                       static_cast<double>(tmp.length_[1]),
                       static_cast<double>(tmp.length_[2])},
                      {Nx, Ny, Nz});
    tree.build();
    const double t_build = get_wtime() - t0;

    const double t1 = get_wtime();
    SlabRedistributor redist(slab, tree
#if defined(USE_MPI)
                             , MPI_COMM_WORLD
#endif
                            );
    const double t_plan = get_wtime() - t1;

    // Fill tmp with ramp = global linear index (cast to real_t).
    for (int i = 0; i < slab.local_0_size; ++i) {
        const int ix = i + slab.local_0_start;
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                tmp.relem(i, j, k) =
                    static_cast<real_t>(static_cast<double>(ix) * Ny * Nz
                                        + static_cast<double>(j) * Nz + k);
            }
        }
    }

    // Forward.
    const double t2 = get_wtime();
    Grid_KSection<real_t> ksec;
    redistribute_slab_to_ksec(tmp, redist, ksec);
    const double t_fwd = get_wtime() - t2;

    // Verify recv values match the ramp at (gix,giy,giz).
    int local_value_ok = 1;
    int local_owner_ok = 1;
    {
        const auto& gix = redist.recv_gix();
        const auto& giy = redist.recv_giy();
        const auto& giz = redist.recv_giz();
        for (std::size_t s = 0; s < ksec.n_local(); ++s) {
            const double expect = static_cast<double>(gix[s]) * Ny * Nz
                                  + static_cast<double>(giy[s]) * Nz + giz[s];
            if (static_cast<double>(ksec.data()[s]) != expect) { local_value_ok = 0; break; }
            if (tree.cpu_for_cell(gix[s], giy[s], giz[s]) != myrank) { local_owner_ok = 0; break; }
        }
    }
    int value_ok = 0, owner_ok = 0;
#if defined(USE_MPI)
    MPI_Allreduce(&local_value_ok, &value_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_owner_ok, &owner_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#else
    value_ok = local_value_ok; owner_ok = local_owner_ok;
#endif

    // Backward.
    // Snapshot original slab into a side buffer for comparison.
    const std::size_t nlocal_slab = static_cast<std::size_t>(slab.local_0_size)
                                  * slab.n_global[1] * slab.stride_z;
    std::vector<real_t> snapshot(nlocal_slab);
    for (int i = 0; i < slab.local_0_size; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
                snapshot[slab.slab_index(i, j, k)] = tmp.relem(i, j, k);

    // Zero tmp, then push ksec data back into it.
    tmp.zero();
    const double t3 = get_wtime();
    redistribute_ksec_to_slab(ksec, redist, tmp);
    const double t_bwd = get_wtime() - t3;

    int local_round_ok = 1;
    for (int i = 0; i < slab.local_0_size; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
                if (tmp.relem(i, j, k) != snapshot[slab.slab_index(i, j, k)]) {
                    local_round_ok = 0; goto done_compare;
                }
done_compare:
    int round_ok = 0;
#if defined(USE_MPI)
    MPI_Allreduce(&local_round_ok, &round_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#else
    round_ok = local_round_ok;
#endif

    // Cell-count statistics: min, max, mean across ranks.
    const long long my_nrecv = static_cast<long long>(ksec.n_local());
    long long mn = my_nrecv, mx = my_nrecv, sum = my_nrecv;
#if defined(USE_MPI)
    MPI_Allreduce(MPI_IN_PLACE, &mn,  1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mx,  1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    const double mean = (nranks > 0) ? static_cast<double>(sum) / nranks : 0.0;
    const long long expect_total = static_cast<long long>(Nx) * Ny * Nz;

    // Per-rank wallclock min/max/mean for the MPI-bound phases.
    double fwd_mn = t_fwd, fwd_mx = t_fwd, fwd_sum = t_fwd;
    double bwd_mn = t_bwd, bwd_mx = t_bwd, bwd_sum = t_bwd;
    double plan_mn = t_plan, plan_mx = t_plan, plan_sum = t_plan;
#if defined(USE_MPI)
    MPI_Allreduce(MPI_IN_PLACE, &fwd_mn,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &fwd_mx,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &fwd_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bwd_mn,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bwd_mx,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bwd_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &plan_mn,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &plan_mx,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &plan_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    const double fwd_mean  = (nranks > 0) ? fwd_sum  / nranks : 0.0;
    const double bwd_mean  = (nranks > 0) ? bwd_sum  / nranks : 0.0;
    const double plan_mean = (nranks > 0) ? plan_sum / nranks : 0.0;

    // Restore tmp to zero so we don't leak probe state into the LPT flow.
    tmp.zero();

    if (myrank == 0) {
        std::printf(" K-section probe (Nx=%d Ny=%d Nz=%d, nranks=%d)\n",
                    Nx, Ny, Nz, nranks);
        std::printf("   factors=[");
        for (size_t i = 0; i < tree.factors().size(); ++i)
            std::printf("%s%d", i ? "," : "", tree.factors()[i]);
        std::printf("]  dirs=[");
        for (size_t i = 0; i < tree.directions().size(); ++i)
            std::printf("%s%d", i ? "," : "", tree.directions()[i]);
        std::printf("]\n");
        std::printf("   per-rank cells: min=%lld max=%lld mean=%.0f total=%lld (expect %lld)\n",
                    mn, mx, mean, sum, expect_total);
        std::printf("   tree_build (rank0)         : %.4f s\n", t_build);
        std::printf("   plan_build min/max/mean(s) : %.4f / %.4f / %.4f\n",
                    plan_mn, plan_mx, plan_mean);
        std::printf("   forward    min/max/mean(s) : %.4f / %.4f / %.4f\n",
                    fwd_mn, fwd_mx, fwd_mean);
        std::printf("   backward   min/max/mean(s) : %.4f / %.4f / %.4f\n",
                    bwd_mn, bwd_mx, bwd_mean);
        std::printf("   recv values correct : %s\n", value_ok ? "PASS" : "FAIL");
        std::printf("   recv ownership      : %s\n", owner_ok ? "PASS" : "FAIL");
        std::printf("   slab->ksec->slab    : %s\n", round_ok ? "PASS" : "FAIL");
        std::fflush(stdout);
    }
}

} // namespace ksection
