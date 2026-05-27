// MPI unit test for KSectionHalo: fill a slab with a global ramp,
// redistribute to ksec, then run update_halo and check that every cell in
// the extended (Lx+1)*(Ly+1)*(Lz+1) buffer holds the correct global value
// (cell at extended (ei, ej, ek) -> global (Xmin+ei, Ymin+ej, Zmin+ek)
// modulo N, with the ramp value = ix*Ny*Nz + iy*Nz + iz).

#include "mpi_ksection.hh"
#include "mpi_ksection_halo.hh"
#include "mpi_ksection_redistribute.hh"

#include <cmath>
#include <cstdio>
#include <mpi.h>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK_ROOT(cond)                                                       \
    do {                                                                       \
        int _ok = (cond) ? 1 : 0;                                              \
        int _all_ok = 0;                                                       \
        MPI_Allreduce(&_ok, &_all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);    \
        if (rank == 0) {                                                       \
            if (_all_ok) ++g_pass;                                             \
            else { ++g_fail; std::fprintf(stderr, "FAIL %s:%d  %s\n",          \
                                          __FILE__, __LINE__, #cond); }       \
        }                                                                      \
    } while (0)

static double global_linear(int ix, int iy, int iz, int Ny, int Nz) {
    return static_cast<double>(ix) * Ny * Nz + static_cast<double>(iy) * Nz + iz;
}

static void run_case(int rank, int nranks, int N) {
    using namespace ksection;
    // Tight slab partition.
    const int base = N / nranks;
    const int rem  = N % nranks;
    const int local_0_size  = (rank < rem) ? base + 1 : base;
    const int local_0_start = (rank < rem) ? rank * (base + 1)
                                           : rem * (base + 1) + (rank - rem) * base;

    SlabDescriptor desc;
    desc.n_global     = {N, N, N};
    desc.local_0_start = local_0_start;
    desc.local_0_size  = local_0_size;
    desc.stride_z      = N;

    KSectionTree tree(nranks, {1.0, 1.0, 1.0}, {N, N, N});
    tree.build();

    // Also build a 2-aligned tree and verify every rank's cell-bbox starts
    // and ends at an even cell index. Used downstream by masked masses to
    // run the 2x2x2 stencil entirely inside the core block.
    KSectionTree tree2(nranks, {1.0, 1.0, 1.0}, {N, N, N});
    tree2.set_cell_alignment(2);
    tree2.build();
    {
        const double dx = 1.0 / N;
        bool ok2 = true;
        for (int r = 0; r < nranks; ++r) {
            auto bb = tree2.bbox_for_rank(r);
            for (int d = 0; d < 3 && ok2; ++d) {
                const int lo = static_cast<int>(std::round(bb[d][0] / dx));
                const int hi = static_cast<int>(std::round(bb[d][1] / dx));
                if ((lo & 1) || (hi & 1)) ok2 = false;
            }
        }
        CHECK_ROOT(ok2);
    }

    SlabRedistributor redist(desc, tree, MPI_COMM_WORLD);

    // Fill slab with global linear index.
    std::vector<double> slab(static_cast<std::size_t>(desc.local_cells()), 0.0);
    for (int i = 0; i < local_0_size; ++i) {
        const int ix = i + local_0_start;
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                slab[desc.slab_index(i, j, k)] = global_linear(ix, j, k, N, N);
    }

    std::vector<double> ksec;
    redist.slab_to_ksec(slab.data(), ksec);

    KSectionHalo halo(redist, tree, MPI_COMM_WORLD);
    std::vector<double> ext;
    halo.update_halo(ksec.data(), ext);

    // Verify every cell in the extended buffer (including halo) matches the
    // expected ramp value at its global coordinates (with wraparound).
    bool ok = true;
    int first_bad_i = -1, first_bad_j = -1, first_bad_k = -1;
    double bad_got = 0.0, bad_expect = 0.0;
    for (int ei = 0; ei <= halo.Lx() && ok; ++ei) {
        for (int ej = 0; ej <= halo.Ly() && ok; ++ej) {
            for (int ek = 0; ek <= halo.Lz() && ok; ++ek) {
                const int gx = (halo.Xmin() + ei + N) % N;
                const int gy = (halo.Ymin() + ej + N) % N;
                const int gz = (halo.Zmin() + ek + N) % N;
                const double expect = global_linear(gx, gy, gz, N, N);
                const double got = ext[halo.ext_index(ei, ej, ek)];
                if (got != expect) {
                    ok = false;
                    first_bad_i = ei; first_bad_j = ej; first_bad_k = ek;
                    bad_got = got; bad_expect = expect;
                }
            }
        }
    }
    if (!ok && rank == 0) {
        std::fprintf(stderr,
                     "  ext mismatch at (ei=%d,ej=%d,ek=%d): got=%.0f expect=%.0f\n",
                     first_bad_i, first_bad_j, first_bad_k, bad_got, bad_expect);
    }
    CHECK_ROOT(ok);

    // Additionally check that the block is rectangular (Lx*Ly*Lz == n_recv).
    const long long core = static_cast<long long>(halo.Lx()) * halo.Ly() * halo.Lz();
    const long long n_recv = static_cast<long long>(ksec.size());
    CHECK_ROOT(core == n_recv);

    if (rank == 0) {
        std::printf("  case N=%d nproc=%d : block=%dx%dx%d (Xmin=%d Ymin=%d Zmin=%d)\n",
                    N, nranks, halo.Lx(), halo.Ly(), halo.Lz(),
                    halo.Xmin(), halo.Ymin(), halo.Zmin());
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (rank == 0)
        std::printf("== test_ksection_halo (nranks=%d) ==\n", nranks);

    for (int N : {16, 24, 32}) {
        if (N < nranks) continue;
        if (rank == 0) std::printf("-- N=%d --\n", N);
        run_case(rank, nranks, N);
    }

    int ec = 0;
    if (rank == 0) {
        std::printf("\nresults: %d passed, %d failed\n", g_pass, g_fail);
        ec = (g_fail == 0) ? 0 : 1;
    }
    MPI_Bcast(&ec, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ec;
}
