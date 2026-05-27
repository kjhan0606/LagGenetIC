// MPI unit test for SlabRedistributor: slab <-> ksec roundtrip on a ramp field.
//
// Layout: tightly-packed real-space slab (stride_z = Nz). Rank r owns
// x-slabs [r * Nx/P, (r+1) * Nx/P). Nx must be divisible by P for this
// simple test (we choose Nx that satisfies it).
//
// Fill every cell with its global linear index, encoded as a double:
//     value = ix * Ny * Nz + iy * Nz + iz
//
// After forward redistribute, on each rank, every received cell must satisfy
//   value == global_linear(out_gix, out_giy, out_giz)
// and the owner of that cell must be myrank (i.e. tree.cpu_for_cell == myrank).
// Total cell count across all ranks after recv == Nx*Ny*Nz.
//
// After backward redistribute, the slab must be bit-exact equal to the
// original.

#include "mpi_ksection.hh"
#include "mpi_ksection_redistribute.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
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

namespace {

double global_linear(int ix, int iy, int iz, int Ny, int Nz) {
    return static_cast<double>(ix) * Ny * Nz + static_cast<double>(iy) * Nz + iz;
}

struct Slab {
    int local_0_start, local_0_size;
};

Slab simple_slab_partition(int N, int rank, int nranks) {
    // Block distribution: first `rem` ranks get one extra slab.
    const int base = N / nranks;
    const int rem  = N % nranks;
    Slab s{};
    s.local_0_size  = (rank < rem) ? base + 1 : base;
    s.local_0_start = (rank < rem) ? rank * (base + 1)
                                   : rem * (base + 1) + (rank - rem) * base;
    return s;
}

void run_case(int rank, int nranks, int N) {
    using namespace ksection;
    const Slab s = simple_slab_partition(N, rank, nranks);
    SlabDescriptor desc;
    desc.n_global     = {N, N, N};
    desc.local_0_start = s.local_0_start;
    desc.local_0_size  = s.local_0_size;
    desc.stride_z      = N;  // tight pack for unit test

    KSectionTree tree(nranks, {1.0, 1.0, 1.0}, {N, N, N});
    tree.build();

    SlabRedistributor redist(desc, tree, MPI_COMM_WORLD);

    // Fill slab with global linear index as double.
    const long long nlocal = desc.local_cells();
    std::vector<double> slab(static_cast<size_t>(nlocal), 0.0);
    for (int i = 0; i < desc.local_0_size; ++i) {
        const int ix = i + desc.local_0_start;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                slab[desc.slab_index(i, j, k)] = global_linear(ix, j, k, N, N);
            }
        }
    }

    // Forward.
    std::vector<double> ksec;
    std::vector<int> gix, giy, giz;
    redist.slab_to_ksec(slab.data(), ksec, &gix, &giy, &giz);

    const int nrecv = redist.n_recv();
    // 1) Local recv consistency: value matches global linear of (gix,giy,giz),
    //    and each cell is actually owned by my rank.
    bool ok_values = true, ok_owner = true;
    for (int s = 0; s < nrecv; ++s) {
        if (ksec[s] != global_linear(gix[s], giy[s], giz[s], N, N)) {
            ok_values = false;
            if (rank == 0)
                std::fprintf(stderr,
                             "  value mismatch at recv slot %d: got %.0f expected %.0f"
                             "  (ix=%d iy=%d iz=%d)\n",
                             s, ksec[s], global_linear(gix[s], giy[s], giz[s], N, N),
                             gix[s], giy[s], giz[s]);
            break;
        }
        if (tree.cpu_for_cell(gix[s], giy[s], giz[s]) != rank) {
            ok_owner = false;
            break;
        }
    }
    CHECK_ROOT(ok_values);
    CHECK_ROOT(ok_owner);

    // 2) Total cell count across all ranks equals N^3.
    long long my_nrecv = nrecv, tot_nrecv = 0;
    MPI_Allreduce(&my_nrecv, &tot_nrecv, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    const long long expect = static_cast<long long>(N) * N * N;
    CHECK_ROOT(tot_nrecv == expect);

    // 3) No duplicate ownership: sum of (gix*Ny*Nz + giy*Nz + giz) across ranks
    //    equals sum_{0..N^3-1} = N^3*(N^3-1)/2. Bit-exact in double up to N=512.
    double my_sum = 0.0, tot_sum = 0.0;
    for (int s = 0; s < nrecv; ++s) my_sum += ksec[s];
    MPI_Allreduce(&my_sum, &tot_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double M = static_cast<double>(expect);
    const double expect_sum = M * (M - 1.0) * 0.5;
    CHECK_ROOT(tot_sum == expect_sum);

    // 4) Backward: roundtrip bit-exact.
    std::vector<double> slab_back(static_cast<size_t>(nlocal), -1.0);
    redist.ksec_to_slab(ksec.data(), slab_back.data());
    bool ok_round = true;
    for (long long t = 0; t < nlocal; ++t) {
        if (slab_back[t] != slab[t]) { ok_round = false; break; }
    }
    CHECK_ROOT(ok_round);

    if (rank == 0) {
        std::printf("  case N=%d nproc=%d : nrecv_rank0=%d, total=%lld (expect %lld)\n",
                    N, nranks, nrecv, tot_nrecv, expect);
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (rank == 0)
        std::printf("== test_ksection_redistribute (nranks=%d) ==\n", nranks);

    // N must be divisible enough that every rank gets >=1 slab and the
    // ksec partition lands on cell boundaries. For nproc up to 12 we use N=24.
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
