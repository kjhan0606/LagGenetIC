// Stand-alone unit tests for KSectionTree.
// Cross-checked against cuRamses ksection.f90 build_from_scratch path
// for uniform-CPU-weight splits.
#include "mpi_ksection.hh"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace {

int g_pass = 0, g_fail = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (cond) { ++g_pass; }                                                \
        else {                                                                 \
            ++g_fail;                                                          \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                                      \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        auto _va = (a); auto _vb = (b);                                        \
        if (_va == _vb) { ++g_pass; }                                          \
        else {                                                                 \
            ++g_fail;                                                          \
            std::fprintf(stderr, "FAIL %s:%d  %s == %s  (lhs=%lld rhs=%lld)\n",\
                         __FILE__, __LINE__, #a, #b,                           \
                         (long long)_va, (long long)_vb);                      \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        double _va = (a), _vb = (b);                                           \
        if (std::fabs(_va - _vb) <= (tol)) { ++g_pass; }                       \
        else {                                                                 \
            ++g_fail;                                                          \
            std::fprintf(stderr, "FAIL %s:%d  |%s - %s| <= %g  (%.17g vs %.17g)\n", \
                         __FILE__, __LINE__, #a, #b, (double)(tol), _va, _vb); \
        }                                                                      \
    } while (0)

void test_factorization() {
    using namespace ksection;
    std::printf("== test_factorization ==\n");
    {
        KSectionTree t(12, {1.0, 1.0, 1.0}, {64, 64, 64});
        t.build();
        // 12 = 2^2 * 3 -> descending primes: [3, 2, 2]
        const auto& f = t.factors();
        CHECK_EQ(f.size(), size_t{3});
        CHECK_EQ(f[0], 3);
        CHECK_EQ(f[1], 2);
        CHECK_EQ(f[2], 2);
        CHECK_EQ(t.kmax(), 3);
    }
    {
        KSectionTree t(8, {1.0, 1.0, 1.0}, {64, 64, 64});
        t.build();
        const auto& f = t.factors();
        CHECK_EQ(f.size(), size_t{3});
        for (int i : f) CHECK_EQ(i, 2);
    }
    {
        KSectionTree t(7, {1.0, 1.0, 1.0}, {64, 64, 64});
        t.build();
        const auto& f = t.factors();
        CHECK_EQ(f.size(), size_t{1});
        CHECK_EQ(f[0], 7);
    }
    {
        KSectionTree t(1, {1.0, 1.0, 1.0}, {64, 64, 64});
        t.build();
        CHECK_EQ(t.factors().size(), size_t{0});
        CHECK_EQ(t.cpu_for_point(0.5, 0.5, 0.5), 0);
    }
}

void test_uniform_walls_4cpu() {
    using namespace ksection;
    std::printf("== test_uniform_walls_4cpu ==\n");
    // nproc=4 = 2*2, cubic box, all axes equal -> dir0=x, dir1=y.
    KSectionTree t(4, {100.0, 100.0, 100.0}, {64, 64, 64});
    t.build();
    CHECK_EQ(t.factors().size(), size_t{2});
    CHECK_EQ(t.factors()[0], 2);
    CHECK_EQ(t.factors()[1], 2);
    CHECK_EQ(t.directions()[0], 0);
    CHECK_EQ(t.directions()[1], 1);

    // Root wall: x = 50.0
    CHECK_EQ(t.walls_of_node(0).size(), size_t{1});
    CHECK_NEAR(t.walls_of_node(0)[0], 50.0, 1e-12);

    // Level-1 nodes (children of root) at node ids 1,2; each splits y at 50.0.
    CHECK_NEAR(t.walls_of_node(1)[0], 50.0, 1e-12);
    CHECK_NEAR(t.walls_of_node(2)[0], 50.0, 1e-12);

    // CPU assignment: cpu0 = (x<50, y<50), cpu1 = (x<50, y>=50),
    //                 cpu2 = (x>=50, y<50), cpu3 = (x>=50, y>=50)
    CHECK_EQ(t.cpu_for_point(25.0, 25.0, 50.0), 0);
    CHECK_EQ(t.cpu_for_point(25.0, 75.0, 50.0), 1);
    CHECK_EQ(t.cpu_for_point(75.0, 25.0, 50.0), 2);
    CHECK_EQ(t.cpu_for_point(75.0, 75.0, 50.0), 3);

    // bbox for each rank
    auto b0 = t.bbox_for_rank(0);
    CHECK_NEAR(b0[0][0],  0.0, 1e-12); CHECK_NEAR(b0[0][1],  50.0, 1e-12);
    CHECK_NEAR(b0[1][0],  0.0, 1e-12); CHECK_NEAR(b0[1][1],  50.0, 1e-12);
    CHECK_NEAR(b0[2][0],  0.0, 1e-12); CHECK_NEAR(b0[2][1], 100.0, 1e-12);
    auto b3 = t.bbox_for_rank(3);
    CHECK_NEAR(b3[0][0], 50.0, 1e-12); CHECK_NEAR(b3[0][1], 100.0, 1e-12);
    CHECK_NEAR(b3[1][0], 50.0, 1e-12); CHECK_NEAR(b3[1][1], 100.0, 1e-12);
    CHECK_NEAR(b3[2][0],  0.0, 1e-12); CHECK_NEAR(b3[2][1], 100.0, 1e-12);
}

void test_unequal_split_6cpu() {
    using namespace ksection;
    std::printf("== test_unequal_split_6cpu ==\n");
    // 6 = 3 * 2; cubic box. dir0=x (k=3), dir1=y (k=2, since after /3, y and z tie -> y wins).
    KSectionTree t(6, {100.0, 100.0, 100.0}, {64, 64, 64});
    t.build();
    CHECK_EQ(t.factors()[0], 3);
    CHECK_EQ(t.factors()[1], 2);
    CHECK_EQ(t.directions()[0], 0);
    CHECK_EQ(t.directions()[1], 1);

    // Walls at root: x at 100/3 and 200/3 (CPU fractions 2/6 and 4/6).
    const auto& w = t.walls_of_node(0);
    CHECK_EQ(w.size(), size_t{2});
    CHECK_NEAR(w[0], 100.0 / 3.0, 1e-12);
    CHECK_NEAR(w[1], 200.0 / 3.0, 1e-12);

    // Each of the 3 x-slabs splits y at 50.
    for (int child = 1; child <= 3; ++child) {
        const auto& wy = t.walls_of_node(child);
        CHECK_EQ(wy.size(), size_t{1});
        CHECK_NEAR(wy[0], 50.0, 1e-12);
    }

    // 6 distinct ranks reached
    std::set<int> seen;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            const double x = (i + 0.5) * 100.0 / 6.0;
            const double y = (j + 0.5) * 100.0 / 6.0;
            seen.insert(t.cpu_for_point(x, y, 50.0));
        }
    }
    CHECK_EQ(seen.size(), size_t{6});
    CHECK_EQ(*seen.begin(), 0);
    CHECK_EQ(*seen.rbegin(), 5);
}

void test_uneven_remainder_5cpu() {
    using namespace ksection;
    std::printf("== test_uneven_remainder_5cpu ==\n");
    // 5 is prime -> single level, k=5. Uniform CPU fractions: walls at 1/5, 2/5, 3/5, 4/5.
    KSectionTree t(5, {100.0, 100.0, 100.0}, {64, 64, 64});
    t.build();
    CHECK_EQ(t.factors().size(), size_t{1});
    CHECK_EQ(t.factors()[0], 5);
    CHECK_EQ(t.kmax(), 5);
    const auto& w = t.walls_of_node(0);
    CHECK_EQ(w.size(), size_t{4});
    for (int j = 0; j < 4; ++j) {
        CHECK_NEAR(w[j], 100.0 * (j + 1) / 5.0, 1e-12);
    }
    for (int r = 0; r < 5; ++r) {
        const double x = (r + 0.5) * 100.0 / 5.0;
        CHECK_EQ(t.cpu_for_point(x, 50.0, 50.0), r);
    }
}

void test_remainder_distribution_7cpu() {
    using namespace ksection;
    std::printf("== test_remainder_distribution_7cpu ==\n");
    // 7 is prime -> single level, k=7. Walls at r/7 for r=1..6.
    KSectionTree t(7, {7.0, 7.0, 7.0}, {64, 64, 64});
    t.build();
    const auto& w = t.walls_of_node(0);
    CHECK_EQ(w.size(), size_t{6});
    for (int j = 0; j < 6; ++j) CHECK_NEAR(w[j], double(j + 1), 1e-12);
    // 7 distinct
    std::set<int> seen;
    for (int r = 0; r < 7; ++r) seen.insert(t.cpu_for_point(r + 0.5, 3.5, 3.5));
    CHECK_EQ(seen.size(), size_t{7});
}

void test_axis_picking_anisotropic() {
    using namespace ksection;
    std::printf("== test_axis_picking_anisotropic ==\n");
    // ngrid=(32,128,64), nproc=2 -> longest axis is y -> dir0=1.
    KSectionTree t(2, {1.0, 1.0, 1.0}, {32, 128, 64});
    t.build();
    CHECK_EQ(t.directions()[0], 1);
    CHECK_NEAR(t.walls_of_node(0)[0], 0.5, 1e-12);
    CHECK_EQ(t.cpu_for_point(0.5, 0.25, 0.5), 0);
    CHECK_EQ(t.cpu_for_point(0.5, 0.75, 0.5), 1);
}

void test_full_cell_coverage() {
    using namespace ksection;
    std::printf("== test_full_cell_coverage ==\n");
    // Every cell maps to a valid rank in [0,nproc); each rank's count
    // matches what we expect from CPU-fraction walls.
    const int N = 32;
    const int nproc = 8;
    KSectionTree t(nproc, {1.0, 1.0, 1.0}, {N, N, N});
    t.build();
    std::vector<long long> cnt(nproc, 0);
    for (int iz = 0; iz < N; ++iz)
        for (int iy = 0; iy < N; ++iy)
            for (int ix = 0; ix < N; ++ix) {
                int r = t.cpu_for_cell(ix, iy, iz);
                CHECK(r >= 0 && r < nproc);
                cnt[r] += 1;
            }
    long long total = 0;
    for (auto c : cnt) total += c;
    CHECK_EQ(total, (long long)N * N * N);
    // For nproc=8 = 2^3 on a 32^3 grid with uniform splits, each rank owns 32^3/8 = 4096 cells exactly.
    for (int r = 0; r < nproc; ++r) CHECK_EQ(cnt[r], 4096LL);
}

void test_invalid_inputs() {
    using namespace ksection;
    std::printf("== test_invalid_inputs ==\n");
    bool threw = false;
    try { KSectionTree t(0, {1.0, 1.0, 1.0}, {1, 1, 1}); } catch (...) { threw = true; }
    CHECK(threw);
    threw = false;
    try { KSectionTree t(2, {0.0, 1.0, 1.0}, {1, 1, 1}); } catch (...) { threw = true; }
    CHECK(threw);
    threw = false;
    try { KSectionTree t(2, {1.0, 1.0, 1.0}, {0, 1, 1}); } catch (...) { threw = true; }
    CHECK(threw);

    KSectionTree t(4, {1.0, 1.0, 1.0}, {8, 8, 8});
    t.build();
    threw = false;
    try { (void)t.bbox_for_rank(99); } catch (...) { threw = true; }
    CHECK(threw);
}

} // namespace

int main() {
    test_factorization();
    test_uniform_walls_4cpu();
    test_unequal_split_6cpu();
    test_uneven_remainder_5cpu();
    test_remainder_distribution_7cpu();
    test_axis_picking_anisotropic();
    test_full_cell_coverage();
    test_invalid_inputs();

    std::printf("\nresults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
