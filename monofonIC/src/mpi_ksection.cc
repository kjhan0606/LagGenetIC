// This file is part of monofonIC (MUSIC2)
// Port of cuRamses ksection.f90 -- static, uniform-weight path only.
#include "mpi_ksection.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ksection {

KSectionTree::KSectionTree(int nproc,
                           const std::array<double, 3>& box,
                           const std::array<int, 3>& ngrid)
    : nproc_(nproc), box_(box), ngrid_(ngrid) {
    if (nproc_ < 1) throw std::invalid_argument("KSectionTree: nproc must be >= 1");
    for (int d = 0; d < 3; ++d) {
        if (box_[d] <= 0.0)
            throw std::invalid_argument("KSectionTree: box length must be positive");
        if (ngrid_[d] < 1)
            throw std::invalid_argument("KSectionTree: ngrid must be >= 1");
    }
}

void KSectionTree::set_cell_alignment(int factor) {
    if (factor < 1)
        throw std::invalid_argument(
            "KSectionTree::set_cell_alignment: factor must be >= 1");
    cell_align_ = factor;
}

void KSectionTree::factorize() {
    factor_.clear();
    int n = nproc_;
    std::vector<std::pair<int, int>> primes_mults; // (prime, multiplicity)
    int i = 2;
    while (1LL * i * i <= n) {
        if (n % i == 0) {
            int m = 0;
            while (n % i == 0) { ++m; n /= i; }
            primes_mults.emplace_back(i, m);
        }
        ++i;
    }
    if (n > 1) primes_mults.emplace_back(n, 1);

    // Sort primes descending (matches cuRamses convention: split by largest first).
    std::sort(primes_mults.begin(), primes_mults.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (const auto& pm : primes_mults) {
        for (int r = 0; r < pm.second; ++r) factor_.push_back(pm.first);
    }
    kmax_ = factor_.empty() ? 1 : factor_.front();
}

void KSectionTree::assign_directions() {
    // Longest-axis selection in cell units (matches cuRamses scale_val behavior:
    // the relative extent along each axis after the splits already applied).
    dir_.assign(factor_.size(), 0);
    std::array<double, 3> extent;
    for (int d = 0; d < 3; ++d) extent[d] = static_cast<double>(ngrid_[d]);

    for (size_t lvl = 0; lvl < factor_.size(); ++lvl) {
        int maxdir = 0;
        for (int d = 1; d < 3; ++d) {
            if (extent[d] > extent[maxdir]) maxdir = d;
        }
        dir_[lvl] = maxdir;
        extent[maxdir] /= static_cast<double>(factor_[lvl]);
    }
}

int KSectionTree::count_nbinodes() const {
    int tot = 1; // root
    long long nc = 1;
    for (int k : factor_) {
        nc *= k;
        tot += static_cast<int>(nc);
    }
    return tot;
}

int KSectionTree::level_of_node(int node) const {
    // Walk down level cursors to find which level `node` lives at.
    if (node == 0) return 0;
    int cur_start = 0;
    long long nc = 1;
    for (size_t lvl = 0; lvl <= factor_.size(); ++lvl) {
        int next_start = cur_start + static_cast<int>(nc);
        if (node < next_start) return static_cast<int>(lvl);
        cur_start = next_start;
        if (lvl < factor_.size()) nc *= factor_[lvl];
    }
    throw std::out_of_range("KSectionTree::level_of_node: node id out of range");
}

void KSectionTree::build() {
    factorize();
    assign_directions();
    nbinodes_ = count_nbinodes();
    root_ = 0;

    wall_.assign(nbinodes_, {});
    next_.assign(nbinodes_, {});
    indx_.assign(nbinodes_, -1);
    cpumin_.assign(nbinodes_, 0);
    cpumax_.assign(nbinodes_, -1);

    std::vector<std::array<double, 3>> bxmin(nbinodes_), bxmax(nbinodes_);
    std::vector<int> imin(nbinodes_, 0), imax(nbinodes_, -1);

    for (int d = 0; d < 3; ++d) { bxmin[root_][d] = 0.0; bxmax[root_][d] = box_[d]; }
    imin[root_] = 0;
    imax[root_] = nproc_ - 1;

    int cur_levelstart = 0;
    long long nc = 1;

    for (size_t lvl = 0; lvl < factor_.size(); ++lvl) {
        const int k = factor_[lvl];
        const int dir = dir_[lvl];

        for (long long i = 0; i < nc; ++i) {
            const int cur = cur_levelstart + static_cast<int>(i);
            if (imax[cur] < imin[cur]) continue; // empty subtree (shouldn't happen here)

            const int lncpu = imax[cur] - imin[cur] + 1;
            if (lncpu == 1) {
                // Leaf; do not split. Children stay zero, indx_ assigned at end.
                continue;
            }

            const int base = lncpu / k;
            const int rem  = lncpu % k;

            // Walls (k-1 of them) at uniform CPU-count fractions.
            wall_[cur].assign(k - 1, 0.0);
            int cum = 0;
            for (int j = 0; j < k - 1; ++j) {
                cum += (j < rem) ? (base + 1) : base;
                const double frac = static_cast<double>(cum) / static_cast<double>(lncpu);
                double wall = bxmin[cur][dir] +
                              frac * (bxmax[cur][dir] - bxmin[cur][dir]);
                if (cell_align_ > 1) {
                    // Snap wall to a multiple of (cell_align_ * dx_dir) so the
                    // cell-space partition is aligned. Clamp to keep walls
                    // strictly monotone and leave at least one alignment unit
                    // per remaining wall + final piece.
                    const double dx = box_[dir] / static_cast<double>(ngrid_[dir]);
                    const double step = static_cast<double>(cell_align_) * dx;
                    wall = std::round(wall / step) * step;
                    const double lower =
                        (j > 0 ? wall_[cur][j - 1] : bxmin[cur][dir]) + step;
                    const double upper =
                        bxmax[cur][dir] - step * static_cast<double>(k - 1 - j);
                    if (wall < lower) wall = lower;
                    if (wall > upper) wall = upper;
                    if (wall < lower || wall > upper)
                        throw std::runtime_error(
                            "KSectionTree: cell alignment cannot fit walls "
                            "(subtree too small for requested alignment)");
                }
                wall_[cur][j] = wall;
            }

            // Children: contiguous block of k nodes starting at base_offset.
            next_[cur].assign(k, 0);
            const int base_child = cur_levelstart + static_cast<int>(nc) +
                                   static_cast<int>(i) * k;
            int cum2 = 0;
            for (int j = 0; j < k; ++j) {
                const int child = base_child + j;
                next_[cur][j] = child;

                bxmin[child] = bxmin[cur];
                bxmax[child] = bxmax[cur];
                if (j > 0)       bxmin[child][dir] = wall_[cur][j - 1];
                if (j < k - 1)   bxmax[child][dir] = wall_[cur][j];

                const int ncpu_j = (j < rem) ? (base + 1) : base;
                imin[child] = imin[cur] + cum2;
                cum2 += ncpu_j;
                imax[child] = imin[cur] + cum2 - 1;
            }
        }

        cur_levelstart += static_cast<int>(nc);
        nc *= k;
    }

    // All leaves at all levels: any node with imin==imax (and either at the final
    // level cursor or never split because it was already singleton) gets its rank id.
    for (int node = 0; node < nbinodes_; ++node) {
        if (imax[node] < imin[node]) continue;
        cpumin_[node] = imin[node];
        cpumax_[node] = imax[node];
        if (imin[node] == imax[node]) {
            indx_[node] = imin[node];
            // ensure no children for leaves
            next_[node].clear();
            wall_[node].clear();
        }
    }
}

int KSectionTree::cpu_for_point(double x, double y, double z) const {
    const std::array<double, 3> p{x, y, z};
    int cur = root_;
    int lvl = 0;
    while (!next_[cur].empty()) {
        const int k = factor_[lvl];
        const int dir = dir_[lvl];
        int child_idx = k - 1; // default: last partition (coord >= all walls)
        for (int j = 0; j < k - 1; ++j) {
            if (p[dir] <= wall_[cur][j]) { child_idx = j; break; }
        }
        cur = next_[cur][child_idx];
        ++lvl;
    }
    return indx_[cur];
}

int KSectionTree::cpu_for_cell(int ix, int iy, int iz) const {
    // Cell-center coords. Note: cells exactly on a wall (rare in real data because
    // walls land on CPU-fraction boundaries, not cell faces) go to the lower side
    // because cmp uses '<='.
    const double dx = box_[0] / ngrid_[0];
    const double dy = box_[1] / ngrid_[1];
    const double dz = box_[2] / ngrid_[2];
    return cpu_for_point((ix + 0.5) * dx, (iy + 0.5) * dy, (iz + 0.5) * dz);
}

std::array<std::array<double, 2>, 3> KSectionTree::bbox_for_rank(int rank) const {
    if (rank < 0 || rank >= nproc_)
        throw std::out_of_range("bbox_for_rank: rank " + std::to_string(rank) +
                                " out of [0," + std::to_string(nproc_) + ")");
    // Walk the tree following cpumin/cpumax containment.
    int cur = root_;
    std::array<std::array<double, 2>, 3> bb;
    for (int d = 0; d < 3; ++d) { bb[d][0] = 0.0; bb[d][1] = box_[d]; }
    int lvl = 0;
    while (!next_[cur].empty()) {
        const int k = factor_[lvl];
        const int dir = dir_[lvl];
        int chosen = -1;
        for (int j = 0; j < k; ++j) {
            const int c = next_[cur][j];
            if (rank >= cpumin_[c] && rank <= cpumax_[c]) { chosen = j; break; }
        }
        if (chosen < 0)
            throw std::logic_error("bbox_for_rank: rank not found in any child");
        if (chosen > 0)     bb[dir][0] = wall_[cur][chosen - 1];
        if (chosen < k - 1) bb[dir][1] = wall_[cur][chosen];
        cur = next_[cur][chosen];
        ++lvl;
    }
    return bb;
}

} // namespace ksection
