// This file is part of monofonIC (MUSIC2)
// K-section recursive domain-decomposition tree.
// Port of cuRamses ksection.f90 (build_from_scratch path).
// Static, uniform-CPU-weight only: no histograms, no load-feedback rebuild.
#pragma once

#include <array>
#include <vector>

namespace ksection {

//! Recursive k-section tree (real-space xyz partition).
//! nproc is prime-factorized as p_0 * p_1 * ... (descending primes); each
//! level splits the longest current axis into k=p_i equal-CPU pieces.
//! For uniform weights, walls are placed by CPU-count fractions, identical
//! to cuRamses build_ksection with update=.false.
class KSectionTree {
public:
    KSectionTree(int nproc,
                 const std::array<double, 3>& box,
                 const std::array<int, 3>& ngrid);

    //! Build the tree (uniform weights, no MPI required).
    void build();

    //! Constrain all walls to fall on multiples of `factor * (box/ngrid)`
    //! along their split direction. With factor=2, every rank's cell-space
    //! bbox starts and ends at an even index, so a 2-aligned 2x2x2 stencil
    //! over any selected cell stays inside the block (no halo needed for
    //! the masked-mass mean computation).
    //!
    //! Must be called before build(). factor=1 (default) preserves the
    //! cuRamses-equivalent partition.
    void set_cell_alignment(int factor);
    int  cell_alignment() const { return cell_align_; }

    //! Map a real-space point to its owning MPI rank (0-based).
    int cpu_for_point(double x, double y, double z) const;

    //! Map a grid cell to its owning MPI rank (cell-center coords).
    int cpu_for_cell(int ix, int iy, int iz) const;

    //! Axis-aligned bounding box for a given rank, [min,max] per axis.
    std::array<std::array<double, 2>, 3> bbox_for_rank(int rank) const;

    // Accessors for tests / diagnostics.
    int nproc() const { return nproc_; }
    int nlevels() const { return static_cast<int>(factor_.size()); }
    int kmax() const { return kmax_; }
    int nbinodes() const { return nbinodes_; }
    const std::vector<int>& factors() const { return factor_; }
    const std::vector<int>& directions() const { return dir_; }
    //! Wall positions at internal node `node`, returns k-1 boundaries along dir_[level].
    const std::vector<double>& walls_of_node(int node) const { return wall_[node]; }
    //! Child node indices at `node` (size k = factor_[its_level]); 0 if leaf/unused.
    const std::vector<int>& next_of_node(int node) const { return next_[node]; }
    //! CPU id at leaf `node` (0-based); -1 if not a leaf.
    int leaf_cpu(int node) const { return indx_[node]; }
    int cpumin(int node) const { return cpumin_[node]; }
    int cpumax(int node) const { return cpumax_[node]; }

private:
    int nproc_;
    std::array<double, 3> box_;
    std::array<int, 3> ngrid_;

    std::vector<int> factor_;   //!< per-level split factor, length nlevels
    std::vector<int> dir_;      //!< per-level split direction (0,1,2)
    int kmax_ = 0;
    int nbinodes_ = 0;
    int root_ = 0;

    // Flat per-node storage; node ids are 0-based, 0..nbinodes_-1.
    std::vector<std::vector<double>> wall_;   //!< [node][0..k-1)  size k-1 for internals, empty for leaves
    std::vector<std::vector<int>>    next_;   //!< [node][0..k)    child node ids; all 0 for leaves
    std::vector<int> indx_;                   //!< rank for leaves, -1 for internals
    std::vector<int> cpumin_;                 //!< inclusive CPU range per subtree
    std::vector<int> cpumax_;

    int cell_align_ = 1;                      //!< wall snapping in cell units

    void factorize();
    void assign_directions();
    int  count_nbinodes() const;
    int  level_of_node(int node) const;
};

} // namespace ksection
