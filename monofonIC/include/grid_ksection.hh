// This file is part of monofonIC (MUSIC2)
// Per-rank container for cells owned under a KSectionTree partition.
//
// Holds a flat array of cell values. The cells' global (ix,iy,iz) indices
// live in the SlabRedistributor (recv_gix/giy/giz) so we don't duplicate
// them here; callers read indices from the redistributor and values from
// this container. Order is the receive order produced by
// SlabRedistributor::slab_to_ksec, and is the order expected back when
// feeding ksec_to_slab.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "mpi_ksection.hh"

namespace ksection {

template <typename T>
class Grid_KSection {
public:
    Grid_KSection() = default;

    void resize(std::size_t n) {
        data_.assign(n, T{});
    }

    std::size_t n_local() const { return data_.size(); }

    T*       data()       { return data_.data(); }
    const T* data() const { return data_.data(); }

    std::vector<T>&   data_vec()       { return data_; }

    void set_meta(const std::array<int, 3>& n_global,
                  const std::array<double, 3>& box) {
        n_global_ = n_global;
        box_      = box;
    }
    const std::array<int, 3>&    n_global() const { return n_global_; }
    const std::array<double, 3>& box()      const { return box_; }

private:
    std::vector<T>   data_;
    std::array<int, 3>    n_global_ = {0, 0, 0};
    std::array<double, 3> box_      = {0.0, 0.0, 0.0};
};

} // namespace ksection
