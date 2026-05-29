// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// 
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <array>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <utility>

#include <general.hh>
#include <grid_fft.hh>

/// @brief base class for convolutions of two or three fields
/// @tparam data_t 
/// @tparam derived_t 
template <typename data_t, typename derived_t>
class BaseConvolver
{
protected:
    std::array<size_t, 3> np_;
    std::array<real_t, 3> length_;

public:

    /// @brief Construct a new Base Convolver object
    /// @param N linear grid size
    /// @param L physical box size
    BaseConvolver(const std::array<size_t, 3> &N, const std::array<real_t, 3> &L)
        : np_(N), length_(L) {}


    /// @brief Construct a new Base Convolver object [deleted copy constructor]  
    BaseConvolver( const BaseConvolver& ) = delete;
    
    /// @brief destructor (virtual)
    virtual ~BaseConvolver() {}

    /// @brief implements convolution of two Fourier-space fields
    /// @tparam kfunc1 field 1
    /// @tparam kfunc2 field 2
    /// @tparam opp output operator
    template <typename kfunc1, typename kfunc2, typename opp>
    void convolve2(kfunc1 kf1, kfunc2 kf2, opp op) {}

    /// @brief implements convolution of three Fourier-space fields
    /// @tparam kfunc1 field 1
    /// @tparam kfunc2 field 2
    /// @tparam kfunc3 field 3
    /// @tparam opp output operator
    template <typename kfunc1, typename kfunc2, typename kfunc3, typename opp>
    void convolve3(kfunc1 kf1, kfunc2 kf2, kfunc3 kf3, opp op) {}

    /// @brief Hessian-cache hook. Default: caching not supported (returns nullptr).
    /// OrszagConvolver overrides this with a real cache lookup/build.
    Grid_FFT<data_t>* try_get_hessian(Grid_FFT<data_t>& /*field*/, const std::array<int,2>& /*idx*/)
    { return nullptr; }

    // Stubs so dispatcher templates compile for derived classes without cache support.
    // The cached branch is gated by `try_get_hessian() != nullptr`, so these are never
    // reached at runtime for non-Orszag derived classes.
    template <typename opp>
    void convolve2_cached(const Grid_FFT<data_t>&, const Grid_FFT<data_t>&, opp) {}
    template <typename kfunc_rhs, typename opp>
    void convolve2_left_cached(const Grid_FFT<data_t>&, kfunc_rhs, opp) {}
    template <typename opp>
    void convolve3_all_cached(const Grid_FFT<data_t>&, const Grid_FFT<data_t>&, const Grid_FFT<data_t>&, opp) {}

    /// Trim cache down to budget. Default: no-op (derived classes without cache).
    void evict_to_budget() {}

public:

    /// @brief convolve two gradient fields in Fourier space a_{,i} * b_{,j}
    /// @tparam opp output operator type
    /// @param inl left input field a
    /// @param d1l direction of first gradient (,i)
    /// @param inr right input field b
    /// @param d1r direction of second gradient (,j)
    /// @param output_op output operator
    template <typename opp>
    void convolve_Gradients(Grid_FFT<data_t> &inl, const std::array<int, 1> &d1l,
                            Grid_FFT<data_t> &inr, const std::array<int, 1> &d1r,
                            opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of two gradients
        static_cast<derived_t &>(*this).convolve2(
            // first gradient
            [&inl,&d1l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d1l[0],{i,j,k});
                return grad1*inl.kelem(i, j, k);
            },
            // second gradient
            [&inr,&d1r](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inr.gradient(d1r[0],{i,j,k});
                return grad1*inr.kelem(i, j, k);
            },
            // -> output operator
            output_op);
    }

    /// @brief convolve a gradient and a Hessian field in Fourier space a_{,i} * b_{,jk}
    /// @tparam opp output operator type
    /// @param inl left input field a
    /// @param d1l direction of gradient (,i)
    /// @param inr right input field b
    /// @param d2r directions of Hessian (,jk)
    /// @param output_op output operator
    template <typename opp>
    void convolve_Gradient_and_Hessian(Grid_FFT<data_t> &inl, const std::array<int, 1> &d1l,
                                       Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r,
                                       opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of gradient and Hessian
        static_cast<derived_t &>(*this).convolve2(
            // gradient
            [&](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto kk = inl.template get_k<real_t>(i, j, k);
                return ccomplex_t(0.0, -kk[d1l[0]]) * inl.kelem(i, j, k);
            },
            // Hessian
            [&](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto kk = inr.template get_k<real_t>(i, j, k);
                return -kk[d2r[0]] * kk[d2r[1]] * inr.kelem(i, j, k);
            },
            // -> output operator
            output_op);
    }

    /// @brief convolve two Hessian fields in Fourier space a_{,ij} * b_{,kl}
    /// @tparam opp output operator type
    /// @param inl left input field a
    /// @param d2l directions of first Hessian (,ij)
    /// @param inr right input field b
    /// @param d2r directions of second Hessian (,kl)
    /// @param output_op output operator
    template <typename opp>
    void convolve_Hessians(Grid_FFT<data_t> &inl, const std::array<int, 2> &d2l,
                           Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r,
                           opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // try cached path first
        auto& self = static_cast<derived_t &>(*this);
        self.evict_to_budget();
        auto* g1 = self.try_get_hessian(inl, d2l);
        auto* g2 = self.try_get_hessian(inr, d2r);
        if (g1 && g2) {
            self.convolve2_cached(*g1, *g2, output_op);
            return;
        }
        // perform convolution of Hessians (uncached path)
        self.convolve2(
            // first Hessian
            [&inl,&d2l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d2l[0],{i,j,k});
                auto grad2 = inl.gradient(d2l[1],{i,j,k});
                return grad1*grad2*inl.kelem(i, j, k);
            },
            // second Hessian
            [&inr,&d2r](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inr.gradient(d2r[0],{i,j,k});
                auto grad2 = inr.gradient(d2r[1],{i,j,k});
                return grad1*grad2*inr.kelem(i, j, k);
            },
            // -> output operator
            output_op);
    }

    /// @brief convolve three Hessian fields in Fourier space a_{,ij} * b_{,kl} * c_{,mn}
    /// @tparam opp output operator
    /// @param inl first input field a
    /// @param d2l directions of first Hessian (,ij)
    /// @param inm second input field b
    /// @param d2m directions of second Hessian (,kl)
    /// @param inr third input field c
    /// @param d2r directions of third Hessian (,mn)
    /// @param output_op output operator
    template <typename opp>
    void convolve_Hessians(Grid_FFT<data_t> &inl, const std::array<int, 2> &d2l,
                           Grid_FFT<data_t> &inm, const std::array<int, 2> &d2m,
                           Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r,
                           opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inm.FourierTransformForward();
        inr.FourierTransformForward();
        // try cached path first
        auto& self = static_cast<derived_t &>(*this);
        self.evict_to_budget();
        auto* g1 = self.try_get_hessian(inl, d2l);
        auto* g2 = self.try_get_hessian(inm, d2m);
        auto* g3 = self.try_get_hessian(inr, d2r);
        if (g1 && g2 && g3) {
            self.convolve3_all_cached(*g1, *g2, *g3, output_op);
            return;
        }
        // perform convolution of Hessians (uncached path)
        self.convolve3(
            // first Hessian
            [&inl, &d2l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d2l[0],{i,j,k});
                auto grad2 = inl.gradient(d2l[1],{i,j,k});
                return grad1*grad2*inl.kelem(i, j, k);
            },
            // second Hessian
            [&inm, &d2m](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inm.gradient(d2m[0],{i,j,k});
                auto grad2 = inm.gradient(d2m[1],{i,j,k});
                return grad1*grad2*inm.kelem(i, j, k);
            },
            // third Hessian
            [&inr, &d2r](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inr.gradient(d2r[0],{i,j,k});
                auto grad2 = inr.gradient(d2r[1],{i,j,k});
                return grad1*grad2*inr.kelem(i, j, k);
            },
            // -> output operator
            output_op);
    }

    /// @brief convolve Hessian field with sum of two Hessian fields in Fourier space a_{,ij} * (b_{,kl} + c_{,mn})
    /// @tparam opp output operator type
    /// @param inl left input field a
    /// @param d2l directions of first Hessian (,ij)
    /// @param inr right input field b
    /// @param d2r1 directions of second Hessian (,kl)
    /// @param d2r2 directions of third Hessian (,mn)
    /// @param output_op output operator
    template <typename opp>
    void convolve_SumOfHessians(Grid_FFT<data_t> &inl, const std::array<int, 2> &d2l,
                                Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r1, const std::array<int, 2> &d2r2,
                                opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // RHS sum functor: IFFT(a+b) preserved bit-exact vs uncached path.
        auto rhs_sum = [&inr, &d2r1, &d2r2](size_t i, size_t j, size_t k) -> ccomplex_t {
            auto grad11 = inr.gradient(d2r1[0],{i,j,k});
            auto grad12 = inr.gradient(d2r1[1],{i,j,k});
            auto grad21 = inr.gradient(d2r2[0],{i,j,k});
            auto grad22 = inr.gradient(d2r2[1],{i,j,k});
            return (grad11*grad12+grad21*grad22)*inr.kelem(i, j, k);
        };
        // Try LHS-only cached hybrid path (strict bit-exact)
        auto& self = static_cast<derived_t &>(*this);
        self.evict_to_budget();
        if (auto* g1 = self.try_get_hessian(inl, d2l)) {
            self.convolve2_left_cached(*g1, rhs_sum, output_op);
            return;
        }
        // Fully uncached path
        self.convolve2(
            [&inl, &d2l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d2l[0],{i,j,k});
                auto grad2 = inl.gradient(d2l[1],{i,j,k});
                return grad1*grad2*inl.kelem(i, j, k);
            },
            rhs_sum,
            output_op);
    }

    /// @brief convolve Hessian field with difference of two Hessian fields in Fourier space a_{,ij} * (b_{,kl} - c_{,mn})
    /// @tparam opp output operator type
    /// @param inl left input field a
    /// @param d2l directions of first Hessian (,ij)
    /// @param inr right input field b
    /// @param d2r1 directions of second Hessian (,kl)
    /// @param d2r2 directions of third Hessian (,mn)
    /// @param output_op output operator
    template <typename opp>
    void convolve_DifferenceOfHessians(Grid_FFT<data_t> &inl, const std::array<int, 2> &d2l,
                                       Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r1, const std::array<int, 2> &d2r2,
                                       opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // RHS diff functor: IFFT(a-b) preserved bit-exact vs uncached path.
        auto rhs_diff = [&inr, &d2r1, &d2r2](size_t i, size_t j, size_t k) -> ccomplex_t {
            auto grad11 = inr.gradient(d2r1[0],{i,j,k});
            auto grad12 = inr.gradient(d2r1[1],{i,j,k});
            auto grad21 = inr.gradient(d2r2[0],{i,j,k});
            auto grad22 = inr.gradient(d2r2[1],{i,j,k});
            return (grad11*grad12-grad21*grad22)*inr.kelem(i, j, k);
        };
        // Try LHS-only cached hybrid path (strict bit-exact)
        auto& self = static_cast<derived_t &>(*this);
        self.evict_to_budget();
        if (auto* g1 = self.try_get_hessian(inl, d2l)) {
            self.convolve2_left_cached(*g1, rhs_diff, output_op);
            return;
        }
        // Fully uncached path
        self.convolve2(
            [&inl, &d2l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d2l[0],{i,j,k});
                auto grad2 = inl.gradient(d2l[1],{i,j,k});
                return grad1*grad2*inl.kelem(i, j, k);
            },
            rhs_diff,
            output_op);
    }
    
    /// @brief performs the multiplication of two fields by convolving them in Fourier space
    /// @tparam opp output operator type
    /// @param inl left input field a)
    /// @param inr right input field b
    /// @param output_op output operator
    template <typename opp> // TOMA
    void multiply_field(Grid_FFT<data_t> &inl, Grid_FFT<data_t> &inr, opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of Hessians
        static_cast<derived_t &>(*this).convolve2(
            [&inl](size_t i, size_t j, size_t k) -> ccomplex_t {return  inl.kelem(i, j, k); },
            [&inr](size_t i, size_t j, size_t k) -> ccomplex_t {return  inr.kelem(i, j, k); },
            output_op);
    }
};

//! low-level implementation of convolutions -- naive convolution class, ignoring aliasing (no padding)
template <typename data_t>
class NaiveConvolver : public BaseConvolver<data_t, NaiveConvolver<data_t>>
{
protected:
    /// @brief buffer for Fourier transformed fields
    Grid_FFT<data_t> *fbuf1_, *fbuf2_;

    /// @brief number of points in each direction
    using BaseConvolver<data_t, NaiveConvolver<data_t>>::np_;

    /// @brief length of each direction
    using BaseConvolver<data_t, NaiveConvolver<data_t>>::length_;

public:
    /// @brief constructor
    /// @param N number of points in each direction
    /// @param L length of each direction
    NaiveConvolver(const std::array<size_t, 3> &N, const std::array<real_t, 3> &L)
        : BaseConvolver<data_t, NaiveConvolver<data_t>>(N, L)
    {
        fbuf1_ = new Grid_FFT<data_t>(N, length_, true, kspace_id);
        fbuf2_ = new Grid_FFT<data_t>(N, length_, true, kspace_id);
    }

    /// @brief destructor
    ~NaiveConvolver()
    {
        delete fbuf1_;
        delete fbuf2_;
    }

    /// @brief convolution of two fields
    template <typename kfunc1, typename kfunc2, typename opp>
    void convolve2(kfunc1 kf1, kfunc2 kf2, opp output_op)
    {
        //... prepare data 1
        fbuf1_->FourierTransformForward(false);
        this->copy_in(kf1, *fbuf1_);

        //... prepare data 2
        fbuf2_->FourierTransformForward(false);
        this->copy_in(kf2, *fbuf2_);

        //... convolve
        fbuf1_->FourierTransformBackward();
        fbuf2_->FourierTransformBackward();

#pragma omp parallel for
        for (size_t i = 0; i < fbuf1_->ntot_; ++i)
        {
            (*fbuf2_).relem(i) *= (*fbuf1_).relem(i);
        }
        fbuf2_->FourierTransformForward();
        // fbuf2_->dealias();
//... copy data back
#pragma omp parallel for
        for (size_t i = 0; i < fbuf2_->ntot_; ++i)
        {
            output_op(i, (*fbuf2_)[i]);
        }


    }

    /// @brief convolution of three fields
    template <typename kfunc1, typename kfunc2, typename kfunc3, typename opp>
    void convolve3(kfunc1 kf1, kfunc2 kf2, kfunc3 kf3, opp output_op)
    {
        //... prepare data 1
        fbuf1_->FourierTransformForward(false);
        this->copy_in(kf1, *fbuf1_);

        //... prepare data 2
        fbuf2_->FourierTransformForward(false);
        this->copy_in(kf2, *fbuf2_);

        //... convolve
        fbuf1_->FourierTransformBackward();
        fbuf2_->FourierTransformBackward();

#pragma omp parallel for
        for (size_t i = 0; i < fbuf1_->ntot_; ++i)
        {
            (*fbuf2_).relem(i) *= (*fbuf1_).relem(i);
        }

        //... prepare data 2
        fbuf1_->FourierTransformForward(false);
        this->copy_in(kf3, *fbuf1_);

        //... convolve
        fbuf1_->FourierTransformBackward();

#pragma omp parallel for
        for (size_t i = 0; i < fbuf1_->ntot_; ++i)
        {
            (*fbuf2_).relem(i) *= (*fbuf1_).relem(i);
        }

        fbuf2_->FourierTransformForward();
//... copy data back
#pragma omp parallel for
        for (size_t i = 0; i < fbuf2_->ntot_; ++i)
        {
            output_op(i, (*fbuf2_)[i]);
        }
    }

//--------------------------------------------------------------------------------------------------------
private:

    /// @brief copy data into a grid
    /// @tparam kfunc abstract function type generating data
    /// @param kf abstract function generating data
    /// @param g grid to copy data into
    template <typename kfunc>
    void copy_in(kfunc kf, Grid_FFT<data_t> &g)
    {
#pragma omp parallel for
        for (size_t i = 0; i < g.size(0); ++i)
        {
            for (size_t j = 0; j < g.size(1); ++j)
            {
                for (size_t k = 0; k < g.size(2); ++k)
                {
                    g.kelem(i, j, k) = kf(i, j, k);
                }
            }
        }
    }
};

//! convolution class, respecting Orszag's 3/2 rule (padding in Fourier space to avoid aliasing)
template <typename data_t>
class OrszagConvolver : public BaseConvolver<data_t, OrszagConvolver<data_t>>
{
private:
    /// @brief buffer for Fourier transformed fields
    Grid_FFT<data_t> *f1p_, *f2p_, *fbuf_;

    using BaseConvolver<data_t, OrszagConvolver<data_t>>::np_;
    using BaseConvolver<data_t, OrszagConvolver<data_t>>::length_;

    ccomplex_t *crecvbuf_; //!< receive buffer for MPI (complex)
    real_t *recvbuf_;     //!< receive buffer for MPI (real)
    size_t maxslicesz_;  //!< maximum size of a slice
    std::vector<ptrdiff_t> offsets_, offsetsp_; //!< offsets for MPI
    std::vector<size_t> sizes_, sizesp_;       //!< sizes for MPI

    // ---- Hessian cache (LRU + memory budget) ----
    // Key: (field pointer, sorted (i,j) index pair). Value: padded real-space Grid_FFT.
    // Populated lazily on first Hessian convolve for fields registered via
    // enable_hessian_cache(); freed via clear_hessian_cache().
    // LRU: lru_.back() is most recently built/hit; eviction pops front when
    // cache_used_bytes_ exceeds cache_budget_bytes_ (0 = unlimited).
    using HessianKey = std::tuple<const void*, int, int>;
    using CacheEntry = std::pair<HessianKey, std::unique_ptr<Grid_FFT<data_t>>>;
    using LruList    = std::list<CacheEntry>;
    std::set<const void*> cacheable_fields_;
    LruList lru_;
    std::map<HessianKey, typename LruList::iterator> hessian_index_;
    size_t cache_budget_bytes_ = 0;   //!< 0 = unlimited
    size_t cache_used_bytes_   = 0;

    static HessianKey make_key(const void* field, const std::array<int,2>& idx)
    {
        int a = idx[0], b = idx[1];
        if (a > b) std::swap(a, b);
        return std::make_tuple(field, a, b);
    }

    static size_t entry_bytes(const Grid_FFT<data_t>& g)
    {
        // padded real-space Grid_FFT: ntot_ counts real_t elements (FFTW R2C
        // layout with the (n2+2) stride). sizeof(real_t) gives bytes.
        return g.memsize() * sizeof(real_t);
    }

    /// @brief get task index for a given index
    /// @param index index
    /// @param offsets offsets
    /// @param sizes sizes
    /// @param ntasks number of tasks
    int get_task(ptrdiff_t index, const std::vector<ptrdiff_t> &offsets, const std::vector<size_t> &sizes, const int ntasks)
    {
        int itask = 0;
        while (itask < ntasks - 1 && offsets[itask + 1] <= index)
            ++itask;
        return itask;
    }

public:

    /// @brief constructor
    /// @param N grid size
    /// @param L grid length
    OrszagConvolver(const std::array<size_t, 3> &N, const std::array<real_t, 3> &L)
        : BaseConvolver<data_t, OrszagConvolver<data_t>>({3 * N[0] / 2, 3 * N[1] / 2, 3 * N[2] / 2}, L)
    {
        //... create temporaries
        f1p_ = new Grid_FFT<data_t>(np_, length_, true, kspace_id);
        f2p_ = new Grid_FFT<data_t>(np_, length_, true, kspace_id);
        fbuf_ = new Grid_FFT<data_t>(N, length_, true, kspace_id); // needed for MPI, or for triple conv.

#if defined(USE_MPI)
        maxslicesz_ = f1p_->sizes_[1] * f1p_->sizes_[3] * 2;

        crecvbuf_ = new ccomplex_t[maxslicesz_ / 2];
        recvbuf_ = reinterpret_cast<real_t *>(&crecvbuf_[0]);

        int ntasks(MPI::get_size());

        offsets_.assign(ntasks, 0);
        offsetsp_.assign(ntasks, 0);
        sizes_.assign(ntasks, 0);
        sizesp_.assign(ntasks, 0);

        size_t tsize = N[0], tsizep = f1p_->size(0);

        MPI_Allgather(&fbuf_->local_1_start_, 1, MPI_LONG_LONG, &offsets_[0], 1,
                      MPI_LONG_LONG, MPI_COMM_WORLD);
        MPI_Allgather(&f1p_->local_1_start_, 1, MPI_LONG_LONG, &offsetsp_[0], 1,
                      MPI_LONG_LONG, MPI_COMM_WORLD);
        MPI_Allgather(&tsize, 1, MPI_LONG_LONG, &sizes_[0], 1, MPI_LONG_LONG,
                      MPI_COMM_WORLD);
        MPI_Allgather(&tsizep, 1, MPI_LONG_LONG, &sizesp_[0], 1, MPI_LONG_LONG,
                      MPI_COMM_WORLD);
#endif
    }

    /// @brief destructor
    ~OrszagConvolver()
    {
        delete f1p_;
        delete f2p_;
        delete fbuf_;
#if defined(USE_MPI)
        delete[] crecvbuf_;
#endif
    }

    // Per-phase profile counters (rank-0 wallclock, accumulated across convolve2 calls).
    // Reset/report from the LPT driver to attribute time to phi(2), phi(3a), etc.
    inline static double t_pad    = 0.0;
    inline static double t_fftbwd = 0.0;
    inline static double t_mult   = 0.0;
    inline static double t_fftfwd = 0.0;
    inline static double t_unpad  = 0.0;
    inline static double t_cache_build = 0.0;   //!< pad+IFFT time spent populating cache entries
    inline static double t_cache_copy  = 0.0;   //!< copy/add/sub of cached padded grids
    inline static long   n_calls  = 0;
    inline static long   n_cache_hit   = 0;     //!< Hessian lookups that hit the cache
    inline static long   n_cache_miss  = 0;     //!< Hessian lookups that built a new cache entry
    inline static long   n_cache_evict = 0;     //!< entries dropped to honour the byte budget

    static void reset_profile()
    {
        t_pad = t_fftbwd = t_mult = t_fftfwd = t_unpad = 0.0;
        t_cache_build = t_cache_copy = 0.0;
        n_calls = 0;
        n_cache_hit = n_cache_miss = n_cache_evict = 0;
    }
    static void report_profile(const std::string& tag)
    {
        const double total = t_pad + t_fftbwd + t_mult + t_fftfwd + t_unpad
                           + t_cache_build + t_cache_copy;
        music::ilog << "  [conv " << tag << "] n=" << n_calls
                    << "  pad=" << t_pad << "s  fft_bwd=" << t_fftbwd
                    << "s  mult=" << t_mult << "s  fft_fwd=" << t_fftfwd
                    << "s  unpad=" << t_unpad
                    << "s  cache_build=" << t_cache_build
                    << "s  cache_copy=" << t_cache_copy
                    << "s  sum=" << total << "s"
                    << "  hits=" << n_cache_hit
                    << " misses=" << n_cache_miss
                    << " evicts=" << n_cache_evict
                    << std::endl;
    }

    // ---- Hessian cache public API ----
    /// Register a field as cacheable. Its Hessians will be populated lazily
    /// on first reference and reused across subsequent convolves until cleared.
    void enable_hessian_cache(const Grid_FFT<data_t>& field)
    {
        cacheable_fields_.insert(static_cast<const void*>(&field));
    }
    /// Free all cached Hessian grids and unregister all cacheable fields.
    void clear_hessian_cache()
    {
        lru_.clear();
        hessian_index_.clear();
        cacheable_fields_.clear();
        cache_used_bytes_ = 0;
    }
    /// Set the maximum bytes the Hessian cache may hold. 0 = unlimited (default).
    /// When the limit is exceeded after a miss-build, the least-recently-used
    /// entries are evicted until the budget is satisfied or only one entry remains.
    void set_hessian_cache_budget_bytes(size_t b) { cache_budget_bytes_ = b; }
    /// Number of currently-cached Hessian grids.
    size_t cached_hessian_count() const { return lru_.size(); }
    /// Bytes currently held by the Hessian cache.
    size_t cached_hessian_bytes() const { return cache_used_bytes_; }

    /// Trim cache LRU front entries until cache_used_bytes_ <= cache_budget_bytes_.
    /// Called by dispatchers at entry (before fresh lookups) so any pointers
    /// returned from a previous call are already dead. Keeps at least one
    /// entry so a single Hessian that exceeds the budget is still usable.
    void evict_to_budget()
    {
        if (cache_budget_bytes_ == 0) return;
        while (cache_used_bytes_ > cache_budget_bytes_ && lru_.size() > 1) {
            auto& front = lru_.front();
            cache_used_bytes_ -= entry_bytes(*front.second);
            hessian_index_.erase(front.first);
            lru_.pop_front();
            ++n_cache_evict;
        }
    }

    // ---- Cache lookup / build (CRTP override of BaseConvolver::try_get_hessian) ----
    Grid_FFT<data_t>* try_get_hessian(Grid_FFT<data_t>& field, const std::array<int,2>& idx)
    {
        const void* fp = static_cast<const void*>(&field);
        if (cacheable_fields_.count(fp) == 0) return nullptr;
        auto key = make_key(fp, idx);
        auto idx_it = hessian_index_.find(key);
        if (idx_it != hessian_index_.end()) {
            // touch: move to back of LRU (most recent).
            lru_.splice(lru_.end(), lru_, idx_it->second);
            ++n_cache_hit;
            return idx_it->second->second.get();
        }
        ++n_cache_miss;
        const double t0 = get_wtime();
        field.FourierTransformForward();
        auto g = std::make_unique<Grid_FFT<data_t>>(np_, length_, true, kspace_id);
        const std::array<int,2> d2 = idx;
        this->pad_insert(
            [&field, d2](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = field.gradient(d2[0], {i,j,k});
                auto grad2 = field.gradient(d2[1], {i,j,k});
                return grad1 * grad2 * field.kelem(i, j, k);
            },
            *g);
        g->FourierTransformBackward();   // -> padded real-space Hessian
        t_cache_build += get_wtime() - t0;
        cache_used_bytes_ += entry_bytes(*g);
        lru_.emplace_back(key, std::move(g));
        auto last_it = std::prev(lru_.end());
        hessian_index_.emplace(std::move(key), last_it);
        return last_it->second.get();
    }

    // ---- Cached convolve variants (all read-only on cached operands) ----
    // copy g2 -> f2p_, f2p_ *= g1, FFT_fwd f2p_, unpad
    template <typename opp>
    void convolve2_cached(const Grid_FFT<data_t>& g1, const Grid_FFT<data_t>& g2, opp output_op)
    {
        ++n_calls;
        double t0 = get_wtime();
        f2p_->copy_from(g2);
        t_cache_copy += get_wtime() - t0;

        t0 = get_wtime();
        #pragma omp parallel for
        for (size_t i = 0; i < f2p_->ntot_; ++i)
            f2p_->relem(i) *= g1.relem(i);
        t_mult += get_wtime() - t0;

        t0 = get_wtime();
        f2p_->FourierTransformForward();
        t_fftfwd += get_wtime() - t0;

        t0 = get_wtime();
        unpad(*f2p_, output_op);
        t_unpad += get_wtime() - t0;
    }

    // Hybrid: LHS cached (read-only padded real-space grid), RHS is an arbitrary
    // k-space functor (e.g. sum/diff of Hessians). Used by SumOf/DiffOfHessians
    // dispatchers to keep IFFT(a+b)/IFFT(a-b) bit-exact while still skipping the
    // LHS pad+IFFT. Saves 1 IFFT per call.
    template <typename kfunc_rhs, typename opp>
    void convolve2_left_cached(const Grid_FFT<data_t>& g1, kfunc_rhs kf2, opp output_op)
    {
        ++n_calls;
        double t0 = get_wtime();
        f2p_->FourierTransformForward(false);
        this->pad_insert(kf2, *f2p_);
        t_pad += get_wtime() - t0;

        t0 = get_wtime();
        f2p_->FourierTransformBackward();
        t_fftbwd += get_wtime() - t0;

        t0 = get_wtime();
        #pragma omp parallel for
        for (size_t i = 0; i < f2p_->ntot_; ++i)
            f2p_->relem(i) *= g1.relem(i);
        t_mult += get_wtime() - t0;

        t0 = get_wtime();
        f2p_->FourierTransformForward();
        t_fftfwd += get_wtime() - t0;

        t0 = get_wtime();
        unpad(*f2p_, output_op);
        t_unpad += get_wtime() - t0;
    }

    // Three-Hessian convolution with all three cached.
    // Step 1: g1 * g2 -> fbuf_ (k-space, N grid).
    // Step 2: pad_insert(fbuf_) -> f2p_, IFFT, f2p_ *= g3, FFT_fwd, unpad.
    template <typename opp>
    void convolve3_all_cached(const Grid_FFT<data_t>& g1,
                              const Grid_FFT<data_t>& g2,
                              const Grid_FFT<data_t>& g3,
                              opp output_op)
    {
        // Step 1: cached pair -> fbuf_ (k-space). Mirrors the original convolve3:
        // unpad's MPI path passes data_t (real_t), so use a generic lambda.
        fbuf_->FourierTransformForward(false);
        auto assign_to_fbuf = [this](auto i, auto v) { (*fbuf_)[i] = v; };
        convolve2_cached(g1, g2, assign_to_fbuf);

        // Step 2: fbuf_ * g3 -> output_op
        ++n_calls;
        double t0 = get_wtime();
        f2p_->FourierTransformForward(false);
        this->pad_insert(
            [this](size_t i, size_t j, size_t k) -> ccomplex_t { return fbuf_->kelem(i,j,k); },
            *f2p_);
        t_pad += get_wtime() - t0;

        t0 = get_wtime();
        f2p_->FourierTransformBackward();
        t_fftbwd += get_wtime() - t0;

        t0 = get_wtime();
        #pragma omp parallel for
        for (size_t i = 0; i < f2p_->ntot_; ++i)
            f2p_->relem(i) *= g3.relem(i);
        t_mult += get_wtime() - t0;

        t0 = get_wtime();
        f2p_->FourierTransformForward();
        t_fftfwd += get_wtime() - t0;

        t0 = get_wtime();
        unpad(*f2p_, output_op);
        t_unpad += get_wtime() - t0;
    }

    /// @brief convolve two fields
    /// @tparam kfunc1 abstract function type generating data for the first field
    /// @tparam kfunc2 abstract function type generating data for the second field
    /// @tparam opp abstract function type for the output operation
    template <typename kfunc1, typename kfunc2, typename opp>
    void convolve2(kfunc1 kf1, kfunc2 kf2, opp output_op)
    {
        ++n_calls;
        double t0;

        //... prepare data 1 + 2 (pad_insert does the FourierInterpolateCopyTo lift to high-res)
        t0 = get_wtime();
        f1p_->FourierTransformForward(false);
        this->pad_insert(kf1, *f1p_);
        f2p_->FourierTransformForward(false);
        this->pad_insert(kf2, *f2p_);
        t_pad += get_wtime() - t0;

        //... convolve
        t0 = get_wtime();
        f1p_->FourierTransformBackward();
        f2p_->FourierTransformBackward();
        t_fftbwd += get_wtime() - t0;

        t0 = get_wtime();
#pragma omp parallel for
        for (size_t i = 0; i < f1p_->ntot_; ++i)
        {
            (*f2p_).relem(i) *= (*f1p_).relem(i);
        }
        t_mult += get_wtime() - t0;

        t0 = get_wtime();
        f2p_->FourierTransformForward();
        t_fftfwd += get_wtime() - t0;

        //... copy data back (unpad does the FourierInterpolateCopyTo drop to low-res)
        t0 = get_wtime();
        unpad(*f2p_, output_op);
        t_unpad += get_wtime() - t0;
    }

    /// @brief convolve three fields
    /// @tparam kfunc1 abstract function type generating data for the first field
    /// @tparam kfunc2 abstract function type generating data for the second field
    /// @tparam kfunc3 abstract function type generating data for the third field
    /// @tparam opp abstract function type for the output operation
    template <typename kfunc1, typename kfunc2, typename kfunc3, typename opp>
    void convolve3(kfunc1 kf1, kfunc2 kf2, kfunc3 kf3, opp output_op)
    {
        auto assign_to = [](auto &g) { return [&](auto i, auto v) { g[i] = v; }; };
        fbuf_->FourierTransformForward(false);
        convolve2(kf1, kf2, assign_to(*fbuf_));
        convolve2([&](size_t i, size_t j, size_t k) -> ccomplex_t { return fbuf_->kelem(i, j, k); }, kf3, output_op);
    }

    // template< typename opp >
    // void test_pad_unpad( Grid_FFT<data_t> & in, Grid_FFT<data_t> & res, opp op )
    // {
    //     //... prepare data 1
    //     f1p_->FourierTransformForward(false);
    //     this->pad_insert( [&in]( size_t i, size_t j, size_t k ){return in.kelem(i,j,k);}, *f1p_ );
    //     f1p_->FourierTransformBackward();
    //     f1p_->FourierTransformForward();
    //     res.FourierTransformForward();
    //     unpad(*f1p_, res, op);
    // }

private:

    /// @brief unpad the result of a convolution and copy it to a grid
    /// @tparam kdep_functor abstract function type generating data for the result
    /// @param kfunc abstract function generating data for the result
    /// @param fp grid to copy the result to
    template <typename kdep_functor>
    void pad_insert( kdep_functor kfunc, Grid_FFT<data_t> &fp)
    {
        const real_t rfac = std::pow(1.5, 1.5);

#if !defined(USE_MPI)
        const size_t nhalf[3] = {fp.n_[0] / 3, fp.n_[1] / 3, fp.n_[2] / 3};

        fp.zero();

        #pragma omp parallel for
        for (size_t i = 0; i < 2 * fp.size(0) / 3; ++i)
        {
            size_t ip = (i > nhalf[0]) ? i + nhalf[0] : i;
            for (size_t j = 0; j < 2 * fp.size(1) / 3; ++j)
            {
                size_t jp = (j > nhalf[1]) ? j + nhalf[1] : j;
                for (size_t k = 0; k < nhalf[2]+1; ++k)
                {
                    size_t kp = (k > nhalf[2]) ? k + nhalf[2] : k;
                    fp.kelem(ip, jp, kp) = kfunc(i, j, k) * rfac;
                }
            }
        }
#else
        fbuf_->FourierTransformForward(false);
        
        #pragma omp parallel for
        for (size_t i = 0; i < fbuf_->size(0); ++i)
        {
            for (size_t j = 0; j < fbuf_->size(1); ++j)
            {
                for (size_t k = 0; k < fbuf_->size(2); ++k)
                {
                    fbuf_->kelem(i, j, k) = kfunc(i, j, k) * rfac;
                }
            }
        }

        fbuf_->FourierInterpolateCopyTo( fp );
        
#endif //defined(USE_MPI)
    }

    /// @brief unpad the result of a convolution and write it to an output operator
    /// @tparam operator_t abstract function type for the output operation
    /// @param fp grid to copy the result from
    /// @param output_op abstract function to write the result to
    template <typename operator_t>
    void unpad( Grid_FFT<data_t> &fp, operator_t output_op)
    {
        const real_t rfac = std::sqrt(fp.n_[0] * fp.n_[1] * fp.n_[2]) / std::sqrt(fbuf_->n_[0] * fbuf_->n_[1] * fbuf_->n_[2]);

        // make sure we're in Fourier space...
        assert(fp.space_ == kspace_id);

#if !defined(USE_MPI) ////////////////////////////////////////////////////////////////////////////////////
        fbuf_->FourierTransformForward(false);
        size_t nhalf[3] = {fbuf_->n_[0] / 2, fbuf_->n_[1] / 2, fbuf_->n_[2] / 2};

        #pragma omp parallel for
        for (size_t i = 0; i < fbuf_->size(0); ++i)
        {
            size_t ip = (i > nhalf[0]) ? i + nhalf[0] : i;
            for (size_t j = 0; j < fbuf_->size(1); ++j)
            {
                size_t jp = (j > nhalf[1]) ? j + nhalf[1] : j;
                for (size_t k = 0; k < fbuf_->size(2); ++k)
                {
                    size_t kp = (k > nhalf[2]) ? k + nhalf[2] : k;
                    fbuf_->kelem(i, j, k) = fp.kelem(ip, jp, kp) / rfac;
                    // zero Nyquist modes since they are not unique after convolution
                    if( i==nhalf[0]||j==nhalf[1]||k==nhalf[2]){
                        fbuf_->kelem(i, j, k) = 0.0; 
                    }
                }
            }
        }

        //... copy data back
        #pragma omp parallel for
        for (size_t i = 0; i < fbuf_->ntot_; ++i)
        {
            output_op(i, (*fbuf_)[i]);
        }

#else /// then USE_MPI is defined //////////////////////////////////////////////////////////////
    
        fp.FourierInterpolateCopyTo( *fbuf_ );

        //... copy data back
        #pragma omp parallel for
        for (size_t i = 0; i < fbuf_->ntot_; ++i)
        {

            output_op(i, (*fbuf_)[i] / rfac);
        }

#endif //defined(USE_MPI)
    }
};
