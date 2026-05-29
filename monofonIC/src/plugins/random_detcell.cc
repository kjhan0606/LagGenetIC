// Deterministic per-cell white noise generator.
//
// Each Fourier-space cell (i,j,k) of an N^3 grid is hashed independently:
//   cell_idx = i + N*(j + N*k)
//   seed64   = SplitMix64(global_seed ^ level*MIX ^ cell_idx)
//   phase    = SplitMix64(seed64)            -> uniform [0,1)
//   ampl_u   = SplitMix64(seed64 ^ ~0)       -> uniform (0,1)
// Then phase -> 2*pi*phase, ampl -> sqrt(-log(ampl_u)) (Rayleigh), and the
// resulting complex zrand is written following the same Hermitian-symmetry
// pattern as NGENIC.
//
// Properties:
//   * OMP- and MPI-deterministic: result depends only on (global_seed, level,
//     grid resolution, lattice), not on thread count or rank decomposition.
//   * Two SplitMix64 hashes per cell ~= 8 ns/cell; ~1 s for 512^3 at 1 thread.
//   * Not bit-compatible with NGENIC. Select with `[random] generator = DetCell`.
//
// Config:
//   [random]
//   generator     = DetCell
//   seed          = <long>          # global seed
//   DetCellLevel  = <int, default 0> # mixed into seed; varying it gives an
//                                    # independent IC realization at the same
//                                    # (seed, resolution, lattice).

#include <general.hh>
#include <random_plugin.hh>
#include <config_file.hh>

#include <cmath>
#include <cstdint>

#include <grid_fft.hh>

namespace {

inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline double u64_to_unit_double(uint64_t v)
{
    // top 53 bits -> [0,1). Standard Lemire-style trick.
    return (v >> 11) * (1.0 / static_cast<double>(1ULL << 53));
}

} // anonymous namespace

class RNG_detcell : public RNG_plugin
{
private:
    uint64_t global_seed_;
    uint64_t level_;
    size_t nres_;

public:
    explicit RNG_detcell(config_file &cf) : RNG_plugin(cf)
    {
        global_seed_ = static_cast<uint64_t>(cf.get_value<long>("random", "seed"));
        level_       = static_cast<uint64_t>(
            cf.get_value_safe<int>("random", "DetCellLevel", 0));
        nres_        = cf.get_value<size_t>("setup", "GridRes");
    }

    virtual ~RNG_detcell() {}

    bool isMultiscale() const { return false; }

    void Fill_Grid(Grid_FFT<real_t> &g)
    {
        g.zero();
        g.FourierTransformForward(false);

        const uint64_t N    = static_cast<uint64_t>(nres_);
        const uint64_t gmix = global_seed_ ^ (level_ * 0x9E3779B97F4A7C15ull);

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < nres_; ++i)
        {
            for (size_t j = 0; j < nres_; ++j)
            {
                const size_t ii  = (i > 0) ? nres_ - i : 0;
                const size_t jj  = (j > 0) ? nres_ - j : 0;
                const size_t ip  = i  - g.local_1_start_;
                const size_t iip = ii - g.local_1_start_;
                const bool i_in_range  = (i  >= size_t(g.local_1_start_) &&
                                          i  <  size_t(g.local_1_start_ + g.local_1_size_));
                const bool ii_in_range = (ii >= size_t(g.local_1_start_) &&
                                          ii <  size_t(g.local_1_start_ + g.local_1_size_));
                if (!(i_in_range || ii_in_range)) continue;

                for (size_t k = 0; k < g.size(2); ++k)
                {
                    if (i == nres_/2 || j == nres_/2 || k == nres_/2) continue;
                    if (i == 0 && j == 0 && k == 0) continue;

                    const uint64_t cell_idx =
                        static_cast<uint64_t>(i) +
                        N * (static_cast<uint64_t>(j) +
                             N * static_cast<uint64_t>(k));
                    const uint64_t s = splitmix64(gmix ^ cell_idx);
                    const double u_phase = u64_to_unit_double(splitmix64(s));

                    // amplitude: redraw if exactly 0 or 1 (matches NGENIC behaviour).
                    uint64_t sa = s ^ 0xFFFFFFFFFFFFFFFFull;
                    double u_ampl = 0.0;
                    do {
                        sa = splitmix64(sa);
                        u_ampl = u64_to_unit_double(sa);
                    } while (u_ampl == 0.0 || u_ampl == 1.0);

                    const double phase = u_phase * 2.0 * M_PI;
                    const double ampl  = std::sqrt(-std::log(u_ampl));
                    const ccomplex_t zrand(ampl * std::cos(phase),
                                           ampl * std::sin(phase));

                    if (k > 0) {
                        if (i_in_range) g.kelem(ip, j, k) = zrand;
                    } else { // k == 0 plane: enforce Hermitian symmetry
                        if (g.is_distributed()) {
                            if (j == 0) {
                                if (i < nres_/2) {
                                    if (i_in_range)  g.kelem(ip,  jj, k) = zrand;
                                    if (ii_in_range) g.kelem(iip, j,  k) = std::conj(zrand);
                                }
                            } else if (j < nres_/2) {
                                if (i_in_range)  g.kelem(ip,  j,  k) = zrand;
                                if (ii_in_range) g.kelem(iip, jj, k) = std::conj(zrand);
                            }
                        } else {
                            if (i == 0) {
                                if (j < nres_/2 && i_in_range) {
                                    g.kelem(ip, j,  k) = zrand;
                                    g.kelem(ip, jj, k) = std::conj(zrand);
                                }
                            } else if (i < nres_/2) {
                                if (i_in_range)  g.kelem(ip,  j,  k) = zrand;
                                if (ii_in_range) g.kelem(iip, jj, k) = std::conj(zrand);
                            }
                        }
                    }
                }
            }
        }
    }
};

namespace
{
RNG_plugin_creator_concrete<RNG_detcell> creator("DetCell");
}
