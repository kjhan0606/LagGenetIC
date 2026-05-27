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
#include <functional>
#include <string>

#include <config_file.hh>
#include <general.hh>
#include <grid_fft.hh>

namespace analytical
{

struct PlaneWaveContext
{
    bool enabled{false};
    int model{0};
    vec3_t<double> kvec{0.0, 0.0, 0.0};
    double phi_amplitude{0.0};
    double phase{0.0};
    double cell_offset{0.0};
    std::array<double, 3> spacing{{0.0, 0.0, 0.0}};
    std::string waveform{"sine"};
    double growth_factor{1.0};
};

struct PlaneWaveState
{
    PlaneWaveContext context;
    bool gradients_dumped{false};
    bool diag_file_prepared{false};
};

bool configure_plane_wave(config_file &cfg,
                          double boxlen,
                          double g1,
                          Grid_FFT<real_t> &phi,
                          PlaneWaveState &state);

using gradient_eval_fn = std::function<ccomplex_t(int, const std::array<size_t, 3> &)>;

void dump_plane_wave_gradients(const config_file &cfg,
                               PlaneWaveState &state,
                               int LPTorder,
                               Grid_FFT<real_t> &phi,
                               Grid_FFT<real_t> &phi2,
                               Grid_FFT<real_t> &phi3a,
                               Grid_FFT<real_t> &phi3b,
                               const std::array<Grid_FFT<real_t> *, 3> &A3,
                               Grid_FFT<real_t> &tmp,
                               const gradient_eval_fn &gradient_eval);

} // namespace analytical

