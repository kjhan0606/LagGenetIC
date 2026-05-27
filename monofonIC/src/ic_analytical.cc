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

#include <ic_analytical.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <logger.hh>

#if defined(USE_MPI)
#include <mpi.h>
#endif

namespace analytical
{

namespace
{

std::string lowercase(std::string value)
{
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline std::string component_suffix(int idim)
{
    return (idim == 0) ? "_x" : (idim == 1 ? "_y" : "_z");
}

} // namespace

bool configure_plane_wave(config_file &cfg,
                          double boxlen,
                          double g1,
                          Grid_FFT<real_t> &phi,
                          PlaneWaveState &state)
{
    const std::string testing_mode = cfg.get_value_safe<std::string>("testing", "test", "none");
    if (testing_mode != "plane_wave")
    {
        state.context.enabled = false;
        state.gradients_dumped = false;
        state.diag_file_prepared = false;
        return false;
    }

    const std::string section = "testing_plane_wave";
    state.context.model = cfg.get_value_safe<int>(section, "model", 0);

    const bool auto_fill_axes = cfg.get_value_safe<bool>(section, "auto_fill_axes", state.context.model > 0);

    std::array<int, 3> mode_idx = {
        cfg.get_value_safe<int>(section, "mode_x", 1),
        cfg.get_value_safe<int>(section, "mode_y", 0),
        cfg.get_value_safe<int>(section, "mode_z", 0)};

    if (state.context.model > 0 && auto_fill_axes)
    {
        if (mode_idx[1] == 0)
            mode_idx[1] = mode_idx[0];
        if (mode_idx[2] == 0)
            mode_idx[2] = mode_idx[0];
    }

    const double phase_shift = cfg.get_value_safe<double>(section, "phase", 0.0);
    const double cell_offset = cfg.get_value_safe<double>(section, "cell_offset", 0.0);

    vec3_t<double> kvec({0.0, 0.0, 0.0});
    for (int idim = 0; idim < 3; ++idim)
    {
        kvec[idim] = (2.0 * M_PI / boxlen) * static_cast<double>(mode_idx[idim]);
    }
    const double k_squared = kvec.norm_squared();

    const bool any_mode_nonzero = std::any_of(mode_idx.begin(), mode_idx.end(), [](int value) { return value != 0; });
    if (!any_mode_nonzero)
    {
        music::elog << "Plane-wave test requested but the provided mode vector is zero." << std::endl;
        throw std::runtime_error("plane_wave testing mode requires a non-zero wave vector");
    }

    std::string waveform = lowercase(cfg.get_value_safe<std::string>(section, "waveform", "cosine"));
    if (state.context.model != 0)
    {
        waveform = "sine";
    }
    else if (waveform != "sine" && waveform != "cosine")
    {
        music::elog << "Unsupported waveform '" << waveform << "' requested for plane_wave test." << std::endl;
        throw std::runtime_error("plane_wave waveform must be either 'sine' or 'cosine'");
    }

    const bool has_phi_amp = cfg.contains_key(section, "phi_amplitude");
    const bool has_psi_amp = cfg.contains_key(section, "psi_amplitude");
    double phi_amplitude = 0.0;
    if (state.context.model == 0)
    {
        if (k_squared == 0.0)
        {
            music::elog << "Plane-wave test requested but the provided mode vector is zero." << std::endl;
            throw std::runtime_error("plane_wave testing mode requires a non-zero wave vector");
        }

        if (has_phi_amp)
        {
            phi_amplitude = cfg.get_value_safe<double>(section, "phi_amplitude", 0.0);
        }
        else if (has_psi_amp)
        {
            const double psi_amplitude = cfg.get_value_safe<double>(section, "psi_amplitude", 0.0);
            const double k_norm = std::sqrt(k_squared);
            phi_amplitude = psi_amplitude / k_norm;
        }
        else
        {
            const double delta_amplitude = cfg.get_value_safe<double>(section, "delta_amplitude", 1.0);
            phi_amplitude = -delta_amplitude / k_squared;
        }
    }
    else
    {
        const double default_amp = has_phi_amp ? 0.0 : 1.0;
        phi_amplitude = cfg.get_value_safe<double>(section, "phi_amplitude", default_amp);
    }

    const double k_norm = std::sqrt(k_squared);
    const double delta_amplitude = (state.context.model == 0)
                                       ? -phi_amplitude * k_squared
                                       : std::numeric_limits<double>::quiet_NaN();
    const double psi_amplitude = (state.context.model == 0)
                                     ? phi_amplitude * k_norm
                                     : std::numeric_limits<double>::quiet_NaN();

    if (CONFIG::MPI_task_rank == 0)
    {
        if (state.context.model == 0)
        {
            music::ilog << "Plane-wave test: overriding phi using model=0"
                        << ", mode=(" << mode_idx[0] << "," << mode_idx[1] << "," << mode_idx[2] << ")"
                        << ", |k|=" << k_norm
                        << ", waveform=" << waveform
                        << ", phi_amp=" << phi_amplitude
                        << ", delta_amp=" << delta_amplitude
                        << ", psi_amp=" << psi_amplitude << std::endl;
        }
        else
        {
            music::ilog << "Plane-wave test: overriding phi using model=" << state.context.model
                        << ", mode=(" << mode_idx[0] << "," << mode_idx[1] << "," << mode_idx[2] << ")"
                        << ", waveform=sine"
                        << ", phi_amp=" << phi_amplitude << std::endl;
        }
    }

    phi.FourierTransformBackward(false);

    const size_t local_nx = phi.size(0);
    const size_t ny = phi.size(1);
    const size_t nz = phi.size(2);
    const auto dx = phi.get_dx();

    auto eval_model = [&](const vec3_t<double> &r) {
        const double phase = kvec.dot(r) + phase_shift;
        if (state.context.model == 0)
        {
            return (waveform == "sine") ? std::sin(phase) : std::cos(phase);
        }

        const double phase_x = kvec[0] * r[0] + phase_shift;
        const double phase_y = kvec[1] * r[1] + phase_shift;
        const double phase_z = kvec[2] * r[2] + phase_shift;

        if (state.context.model == 1)
        {
            return std::sin(phase_x) + std::sin(phase_y) + std::sin(phase_z);
        }

        if (state.context.model == 2)
        {
            return std::sin(phase_x) + std::sin(phase_y) * std::sin(phase_z);
        }

        if (state.context.model == 3)
        {
            return std::sin(phase_x) * (std::sin(phase_y) + std::sin(phase_z));

        }

        music::elog << "plane_wave model " << state.context.model << " is not implemented." << std::endl;
        throw std::runtime_error("plane_wave model not implemented");
    };

#pragma omp parallel for collapse(3)
    for (size_t il = 0; il < local_nx; ++il)
    {
        for (size_t j = 0; j < ny; ++j)
        {
            for (size_t kidx = 0; kidx < nz; ++kidx)
            {
                const size_t idx = phi.get_idx(il, j, kidx);
                auto r = phi.get_r<double>(il, j, kidx);
                r[0] += cell_offset * dx[0];
                r[1] += cell_offset * dx[1];
                r[2] += cell_offset * dx[2];
                phi.relem(idx) = static_cast<real_t>(phi_amplitude * eval_model(r));
            }
        }
    }

    phi.FourierTransformForward();
    phi.zero_DC_mode();

    state.context.enabled = true;
    state.context.kvec = kvec;
    state.context.phi_amplitude = phi_amplitude;
    state.context.phase = phase_shift;
    state.context.cell_offset = cell_offset;
    state.context.waveform = waveform;
    state.context.growth_factor = g1;
    state.context.spacing = {static_cast<double>(phi.get_dx(0)),
                             static_cast<double>(phi.get_dx(1)),
                             static_cast<double>(phi.get_dx(2))};

    state.gradients_dumped = false;
    state.diag_file_prepared = false;

    return true;
}

void dump_plane_wave_gradients(const config_file &cfg,
                               PlaneWaveState &state,
                               int LPTorder,
                               Grid_FFT<real_t> &phi,
                               Grid_FFT<real_t> &phi2,
                               Grid_FFT<real_t> &phi3a,
                               Grid_FFT<real_t> &phi3b,
                               const std::array<Grid_FFT<real_t> *, 3> &A3,
                               Grid_FFT<real_t> &tmp,
                               const gradient_eval_fn &gradient_eval)
{
    if (!state.context.enabled || state.gradients_dumped)
        return;

    const std::string diag_default = cfg.get_value_safe<std::string>("output", "filename", "plane_wave_testing.hdf5");
    const std::string diag_filename_cfg = cfg.get_value_safe<std::string>("testing_plane_wave", "diagnostic_filename", diag_default);
    const std::string diag_filename = cfg.get_path_relative_to_config(diag_filename_cfg);

#if defined(USE_HDF5)
    if (!state.diag_file_prepared)
    {
        if (CONFIG::MPI_task_rank == 0)
        {
            ::unlink(diag_filename.c_str());
        }
#if defined(USE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        state.diag_file_prepared = true;
    }

    auto write_gradient = [&](Grid_FFT<real_t> &potential, const std::string &prefix) {
        for (int idim = 0; idim < 3; ++idim)
        {
            tmp.FourierTransformForward(false);

#pragma omp parallel for
            for (size_t i = 0; i < phi.size(0); ++i)
            {
                for (size_t j = 0; j < phi.size(1); ++j)
                {
                    for (size_t k = 0; k < phi.size(2); ++k)
                    {
                        const size_t idx = phi.get_idx(i, j, k);
                        std::array<size_t, 3> ijk = tmp.get_k3(i, j, k);
                        tmp.kelem(idx) = gradient_eval(idim, ijk) * potential.kelem(idx);
                    }
                }
            }

            tmp.zero_DC_mode();
            tmp.FourierTransformBackward();
            const std::string dataset = prefix + component_suffix(idim);
            tmp.Write_to_HDF5(diag_filename, dataset);
            if (CONFIG::MPI_task_rank == 0)
            {
                music::ilog << "Plane-wave test: wrote " << dataset << " to " << diag_filename << std::endl;
            }
        }
    };

    auto write_curl = [&](const std::string &prefix) {
        for (int idim = 0; idim < 3; ++idim)
        {
            const int idimp = (idim + 1) % 3;
            const int idimpp = (idim + 2) % 3;
            tmp.FourierTransformForward(false);

#pragma omp parallel for
            for (size_t i = 0; i < phi.size(0); ++i)
            {
                for (size_t j = 0; j < phi.size(1); ++j)
                {
                    for (size_t k = 0; k < phi.size(2); ++k)
                    {
                        const size_t idx = phi.get_idx(i, j, k);
                        std::array<size_t, 3> ijk = tmp.get_k3(i, j, k);
                        tmp.kelem(idx) = gradient_eval(idimp, ijk) * A3[idimpp]->kelem(idx)
                                       - gradient_eval(idimpp, ijk) * A3[idimp]->kelem(idx);
                    }
                }
            }

            tmp.zero_DC_mode();
            tmp.FourierTransformBackward();
            const std::string dataset = prefix + component_suffix(idim);
            tmp.Write_to_HDF5(diag_filename, dataset);
            if (CONFIG::MPI_task_rank == 0)
            {
                music::ilog << "Plane-wave test: wrote " << dataset << " to " << diag_filename << std::endl;
            }
        }
    };

    write_gradient(phi, "grad_phi1");
    if (LPTorder > 1 && phi2.is_allocated())
    {
        write_gradient(phi2, "grad_phi2");
    }
    if (LPTorder > 2 && phi3a.is_allocated())
    {
        write_gradient(phi3a, "grad_phi3a");
        write_gradient(phi3b, "grad_phi3b");
        write_curl("curl_A3c");
    }
#else
    if (CONFIG::MPI_task_rank == 0)
    {
        music::wlog << "Plane-wave gradients requested but HDF5 support is disabled; skipping diagnostic output." << std::endl;
    }
#endif

    state.gradients_dumped = true;
}

} // namespace analytical

