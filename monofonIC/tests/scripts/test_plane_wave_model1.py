#!/usr/bin/env python3
"""
Analytical regression test for plane-wave model 1.
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np

from plane_wave_test_utils import (
    compute_growth_factors,
    coordinate_grid,
    read_simulation_parameters,
    summarize_field,
)


def build_expectations(
    box_length: float,
    resolution: int,
    g1: float,
    g2: float,
    g3a: float,
    g3b: float,
    g3c: float,
) -> dict[str, np.ndarray]:
    qx, qy, qz = coordinate_grid(box_length, resolution)
    kmode = 2.0 * np.pi / box_length
    sin_x = np.sin(kmode * qx)
    sin_y = np.sin(kmode * qy)
    sin_z = np.sin(kmode * qz)
    cos_x = np.cos(kmode * qx)
    cos_2y = np.cos(2.0 * kmode * qy)
    cos_2z = np.cos(2.0 * kmode * qz)

    gradients = {}
    gradients["grad_phi1_x"] = kmode * cos_x * g1

    gradients["grad_phi2_x"] = (
        -0.5 * (kmode ** 3) * cos_x * (sin_y + sin_z) * g2
    )

    gradients["grad_phi3a_x"] = (
        (1.0 / 3.0) * (kmode ** 5) * cos_x * sin_y * sin_z * g3a
    )

    term1 = 2.0 * np.sin(2.0 * kmode * qx) * sin_y
    term2 = 10.0 - cos_2y - cos_2z + 4.0 * (sin_x + 5.0 * sin_y) * sin_z
    gradients["grad_phi3b_x"] = (
        (1.0 / 40.0) * (kmode ** 5) * (term1 + cos_x * term2) * g3b
    )

    gradients["curl_A3c_x"] = (
        -(1.0 / 10.0)
        * (kmode ** 5)
        * cos_x
        * (cos_2y + cos_2z + sin_x * (sin_y + sin_z))
        * g3c
    )

    return gradients


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate plane-wave model 1 output.")
    parser.add_argument("--diag", type=pathlib.Path, required=True, help="diagnostic HDF5 file")
    parser.add_argument("--config", type=pathlib.Path, required=True, help="config file path")
    args = parser.parse_args()

    params = read_simulation_parameters(args.config)
    box_length = params["box_length"]
    g1, g2, g3a, g3b, g3c = compute_growth_factors(params["zstart"], params["omega_m"], params["omega_de"])

    with h5py.File(args.diag, "r") as handle:
        resolution = handle["grad_phi1_x"].shape[0]
        expectations = build_expectations(box_length, resolution, g1, g2, g3a, g3b, g3c)
        for dataset in ("grad_phi1_x", "grad_phi2_x", "grad_phi3a_x", "grad_phi3b_x", "curl_A3c_x"):
            summarize_field(dataset, handle[dataset][...], expectations[dataset])

    print("Plane-wave model 1 diagnostics match analytical expectations.")


if __name__ == "__main__":
    main()
