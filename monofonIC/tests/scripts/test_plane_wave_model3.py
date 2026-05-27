#!/usr/bin/env python3
"""
Analytical regression test for plane-wave model 3.
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
    sin_2x = np.sin(2.0 * kmode * qx)
    cos_2x = np.cos(2.0 * kmode * qx)
    cos_2y = np.cos(2.0 * kmode * qy)
    cos_2z = np.cos(2.0 * kmode * qz)

    gradients = {}
    gradients["grad_phi1_x"] = kmode * cos_x * (sin_y + sin_z) * g1

    gradients["grad_phi2_x"] = (
        -0.5 * (kmode ** 3) * sin_2x * (1.0 + sin_y * sin_z) * g2
    )

    gradients["grad_phi3a_x"] = (
        (1.0 / 60.0)
        * (kmode ** 5)
        * cos_x
        * (sin_y + sin_z)
        * (7.0 - 9.0 * cos_2x + 10.0 * sin_y * sin_z)
        * g3a
    )

    gradients["grad_phi3b_x"] = (
        -(1.0 / 560.0)
        * (kmode ** 5)
        * cos_x
        * (sin_y + sin_z)
        * (
            -253.0
            + 177.0 * cos_2x
            + 14.0 * (cos_2y + cos_2z)
            + (-257.0 + 45.0 * cos_2x) * sin_y * sin_z
        )
        * g3b
    )

    block_c = (
        294.0 * np.sin(kmode * (2.0 * qx - qy))
        + 84.0 * np.sin(kmode * qy)
        + 504.0 * np.sin(3.0 * kmode * qy)
        - 294.0 * np.sin(kmode * (2.0 * qx + qy))
        - 45.0 * np.sin(kmode * (2.0 * qx - qy - 2.0 * qz))
        + 830.0 * np.sin(kmode * (qy - 2.0 * qz))
        + 45.0 * np.sin(kmode * (2.0 * qx + qy - 2.0 * qz))
        + 294.0 * np.sin(kmode * (2.0 * qx - qz))
        - 45.0 * np.sin(kmode * (2.0 * qx - 2.0 * qy - qz))
        - 830.0 * np.sin(kmode * (2.0 * qy - qz))
        + 84.0 * np.sin(kmode * qz)
        + 504.0 * np.sin(3.0 * kmode * qz)
        - 294.0 * np.sin(kmode * (2.0 * qx + qz))
        + 5.0
        * (
            9.0 * np.sin(kmode * (2.0 * qx - 2.0 * qy + qz))
            + 166.0 * np.sin(kmode * (2.0 * qy + qz))
            + 9.0 * np.sin(kmode * (2.0 * (qx + qy) + qz))
            - 9.0 * np.sin(kmode * (2.0 * qx - qy + 2.0 * qz))
            + 166.0 * np.sin(kmode * (qy + 2.0 * qz))
            + 9.0 * np.sin(kmode * (2.0 * qx + qy + 2.0 * qz))
            - 9.0 * np.sin(2.0 * kmode * (qx + qy) - kmode * qz)
        )
    )

    gradients["curl_A3c_x"] = (
        (1.0 / 6720.0) * (kmode ** 5) * np.cos(kmode * qx) * block_c * g3c
    )

    return gradients


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate plane-wave model 3 output.")
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

    print("Plane-wave model 3 diagnostics match analytical expectations.")


if __name__ == "__main__":
    main()
