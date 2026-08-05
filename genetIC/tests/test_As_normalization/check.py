#!/usr/bin/env python3
"""Compare equivalent A_s and sigma_8 normalizations through the full binary."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent
DEFAULT_BINARY = PACKAGE / "genetIC" / "genetIC"
TRANSFER = PACKAGE / "genetIC" / "tests" / "camb_transfer_kmax40_z0_post2015.dat"

# The CAMB table, h=0.701, n_s=0.96, and k_p=0.05 Mpc^-1 give
# sigma_8=0.817 for this primordial amplitude at z=0.
SIGMA8 = 0.817
AS_EQUIVALENT = 1.7894007672836557e-9
PIVOT = 0.05


def run_case(binary: Path, root: Path, name: str, normalization: str) -> np.ndarray:
    output = root / name
    output.mkdir()
    template = (HERE / "param.template.txt").read_text()
    params = template.replace("@NORMALIZATION@", normalization)
    params = params.replace("@TRANSFER@", str(TRANSFER))
    params = params.replace("@OUTDIR@", str(output))
    parameter_file = root / f"{name}.txt"
    parameter_file.write_text(params)

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [str(binary), str(parameter_file)],
        cwd=root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Normalization case {name} failed with return code "
            f"{result.returncode}.\nOutput:\n{result.stdout}"
        )
    return np.loadtxt(output / "normalization_0.ps")


def run_invalid_case(
    binary: Path,
    root: Path,
    name: str,
    normalization: str,
    expected_message: str,
) -> None:
    output = root / name
    output.mkdir()
    template = (HERE / "param.template.txt").read_text()
    params = template.replace("@NORMALIZATION@", normalization)
    params = params.replace("@TRANSFER@", str(TRANSFER))
    params = params.replace("@OUTDIR@", str(output))
    parameter_file = root / f"{name}.txt"
    parameter_file.write_text(params)

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [str(binary), str(parameter_file)],
        cwd=root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode == 0 or expected_message not in result.stdout:
        raise RuntimeError(
            f"Invalid normalization case {name} was not rejected as expected.\n"
            f"Return code: {result.returncode}\nOutput:\n{result.stdout}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", nargs="?", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()

    with tempfile.TemporaryDirectory(prefix="genetic_As_test_") as directory:
        root = Path(directory)
        sigma8_result = run_case(binary, root, "sigma8", f"s8 {SIGMA8:.17g}")
        As_result = run_case(
            binary,
            root,
            "As",
            f"A_s {AS_EQUIVALENT:.17g}\nk_p {PIVOT:.17g}",
        )
        run_invalid_case(
            binary,
            root,
            "both",
            f"s8 {SIGMA8:.17g}\nA_s {AS_EQUIVALENT:.17g}",
            "Specify only one of s8 or A_s",
        )
        run_invalid_case(
            binary,
            root,
            "neither",
            "# no normalization",
            "Specify exactly one of s8 or A_s before the camb command",
        )

    if sigma8_result.shape != As_result.shape:
        raise RuntimeError(
            f"Power-spectrum shapes differ: {sigma8_result.shape} versus {As_result.shape}"
        )

    relative = np.abs(As_result[:, 2:4] / sigma8_result[:, 2:4] - 1.0)
    maximum = float(np.nanmax(relative))
    if not np.allclose(As_result, sigma8_result, rtol=3.0e-6, atol=1.0e-14):
        ratio = As_result[:, 2:4] / sigma8_result[:, 2:4]
        raise RuntimeError(
            f"A_s and sigma_8 power spectra differ by up to {maximum:.3e}. "
            f"The median power ratio is {np.nanmedian(ratio):.16e}"
        )

    print(
        f"PASS: A_s={AS_EQUIVALENT:.16e} and sigma_8={SIGMA8} agree "
        f"to max relative difference {maximum:.3e}"
    )


if __name__ == "__main__":
    main()
