#!/usr/bin/env python3
"""Check the analytic A_s factor applied by the tabulated CAMB plugin."""

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


HUBBLE = 0.701
NS = 0.96
AS = 1.7894007672836557e-9
PIVOT = 0.05


def generate(executable: Path, config: Path, output: Path, cwd: Path) -> np.ndarray:
    result = subprocess.run(
        [str(executable), "--generate", "CAMB_file", str(config), str(output)],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Transfer generation for {config.name} failed with return code "
            f"{result.returncode}.\nOutput:\n{result.stdout}"
        )
    return np.loadtxt(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("As_config", type=Path)
    parser.add_argument("sigma8_config", type=Path)
    parser.add_argument("transfer_file", type=Path)
    args = parser.parse_args()

    executable = args.executable.resolve()
    As_config = args.As_config.resolve()
    sigma8_config = args.sigma8_config.resolve()
    transfer_file = args.transfer_file.resolve()

    with tempfile.TemporaryDirectory(prefix="monofonic_camb_As_") as directory:
        work = Path(directory)
        shutil.copy2(transfer_file, work / transfer_file.name)
        As_values = generate(executable, As_config, work / "As.txt", work)
        raw_values = generate(executable, sigma8_config, work / "raw.txt", work)

    if As_values.shape != raw_values.shape:
        raise RuntimeError(
            f"Transfer output shapes differ: {As_values.shape} versus {raw_values.shape}"
        )

    pivot_h = PIVOT / HUBBLE
    tnorm = math.sqrt(
        2.0
        * math.pi
        * math.pi
        * AS
        * (1.0 / pivot_h) ** (NS - 1.0)
        / (2.0 * math.pi) ** 3
    )
    expected = HUBBLE * HUBBLE * tnorm

    valid = np.isfinite(As_values[:, 1:]) & np.isfinite(raw_values[:, 1:])
    valid &= np.abs(raw_values[:, 1:]) > 1.0e-30
    ratio = As_values[:, 1:][valid] / raw_values[:, 1:][valid]
    maximum = float(np.max(np.abs(ratio / expected - 1.0)))
    if not np.allclose(ratio, expected, rtol=2.0e-12, atol=0.0):
        raise RuntimeError(
            f"CAMB-file A_s factor is inconsistent with the analytic value by {maximum:.3e}"
        )

    print(
        f"PASS: CAMB-file A_s factor {expected:.16e} agrees across all transfer columns "
        f"to {maximum:.3e} relative"
    )


if __name__ == "__main__":
    main()
