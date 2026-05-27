#!/usr/bin/env python3
"""
Shared helpers for plane-wave analytical regression tests.
"""
from __future__ import annotations

import configparser
import math
import pathlib
from typing import Callable, Dict, Tuple

import numpy as np


# Default Planck2018EE+BAO+SN matter density used in monofonIC examples
DEFAULT_OMEGA_M = 0.3099


def _build_parser(config_path: pathlib.Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    with config_path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    return parser


def read_simulation_parameters(config_path: pathlib.Path) -> Dict[str, float]:
    """
    Fetch the subset of configuration values needed for the analytical checks.
    """
    parser = _build_parser(config_path)
    try:
        box_length = parser.getfloat("setup", "boxlength")
        zstart = parser.getfloat("setup", "zstart")
    except (configparser.NoSectionError, configparser.NoOptionError) as exc:
        raise RuntimeError(f"Missing setup parameters in {config_path}") from exc

    omega_m = parser.getfloat("cosmology", "omega_m", fallback=DEFAULT_OMEGA_M)
    omega_de = parser.getfloat("cosmology", "omega_de", fallback=1.0 - omega_m)

    return {
        "box_length": box_length,
        "zstart": zstart,
        "omega_m": omega_m,
        "omega_de": omega_de,
    }


def _require_hypergeometric() -> Callable[[float, float, float, float], float]:
    """
    Provide a hypergeometric 2F1 evaluator backed by scipy or mpmath.
    """
    try:
        from scipy import special as _special  # type: ignore

        def _hyp(a: float, b: float, c: float, z: float) -> float:
            return float(_special.hyp2f1(a, b, c, z))

        return _hyp
    except ImportError:
        try:
            import mpmath  # type: ignore

            def _hyp(a: float, b: float, c: float, z: float) -> float:
                return float(mpmath.hyp2f1(a, b, c, z))

            return _hyp
        except ImportError as exc:
            raise RuntimeError(
                "Hypergeometric evaluation requires scipy or mpmath. "
                "Install one with `pip install mpmath`."
            ) from exc


def compute_growth_factors(zstart: float, omega_m: float, omega_de: float) -> Tuple[float, float, float, float, float]:
    """
    Reproduce the analytical growth-factor approximations from the validation notebooks.
    """
    lam = omega_de / omega_m
    a = 1.0 / (1.0 + zstart)
    g1 = a - (2.0 * lam / 11.0) * (a ** 4)

    hyper = _require_hypergeometric()
    g1_norm = math.sqrt(1.0 + lam) * hyper(1.5, 5.0 / 6.0, 11.0 / 6.0, -lam)
    g1 /= g1_norm

    g2 = (
        -(3.0 / 7.0) * g1 ** 2
        - (3.0 * lam / 1001.0) * g1 ** 5
        - (960.0 * lam ** 2 / 3_556_553.0) * g1 ** 8
    )
    g3a = (
        -(1.0 / 3.0) * g1 ** 3
        - (4.0 * lam / 825.0) * g1 ** 6
        - (109.0 * lam ** 2 / 215_985.0) * g1 ** 9
    )
    g3b = (
        (10.0 / 21.0) * g1 ** 3
        + (538.0 * lam / 75_075.0) * g1 ** 6
        + (3_581.0 * lam ** 2 / 4_894_845.0) * g1 ** 9
    )
    g3c = (
        -(1.0 / 7.0) * g1 ** 3
        - (2.0 * lam / 1001.0) * g1 ** 6
        - (320.0 * lam ** 2 / 1_524_237.0) * g1 ** 9
    )
    return g1, g2, g3a, g3b, g3c


def coordinate_grid(box_length: float, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return qx, qy, qz meshes covering [0, box_length) at the grid resolution.
    """
    coords = np.linspace(0.0, box_length, resolution, endpoint=False)
    return np.meshgrid(coords, coords, coords, indexing="ij")


def summarize_field(
    name: str,
    numerical: np.ndarray,
    analytical: np.ndarray,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-6,
) -> None:
    """
    Compare numerical and analytical data and raise if tolerances are exceeded.
    """
    matches = np.allclose(numerical, analytical, rtol=rtol, atol=atol)
    diff = numerical - analytical
    max_abs = float(np.max(np.abs(diff)))
    print(f"{name:>16s} : max_abs={max_abs:.3e}")
    if not matches:
        raise AssertionError(
            f"{name} failed tolerance check (max_abs={max_abs:.3e}, "
            f"rtol={rtol:.1e}, atol={atol:.1e})"
        )
