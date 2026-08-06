from __future__ import annotations

import numpy as np

from measure_inverted_voids import (
    find_local_minimum,
    first_compensation_radius,
    outermost_threshold_radius,
    periodic_mean,
    periodic_radii,
)


def test_periodic_mean_across_box_face() -> None:
    centre, concentration = periodic_mean(np.asarray([0.99, 0.01]))
    assert min(abs(centre), abs(centre - 1.0)) < 1.0e-12
    assert concentration > 0.99


def test_profile_radii() -> None:
    radii = np.asarray([2.0, 4.0, 6.0, 8.0])
    enclosed = np.asarray([-0.9, -0.85, -0.7, 0.1])
    assert outermost_threshold_radius(radii, enclosed, -0.8) == 4.0
    assert first_compensation_radius(radii, enclosed) == 8.0


def test_local_minimum_wraps_across_box_face() -> None:
    density = np.ones((8, 8, 8))
    density[7, 0, 0] = -2.0
    centre, value = find_local_minimum(
        density, guess=(0.01, 0.01, 0.01), search_radius=2.0, box_size=8.0
    )
    assert centre == (7.5 / 8.0, 0.5 / 8.0, 0.5 / 8.0)
    assert value == -2.0


def test_particle_radius_uses_minimum_periodic_image() -> None:
    positions = np.asarray([[0.99, 0.5, 0.5], [0.25, 0.5, 0.5]])
    radii = periodic_radii(positions, centre=(0.01, 0.5, 0.5), box_size=100.0)
    np.testing.assert_allclose(radii, [2.0, 24.0])
