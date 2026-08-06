from __future__ import annotations

import numpy as np

from select_parent_targets import (
    Candidate,
    attach_geometry,
    choose_targets,
    periodic_axis_geometry,
)


def candidate(halo_id: int, count: int, centre: float) -> Candidate:
    item = Candidate(
        halo_id=halo_id,
        particle_count=count,
        hop_mass=float(count),
        contamination=0.0,
        eulerian_centre=(centre, centre, centre),
        boundary_safe=True,
        lagrangian_width=(0.2, 0.2, 0.2),
    )
    item.mass_rank = halo_id
    return item


def test_periodic_axis_geometry_detects_wrapping_mask() -> None:
    start, width, centre, wraps = periodic_axis_geometry(
        np.asarray([510, 511, 0, 1]), 512
    )
    assert start == 510 / 512
    assert width == 4 / 512
    assert centre == 0.0
    assert wraps


def test_wrapping_mask_is_safe_after_recentering() -> None:
    item = candidate(1, 4, 0.5)
    ix = np.asarray([510, 511, 0, 1], dtype=np.int64)
    iy = np.full(ix.shape, 100, dtype=np.int64)
    iz = np.full(ix.shape, 200, dtype=np.int64)
    ids = ix * 512 * 512 + iy * 512 + iz
    attach_geometry(item, ids, 512, edge_buffer_cells=2)
    assert item.wraps_boundary == (True, False, False)
    assert item.boundary_safe
    assert item.recenter_shift is not None


def test_target_choice_applies_separation_cut() -> None:
    candidates = [
        candidate(1, 9000, 0.10),
        candidate(2, 8000, 0.12),
        candidate(3, 7000, 0.40),
        candidate(4, 6000, 0.70),
    ]
    selected = choose_targets(candidates, 3, 3000, 0.5, 0.1)
    assert [item.halo_id for item in selected] == [1, 3, 4]
