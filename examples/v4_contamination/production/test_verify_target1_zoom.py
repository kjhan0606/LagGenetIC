from __future__ import annotations

from pathlib import Path
import re
import struct

import numpy as np
import pytest

from verify_target1_zoom import (
    check_ic_pair,
    check_ramses,
    check_sign_file,
    fortran_record_stream,
    read_part_ids,
)


HERE = Path(__file__).resolve().parent


def write_records(path: Path, records: list[bytes]) -> None:
    with path.open("wb") as handle:
        for payload in records:
            marker = struct.pack("<i", len(payload))
            handle.write(marker)
            handle.write(payload)
            handle.write(marker)


def write_cube(path: Path, values: np.ndarray) -> None:
    nz, ny, nx = values.shape
    records = [struct.pack("<iii", nx, ny, nz)]
    records.extend(np.asarray(plane, dtype="<f4").tobytes() for plane in values)
    write_records(path, records)


def test_fortran_record_stream_rejects_bad_trailer(tmp_path: Path) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(struct.pack("<i", 4) + b"data" + struct.pack("<i", 3))
    with pytest.raises(ValueError, match="record marker mismatch"):
        list(fortran_record_stream(path))


def test_sign_file_requires_exact_negation(tmp_path: Path) -> None:
    normal = tmp_path / "normal"
    inverted = tmp_path / "inverted"
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 1.0
    write_cube(normal, values)
    write_cube(inverted, -values)
    assert check_sign_file(normal, inverted) == (8, 0.0)
    values[0, 0, 0] += 0.25
    write_cube(inverted, -values)
    with pytest.raises(AssertionError, match="sign-reversal residual"):
        check_sign_file(normal, inverted)


def test_ic_pair_counts_centered_refmap(tmp_path: Path) -> None:
    base_size = 2
    normal = tmp_path / "normal"
    inverted = tmp_path / "inverted"
    hierarchy = ((2, (2, 2, 2)), (4, (1, 2, 2)), (8, (2, 2, 2)))
    for effective_size, shape in hierarchy:
        normal_dir = normal / f"v4_target1_normal.grafic_{effective_size}"
        inverted_dir = inverted / f"v4_target1_inverted.grafic_{effective_size}"
        normal_dir.mkdir(parents=True)
        inverted_dir.mkdir(parents=True)
        density = np.ones(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.float32)
        mask.flat[0] = 1.0
        write_cube(normal_dir / "ic_deltab", density)
        write_cube(inverted_dir / "ic_deltab", -density)
        write_cube(normal_dir / "ic_refmap", mask)
        write_cube(inverted_dir / "ic_refmap", mask)
        write_cube(normal_dir / "ic_particle_ids", mask)
        write_cube(inverted_dir / "ic_particle_ids", mask)
    values, target_mean, effective_sizes, cube_cells = check_ic_pair(
        normal, inverted, 1, base_size
    )
    assert values == 20
    assert target_mean == 1.0
    assert effective_sizes == [2, 4, 8]
    assert cube_cells == [8, 4, 8]


def test_read_part_ids_finds_64_bit_identity_record(tmp_path: Path) -> None:
    path = tmp_path / "part.out"
    ids = np.asarray([5, 9, 17], dtype="<i8")
    records = [b"header", struct.pack("<i", 3), struct.pack("<i", ids.size)]
    records.extend([b""] * 12)
    records.append(ids.tobytes())
    write_records(path, records)
    np.testing.assert_array_equal(read_part_ids(path), ids)


def test_multilevel_ramses_particle_accounting(tmp_path: Path) -> None:
    ramses = tmp_path / "ramses"
    output = ramses / "output_00001"
    output.mkdir(parents=True)
    (ramses / "ramses.log").write_text(
        "Level 2 has 0 grids\nLevel 2 has 1 grids\nLevel 3 has 8 grids\n"
    )
    ids = np.concatenate(
        (np.arange(7, dtype="<i8"), np.arange(8, 72, dtype="<i8"))
    )
    for rank, rank_ids in enumerate(np.array_split(ids, 2), start=1):
        records = [b"header", struct.pack("<i", 3), struct.pack("<i", rank_ids.size)]
        records.extend([b""] * 12)
        records.append(rank_ids.tobytes())
        write_records(output / f"part_00001.out{rank:05d}", records)
    grids, particles = check_ramses(ramses, 1, 2, [8, 64], 2)
    assert grids == {2: 1, 3: 8}
    assert particles == 71


@pytest.mark.parametrize(
    "filename",
    ("genetic_target1_zoom_normal.txt", "genetic_target1_zoom_inverted.txt"),
)
def test_zoom_geometry_is_64_mpc_at_effective_1024(filename: str) -> None:
    lines = [
        line.split()
        for line in (HERE / filename).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    base = next(tokens for tokens in lines if tokens[0] == "base_grid")
    zoom = next(tokens for tokens in lines if tokens[0] == "zoom_grid")
    box_size = float(base[1])
    base_cells = int(base[2])
    subbox_factor = int(zoom[1])
    fine_cells = int(zoom[2])
    fine_box_size = box_size / subbox_factor
    fine_cell_size = fine_box_size / fine_cells
    assert fine_box_size == 64.0
    assert fine_cell_size == 0.5
    assert box_size / fine_cell_size == 2 * base_cells


def test_ingestion_grid_capacity_covers_sixteen_rank_base_mesh() -> None:
    namelist = (HERE / "ramses_target1_zoom_ingest.nml").read_text()
    ngridmax = int(re.search(r"^ngridmax=(\d+)$", namelist, re.MULTILINE).group(1))
    m_refine = re.search(
        r"^m_refine=(\d+)\*([0-9.]+)$", namelist, re.MULTILINE
    )
    base_octs_per_rank = 512**3 / 8 / 16
    assert 0.85 * ngridmax > 1.25 * base_octs_per_rank
    assert int(m_refine.group(1)) >= 10
    assert float(m_refine.group(2)) == 8.0


@pytest.mark.parametrize(
    "filename",
    ("genetic_target1_level11_normal.txt", "genetic_target1_level11_inverted.txt"),
)
def test_level11_zoom_preserves_the_64_mpc_patch(filename: str) -> None:
    lines = [
        line.split()
        for line in (HERE / filename).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    base = next(tokens for tokens in lines if tokens[0] == "base_grid")
    zooms = [tokens for tokens in lines if tokens[0] == "zoom_grid"]
    assert zooms == [["zoom_grid", "8", "128"], ["zoom_grid", "1", "256"]]
    box_size = float(base[1])
    base_cells = int(base[2])
    physical_size = box_size
    effective_sizes = []
    for zoom in zooms:
        physical_size /= int(zoom[1])
        cell_size = physical_size / int(zoom[2])
        effective_sizes.append(round(box_size / cell_size))
    assert physical_size == 64.0
    assert effective_sizes == [2 * base_cells, 4 * base_cells]


def test_level11_ingestion_capacity_and_refinement_threshold() -> None:
    namelist = (HERE / "ramses_target1_level11_ingest.nml").read_text()
    ngridmax = int(re.search(r"^ngridmax=(\d+)$", namelist, re.MULTILINE).group(1))
    npartmax = int(re.search(r"^npartmax=(\d+)$", namelist, re.MULTILINE).group(1))
    levelmax = int(re.search(r"^levelmax=(\d+)$", namelist, re.MULTILINE).group(1))
    m_refine = re.search(
        r"^m_refine=(\d+)\*([0-9.]+)$", namelist, re.MULTILINE
    )
    assert ngridmax == 1_600_000
    assert npartmax == 12_000_000
    assert levelmax == 11
    assert int(m_refine.group(1)) >= levelmax
    assert float(m_refine.group(2)) == 8.0
