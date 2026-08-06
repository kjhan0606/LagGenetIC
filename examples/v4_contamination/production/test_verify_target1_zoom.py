from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from verify_target1_zoom import (
    check_ic_pair,
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
    for effective_size, shape in ((2, (2, 2, 2)), (4, (1, 2, 2))):
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
    values, target_mean = check_ic_pair(normal, inverted, 1, base_size)
    assert values == 12
    assert target_mean == 1.0


def test_read_part_ids_finds_64_bit_identity_record(tmp_path: Path) -> None:
    path = tmp_path / "part.out"
    ids = np.asarray([5, 9, 17], dtype="<i8")
    records = [b"header", struct.pack("<i", 3), struct.pack("<i", ids.size)]
    records.extend([b""] * 12)
    records.append(ids.tobytes())
    write_records(path, records)
    np.testing.assert_array_equal(read_part_ids(path), ids)


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
