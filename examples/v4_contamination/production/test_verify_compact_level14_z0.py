from __future__ import annotations

from pathlib import Path
import re
import struct

import numpy as np
import pytest

from verify_compact_level14_z0 import check_id_conservation, check_outputs


HERE = Path(__file__).resolve().parent


def record(payload: bytes) -> bytes:
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def write_part(path: Path, ncpu: int, ids: list[int]) -> None:
    count = len(ids)
    chunks = [
        struct.pack("<i", ncpu),
        struct.pack("<i", 3),
        struct.pack("<i", count),
        struct.pack("<4i", 1, 2, 3, 4),
        struct.pack("<q", 0),
        struct.pack("<d", 0.0),
        struct.pack("<d", 0.0),
        struct.pack("<i", 0),
    ]
    chunks.extend(np.zeros(count, dtype="<f8").tobytes() for _ in range(7))
    chunks.append(np.asarray(ids, dtype="<i8").tobytes())
    path.write_bytes(b"".join(record(chunk) for chunk in chunks))


def write_output(run_dir: Path, index: int, aexp: float, ranks: int = 2) -> Path:
    suffix = f"{index:05d}"
    output = run_dir / f"output_{suffix}"
    output.mkdir()
    (output / "COMPLETE").touch()
    (output / f"info_{suffix}.txt").write_text(
        f"ncpu = {ranks}\nlevelmin = 9\nlevelmax = 14\naexp = {aexp:.16E}\n"
    )
    for rank in range(1, ranks + 1):
        (output / f"part_{suffix}.out{rank:05d}").touch()
    return output


def test_z0_namelist_restarts_verified_hierarchy() -> None:
    namelist = (HERE / "ramses_compact726_level14_z0.nml").read_text()
    assert re.search(r"^nrestart=1$", namelist, re.MULTILINE)
    assert re.search(r"^noutput=5$", namelist, re.MULTILINE)
    assert re.search(
        r"^aout=0\.10,0\.25,0\.50,0\.75,1\.00$", namelist, re.MULTILINE
    )
    assert re.search(r"^levelmin=9$", namelist, re.MULTILINE)
    assert re.search(r"^levelmax=14$", namelist, re.MULTILINE)
    assert re.search(r"^m_refine=15\*8\.0$", namelist, re.MULTILINE)
    assert "void_refine=.true." not in namelist


def test_accepts_initial_state_and_five_scheduled_outputs(tmp_path: Path) -> None:
    for index, aexp in enumerate((0.02, 0.10, 0.25, 0.50, 0.75, 1.00), start=1):
        write_output(tmp_path, index, aexp)

    initial, final = check_outputs(tmp_path, expected_ranks=2)

    assert initial.name == "output_00001"
    assert final.name == "output_00006"


def test_rejects_snapshot_without_complete_marker(tmp_path: Path) -> None:
    for index, aexp in enumerate((0.02, 0.10, 0.25, 0.50, 0.75, 1.00), start=1):
        output = write_output(tmp_path, index, aexp)
    (output / "COMPLETE").unlink()

    with pytest.raises(AssertionError, match="COMPLETE marker"):
        check_outputs(tmp_path, expected_ranks=2)


def test_exact_initial_final_id_set_equality(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    final = tmp_path / "final"
    initial.mkdir()
    final.mkdir()
    write_part(initial / "part_00001.out00001", 2, [0, 3, 8])
    write_part(initial / "part_00001.out00002", 2, [2, 5, 7])
    write_part(final / "part_00006.out00001", 2, [7, 0, 5])
    write_part(final / "part_00006.out00002", 2, [8, 2, 3])

    check_id_conservation(initial, final, 2, id_capacity=9, expected_particles=6)


def test_rejects_final_id_absent_from_handoff(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    final = tmp_path / "final"
    initial.mkdir()
    final.mkdir()
    write_part(initial / "part_00001.out00001", 2, [0, 1])
    write_part(initial / "part_00001.out00002", 2, [2, 3])
    write_part(final / "part_00006.out00001", 2, [0, 1])
    write_part(final / "part_00006.out00002", 2, [2, 4])

    with pytest.raises(AssertionError, match="was absent initially"):
        check_id_conservation(initial, final, 2, id_capacity=5, expected_particles=4)
