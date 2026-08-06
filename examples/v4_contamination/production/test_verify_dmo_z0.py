from __future__ import annotations

from pathlib import Path

import pytest

from verify_dmo_z0 import check_outputs


def write_output(run_dir: Path, index: int, aexp: float, ranks: int = 64) -> None:
    suffix = f"{index:05d}"
    output = run_dir / f"output_{suffix}"
    output.mkdir()
    (output / f"info_{suffix}.txt").write_text(
        f"ncpu = {ranks}\naexp = {aexp:.16E}\n"
    )


def test_accepts_initial_state_and_five_scheduled_outputs(tmp_path: Path) -> None:
    for index, aexp in enumerate((0.02, 0.10, 0.25, 0.50, 0.75, 1.00), start=1):
        write_output(tmp_path, index, aexp)

    final_output = check_outputs(tmp_path, expected_ranks=64)

    assert final_output.name == "output_00006"


def test_rejects_missing_initial_state(tmp_path: Path) -> None:
    for index, aexp in enumerate((0.10, 0.25, 0.50, 0.75, 0.90, 1.00), start=1):
        write_output(tmp_path, index, aexp)

    with pytest.raises(AssertionError, match="initial snapshot scale factor"):
        check_outputs(tmp_path, expected_ranks=64)
