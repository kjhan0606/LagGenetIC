from __future__ import annotations

from pathlib import Path
import re


HERE = Path(__file__).resolve().parent


def test_compact_level14_geometry_uses_a_32_mpc_patch() -> None:
    path = HERE / "genetic_compact729_level14_inverted.txt"
    lines = [
        line.split()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    base = next(tokens for tokens in lines if tokens[0] == "base_grid")
    zooms = [tokens for tokens in lines if tokens[0] == "zoom_grid"]
    assert zooms == [
        ["zoom_grid", "8", "128"],
        ["zoom_grid", "2", "128"],
        ["zoom_grid", "1", "256"],
        ["zoom_grid", "1", "512"],
        ["zoom_grid", "1", "1024"],
    ]
    box_size = float(base[1])
    physical_size = box_size
    cells = int(base[2]) ** 3
    effective_sizes = []
    for zoom in zooms:
        physical_size /= int(zoom[1])
        local_cells = int(zoom[2])
        cells += local_cells**3
        effective_sizes.append(round(box_size / (physical_size / local_cells)))
    assert 512.0 / int(zooms[0][1]) == 64.0
    assert physical_size == 32.0
    assert effective_sizes == [1024, 2048, 4096, 8192, 16384]
    assert cells == 1_363_148_800


def test_compact_level14_ramses_capacity_and_inputs() -> None:
    namelist = (HERE / "ramses_compact729_level14_pilot.nml").read_text()
    initfiles = re.findall(r"^initfile\(\d+\)='([^']+)'$", namelist, re.MULTILINE)
    assert len(initfiles) == 6
    assert initfiles[-1].endswith("grafic_16384")
    levelmax = int(re.search(r"^levelmax=(\d+)$", namelist, re.MULTILINE).group(1))
    ngridmax = int(re.search(r"^ngridmax=(\d+)$", namelist, re.MULTILINE).group(1))
    npartmax = int(re.search(r"^npartmax=(\d+)$", namelist, re.MULTILINE).group(1))
    m_refine = re.search(r"^m_refine=(\d+)\*([0-9.]+)$", namelist, re.MULTILINE)
    assert levelmax == 14
    assert ngridmax == 1_200_000
    assert npartmax == 6_000_000
    assert int(m_refine.group(1)) >= levelmax
    assert float(m_refine.group(2)) == 8.0


def test_compact_level14_transfer_extends_beyond_the_grid_requirement() -> None:
    ini = (HERE / "lagcamb_z49_level14.ini").read_text()
    transfer_kmax = float(
        re.search(r"^transfer_kmax\s*=\s*([0-9.]+)$", ini, re.MULTILINE).group(1)
    )
    assert transfer_kmax > 201.06192983
