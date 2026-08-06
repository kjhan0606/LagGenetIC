from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from verify_inverted_parent_ic import check_density_pair


def write_grafic(path: Path, cube: np.ndarray) -> None:
    n3, n2, n1 = cube.shape
    header = struct.pack("<iii", n1, n2, n3) + bytes(32)
    with path.open("wb") as handle:
        handle.write(struct.pack("<i", 44))
        handle.write(header)
        handle.write(struct.pack("<i", 44))
        for slab in cube:
            payload = np.asarray(slab, dtype="<f4").tobytes()
            handle.write(struct.pack("<i", len(payload)))
            handle.write(payload)
            handle.write(struct.pack("<i", len(payload)))


def test_exact_sign_pair(tmp_path: Path) -> None:
    normal = np.arange(64, dtype=np.float32).reshape(4, 4, 4) - 20.0
    write_grafic(tmp_path / "normal", normal)
    write_grafic(tmp_path / "inverted", -normal)
    check_density_pair(tmp_path / "normal", tmp_path / "inverted", 4)
