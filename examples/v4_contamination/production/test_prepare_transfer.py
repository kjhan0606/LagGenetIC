#!/usr/bin/env python3
"""Regression tests for the extended lagCAMB table adapter."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from prepare_transfer import check_k_coverage, convert


class PrepareTransferTests(unittest.TestCase):
    def test_extracts_standard_columns_without_changing_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.dat"
            destination = root / "standard.dat"
            header = "# " + " ".join(f"c{i}" for i in range(15)) + "\n"
            row1 = [f"{i}.125E+01" for i in range(15)]
            row2 = [f"-{i}.250E-02" for i in range(15)]
            source.write_text(
                header + " ".join(row1) + "\n" + " ".join(row2) + "\n"
            )

            rows, columns = convert(source, destination)

            self.assertEqual((rows, columns), (2, 15))
            output = destination.read_text().splitlines()
            self.assertEqual(output[0].split()[1:], [f"c{i}" for i in range(13)])
            self.assertEqual(output[1].split(), row1[:13])
            self.assertEqual(output[2].split(), row2[:13])

    def test_rejects_short_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.dat"
            destination = root / "standard.dat"
            source.write_text(" ".join("1" for _ in range(12)) + "\n")
            with self.assertRaisesRegex(ValueError, "need at least 13"):
                convert(source, destination)
            self.assertFalse(destination.exists())

    def test_checks_the_requested_wave_number_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "transfer.dat"
            rows = [" ".join([str(k)] + ["1"] * 12) for k in (0.1, 10.0, 250.0)]
            path.write_text("\n".join(rows) + "\n")
            self.assertEqual(check_k_coverage(path, 220.0), (0.1, 250.0))
            with self.assertRaisesRegex(ValueError, "does not reach"):
                check_k_coverage(path, 300.0)


if __name__ == "__main__":
    unittest.main()
