"""Real-executable regression: lagCAMB appends two columns to CAMB's 13.

Run: python -m unittest discover -s tests -p test_lagcamb_columns.py
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LagCAMBColumnsTest(unittest.TestCase):
    def test_appended_columns_preserve_standard_matter_spectrum(self):
        executable = ROOT / "genetIC"
        if not executable.is_file():
            self.skipTest("Build genetIC first")
        source = (ROOT / "tests/camb_transfer_kmax40_z0_post2015.dat").read_text()
        outputs = []
        with tempfile.TemporaryDirectory() as temporary:
            for columns in (13, 15, 14):
                folder = Path(temporary) / str(columns)
                folder.mkdir()
                (folder / "output").mkdir()
                table = "# standard CAMB with optional trailing dark sector columns\n"
                table += "\n".join(line + " 0.0"*(columns-13) for line in source.splitlines())+"\n"
                (folder / "transfer.dat").write_text(table)
                (folder / "params.txt").write_text(
                    "Om .3\nOb .05\nOl .7\nhubble .7\nns .96\nA_s 2.1e-9\nk_p .05\nzin 0\n"
                    "camb transfer.dat\nrandom_seed 1234\noutname test\noutdir output\n"
                    "outformat grafic\nbase_grid 100 8\ndone\ndump_ps 0\n"
                )
                run = subprocess.run([str(executable), "params.txt"], cwd=folder,
                                     env=dict(os.environ, OMP_NUM_THREADS="1"), capture_output=True, text=True)
                if columns == 14:
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("sensible number of columns", run.stdout+run.stderr)
                else:
                    self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                    outputs.append(next((folder / "output").glob("*.ps")).read_bytes())
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
