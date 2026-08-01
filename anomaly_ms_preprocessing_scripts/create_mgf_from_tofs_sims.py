"""
imzml_to_mgf.py
===============
Convert ImzML/IBD imaging mass spectrometry files to MGF format,
then split the output into chunks of at most MAX_MB megabytes each.

PEPMASS is assigned from pixel (x, y) spatial position mapped to a
synthetic m/z range so that HyperSpec buckets distribute evenly
instead of collapsing into dominant-fragment buckets.

Configure everything in the CONFIG block at the bottom.
"""

import os
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser


# ──────────────────────────────────────────────────────────────
# CONFIG — edit everything here
# ──────────────────────────────────────────────────────────────
BASE_DIR   = "../hdd/data/fahmed/battery_mzml_ibd/"
OUTPUT_DIR = "../hdd/data/fahmed/battery_mgf_files/"

IMZML_FILES = [
    "Pre_30_Ref_DE_250_2040_5shots_1frame_FIBlong_FIBpolish_1.ImzML",
    "Cycled_graphite_9014_DE_2048_250_after SE cleaning_1.ImzML",
    "Gr_HC_Si_one layer_negative_DE_2048pixels_rotated_after FIB_2.ImzML",
]

TITLES = [
    "Pre_30_Ref_DE_250_2040_5shots_1frame_FIBlong_FIBpolish_1",
    "Cycled_graphite_9014_DE_2048_250_after SE cleaning_1",
    "Gr_HC_Si_one layer_negative_DE_2048pixels_rotated_after FIB_2",
]

# Spatial → PEPMASS mapping range (should cover your actual m/z range)
MZ_MIN = 50.0
MZ_MAX = 500.0

# Split output files at this size (set to None to write one big file)
MAX_MB = 100

LOG_INTERVAL = 10_000   # print progress every N spectra


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def spatial_pepmass(x, y, min_x, min_y, max_x, max_y):
    """
    Map pixel (x, y) to a synthetic PEPMASS in [MZ_MIN, MZ_MAX].
    Interleaves x and y so adjacent rows land in different buckets.
    """
    nx = (x - min_x) / max(max_x - min_x, 1)
    ny = (y - min_y) / max(max_y - min_y, 1)
    linear = (ny + nx / max(max_x - min_x + 1, 1)) % 1.0
    return MZ_MIN + linear * (MZ_MAX - MZ_MIN)


class SplitWriter:
    """
    Writes MGF blocks to a series of split files capped at max_bytes each.
    If max_bytes is None, everything goes into a single file.
    """
    def __init__(self, out_dir: str, max_bytes):
        self.out_dir       = out_dir
        self.max_bytes     = max_bytes
        self.part          = 0
        self.current_bytes = 0
        os.makedirs(out_dir, exist_ok=True)
        self._fh = open(self._path(), "w")

    def _path(self):
        return os.path.join(self.out_dir, f"split_{self.part:04d}.mgf")

    def write_block(self, block: str) -> None:
        block_bytes = len(block.encode("utf-8"))
        if (
            self.max_bytes is not None
            and self.current_bytes > 0
            and self.current_bytes + block_bytes > self.max_bytes
        ):
            self._fh.close()
            self.part += 1
            self.current_bytes = 0
            self._fh = open(self._path(), "w")
        self._fh.write(block)
        self.current_bytes += block_bytes

    def close(self) -> int:
        self._fh.close()
        return self.part + 1   # number of files written


# ──────────────────────────────────────────────────────────────
# Main converter
# ──────────────────────────────────────────────────────────────
def convert(imzml_filename: str, title: str) -> None:
    imzml_path = os.path.join(BASE_DIR, imzml_filename)

    # Each ImzML file gets its own split subdirectory
    stem      = imzml_filename.replace(".ImzML", "").replace(".imzml", "")
    out_dir   = os.path.join(OUTPUT_DIR, f"{stem}_mgf_split")
    max_bytes = MAX_MB * 1024 * 1024 if MAX_MB is not None else None

    print(f"\n[convert] {imzml_filename}")
    p = ImzMLParser(imzml_path)
    n_spectra = len(p.coordinates)
    print(f"          {n_spectra} spectra  →  {out_dir}")

    all_x = [c[0] for c in p.coordinates]
    all_y = [c[1] for c in p.coordinates]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    writer  = SplitWriter(out_dir, max_bytes)
    written = skipped = 0

    for i, (x, y, z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(i)
        mzs         = np.asarray(mzs,         dtype=np.float64)
        intensities = np.asarray(intensities,  dtype=np.float64)

        if len(mzs) == 0:
            skipped += 1
            continue

        pepmass     = spatial_pepmass(x, y, min_x, min_y, max_x, max_y)
        pepmass_idx = int(np.argmin(np.abs(mzs - pepmass)))

        lines = [
            "BEGIN IONS\n",
            f"TITLE=spectrum_{i}_x{x}_y{y}_z{z}_{title}\n",
            f"SCANS={i + 1}\n",
            "RTINSECONDS=0\n",
            f"PEPMASS={pepmass:.6f} {intensities[pepmass_idx]:.6f}\n",
            "CHARGE=1+\n",
        ]
        for mz, inten in zip(mzs, intensities):
            lines.append(f"{mz:.6f} {inten:.6f}\n")
        lines.append("END IONS\n\n")

        writer.write_block("".join(lines))
        written += 1

        if i % LOG_INTERVAL == 0:
            print(f"          Progress: {i}/{n_spectra} ({100*i/n_spectra:.1f}%)")

    n_files = writer.close()
    print(f"          Done — {written} spectra written, {skipped} empty skipped")
    print(f"          Split into {n_files} file(s) in: {out_dir}")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(IMZML_FILES) != len(TITLES):
        raise ValueError(
            f"IMZML_FILES ({len(IMZML_FILES)}) and TITLES ({len(TITLES)}) must have the same length."
        )

    for imzml_file, title in zip(IMZML_FILES, TITLES):
        convert(imzml_file, title)

    print("\nAll files converted.")