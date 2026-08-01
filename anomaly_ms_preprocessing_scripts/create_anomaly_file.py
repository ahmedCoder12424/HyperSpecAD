"""
create_anomalies.py
===================
Read spectra from a SOURCE MGF file, create anomalies by optionally
remapping PEPMASS (precursor mass) into the m/z range of the
clean/inorganic target dataset, and write one output MGF per
anomaly-count step.

Peak m/z and intensity values are always left untouched — the anomaly
signal comes from the (real, proteomics-derived) peak pattern, not from
randomizing the peaks themselves.

If ADJUST_PEPMASS = True  → PEPMASS is redrawn uniformly in [MZ_MIN, MZ_MAX]
If ADJUST_PEPMASS = False → PEPMASS is left as the original source value

TOTAL_N is derived automatically by counting spectra in CLEAN_DATA_DIR.

Configure everything in the CONFIG block at the bottom.
"""

import random
from pathlib import Path

import numpy as np
from pyteomics import mgf


# ──────────────────────────────────────────────────────────────
# 0. Count total spectra available in a directory of clean MGF files
# ──────────────────────────────────────────────────────────────
def count_total_spectra(source: str) -> int:
    """
    Count how many spectra (BEGIN IONS blocks) exist across all .mgf
    files in `source` (a directory) or in a single .mgf file.

    This does a lightweight line-scan for 'BEGIN IONS' rather than fully
    parsing every spectrum, so it stays fast even on very large files.
    """
    source_path = Path(source)

    if source_path.is_dir():
        mgf_files = sorted(source_path.glob("*.mgf"))
        if not mgf_files:
            raise FileNotFoundError(f"No .mgf files found in directory: {source_path}")
    elif source_path.is_file():
        mgf_files = [source_path]
    else:
        raise FileNotFoundError(f"Source not found: {source_path}")

    total = 0
    print(f"[count] Scanning {len(mgf_files)} file(s) in: {source_path}")
    for mgf_file in mgf_files:
        count = 0
        with open(mgf_file, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("BEGIN IONS"):
                    count += 1
        print(f"       {mgf_file.name}: {count} spectra")
        total += count

    print(f"       Total spectra found: {total}")
    return total


# ──────────────────────────────────────────────────────────────
# 1. Load spectra from a file or every MGF in a directory
# ──────────────────────────────────────────────────────────────
def load_spectra(source: str, n: int, seed: int) -> list:
    source_path = Path(source)

    if source_path.is_dir():
        mgf_files = sorted(source_path.glob("*.mgf"))
        if not mgf_files:
            raise FileNotFoundError(f"No .mgf files found in directory: {source_path}")
        print(f"[load] Directory mode — found {len(mgf_files)} MGF file(s) in: {source_path}")
    elif source_path.is_file():
        mgf_files = [source_path]
        print(f"[load] Single-file mode: {source_path}")
    else:
        raise FileNotFoundError(f"Source not found: {source_path}")

    # Load all spectra from all files first
    all_spectra = []
    for mgf_file in mgf_files:
        print(f"       Reading {mgf_file.name} ...", end=" ")
        before = len(all_spectra)
        with mgf.read(str(mgf_file)) as reader:
            for spectrum in reader:
                all_spectra.append(spectrum)
        print(f"+{len(all_spectra) - before} spectra")

    print(f"       Total loaded: {len(all_spectra)} spectra across {len(mgf_files)} file(s).")

    if not all_spectra:
        raise ValueError("No spectra found in source.")

    # Sample randomly across the full pool for variety
    k = min(n, len(all_spectra))
    random.seed(seed)
    selected = random.sample(all_spectra, k)
    print(f"       Sampled {k} spectra for anomaly creation.")
    return selected


# ──────────────────────────────────────────────────────────────
# 2. Optionally remap PEPMASS into [mz_min, mz_max]. Peaks are always
#    left untouched.
# ──────────────────────────────────────────────────────────────
def create_anomaly(spectrum: dict, mz_min: float, mz_max: float,
                   rng: random.Random, adjust_pepmass: bool) -> dict:
    """
    Return a new spectrum, tagged as an anomaly. Peak m/z and intensity
    arrays are always left exactly as in the source spectrum.

    If adjust_pepmass is True, PEPMASS is redrawn uniformly from
    [mz_min, mz_max] so it lands in the same precursor-mass window as
    the clean/inorganic target dataset (the original value is preserved
    in anomaly_orig_pepmass for traceability).

    If adjust_pepmass is False, PEPMASS is left exactly as in the
    source spectrum.
    """
    sp = dict(spectrum)
    sp["params"] = dict(spectrum.get("params", {}))

    mz_array = spectrum["m/z array"]
    n_peaks  = len(mz_array)

    if n_peaks == 0:
        return sp

    if adjust_pepmass:
        new_pepmass  = mz_min + rng.random() * (mz_max - mz_min)
        orig_pepmass = sp["params"].get("pepmass")

        # preserve pepmass's original type (pyteomics often stores it as a
        # (mass, intensity) tuple)
        sp["params"]["pepmass"] = (new_pepmass,) if isinstance(orig_pepmass, tuple) else new_pepmass

        if orig_pepmass is not None:
            orig_val = orig_pepmass[0] if isinstance(orig_pepmass, tuple) else orig_pepmass
            sp["params"]["anomaly_orig_pepmass"] = f"{orig_val:.6f}"

        sp["params"]["anomaly_pepmass_range"] = f"{mz_min:.4f}-{mz_max:.4f}"
    # else: leave sp["params"]["pepmass"] exactly as copied from source

    # Tag for traceability
    sp["params"]["anomaly"] = "true"
    sp["params"]["anomaly_pepmass_adjusted"] = "true" if adjust_pepmass else "false"

    return sp


# ──────────────────────────────────────────────────────────────
# 3. Build anomaly list and write to file
# ──────────────────────────────────────────────────────────────
def build_and_write_anomalies(spectra: list, output_dir: Path,
                               mz_min: float, mz_max: float,
                               seed: int, pct: float,
                               adjust_pepmass: bool) -> Path:
    rng = random.Random(seed)
    anomalies = [create_anomaly(sp, mz_min, mz_max, rng, adjust_pepmass=adjust_pepmass)
                 for sp in spectra]

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_pmadj" if adjust_pepmass else "_pmorig"
    out_path = output_dir / f"anomalies_n{len(anomalies)}_pct{pct}{tag}.mgf"

    mgf.write(anomalies, str(out_path))
    print(f"  → Written {len(anomalies):>5} anomalous spectra  →  {out_path.name}")
    return out_path


# ──────────────────────────────────────────────────────────────
# CONFIG — edit everything here
# ──────────────────────────────────────────────────────────────
SOURCE          = "/hdd/data/fahmed/pigments_mgf"  # path to a .mgf file OR a directory of .mgf files
CLEAN_DATA_DIR  = "/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc"  # used to derive TOTAL_N
OUTPUT_DIR      = Path("/hdd/data/fahmed/battery_mgf_files/anomalies_proteomics_Gr_HC_Si_trunc_correct_percent")

MZ_MIN      = 1.0       # float lower bound for PEPMASS remap (match clean/inorganic data's range)
MZ_MAX      = 215.0     # float upper bound for PEPMASS remap

ADJUST_PEPMASS = feature_values   # True  → PEPMASS redrawn uniformly in [MZ_MIN, MZ_MAX]
                        # False → PEPMASS left as the original source value

SEED        = 42

# Loop: generate one anomaly file per percentage step
PERCENTAGES = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]   # % of TOTAL_N — edit as needed


# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────
TOTAL_N = count_total_spectra(CLEAN_DATA_DIR)

print(f"Source         : {SOURCE}")
print(f"Adjust PEPMASS : {ADJUST_PEPMASS}")
print(f"PEPMASS range  : [{MZ_MIN}, {MZ_MAX}]  (peaks always left unchanged)")
print(f"Pool N         : {TOTAL_N}  (derived from {CLEAN_DATA_DIR})")
print(f"Steps          : {PERCENTAGES}%\n")

for pct in PERCENTAGES:
    n = max(1, int(round(TOTAL_N * pct / 100)))
    print(f"[{pct:>3}%]  n = {n}")

    spectra = load_spectra(SOURCE, n=n, seed=SEED)

    build_and_write_anomalies(
        spectra=spectra,
        output_dir=OUTPUT_DIR,
        mz_min=MZ_MIN,
        mz_max=MZ_MAX,
        seed=SEED,
        pct=pct,
        adjust_pepmass=ADJUST_PEPMASS,
    )
    print()

print("Done.")