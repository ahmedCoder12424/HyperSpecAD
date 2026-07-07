

# # read the file 
# # read the proteomomics 
# # inject spectra, wrtie the file 

# from pyteomics import mgf



# anomaly_spectras = []
# with mgf.read('../hdd/data/sumukh/raw-ms-dataset/PXD001468/b1942_293T_proteinID_06B_QE3_122212.mgf') as reader:
#     for spectrum in reader:
#         # print(spectrum['params']['title'])          # Spectrum title
#         # print(spectrum['params']['pepmass'])        # Precursor mass
#         # print(spectrum['m/z array'])                # m/z values (numpy array)
#         # print(spectrum['intensity array'])   
#         anomaly_spectras.append(spectrum)
#         if (len(anomaly_spectras)==4):
#             break
# print(len(anomaly_spectras))


"""
inject_spectra.py
=================
Read spectra from a SOURCE MGF file (Group 2 / anomalies),
then inject a controlled number of them into every MGF file
inside a TARGET FOLDER (Group 1), writing new files named
  <original_stem>_with_anomaly.mgf

Usage
-----
python inject_spectra.py \
    --source  /path/to/anomaly.mgf \
    --target  /path/to/group1_folder/ \
    --output  /path/to/output_folder/ \
    --n       4                          # how many spectra to inject (int)
    --pct     10                         # OR inject 10 % of source spectra
    --mode    append                     # append | prepend | random
    --seed    42                         # random seed (for reproducibility)

Only one of --n or --pct is required; if both are given --n wins.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional, List

from pyteomics import mgf


# ─────────────────────────────────────────────
# 1. Load spectra from source (anomaly) file
# ─────────────────────────────────────────────
def load_source_spectra(source_path: str, n: Optional[int], pct: Optional[float], seed: int) -> list:
    """Return a list of spectrum dicts to inject."""
    print(f"[1/3] Reading source spectra from: {source_path}")
    spectra = []
    with mgf.read(source_path) as reader:
        for spectrum in reader:
            spectra.append(spectrum)

    print(f"      Found {len(spectra)} spectra in source file.")

    if not spectra:
        raise ValueError("Source MGF file contains no spectra.")

    # Determine how many to inject
    if n is not None:
        k = min(n, len(spectra))
    elif pct is not None:
        k = max(1, int(round(len(spectra) * pct / 100)))
    else:
        raise ValueError("Provide either --n or --pct.")

    # Sample without replacement (or take first k)
    random.seed(seed)
    selected = random.sample(spectra, k) if k < len(spectra) else spectra
    print(f"      Selected {k} spectra for injection.")
    return selected


# ─────────────────────────────────────────────
# 2. Inject into a single target MGF file
# ─────────────────────────────────────────────
def inject_into_file(target_path: Path, injection_spectra: list,
                     output_dir: Path, mode: str, seed: int) -> Path:
    """Read target, merge with injection spectra, write new file."""
    with mgf.read(str(target_path)) as reader:
        target_spectra = list(reader)

    n_target = len(target_spectra)
    n_inject = len(injection_spectra)

    # # Tag injected spectra so they're traceable
    # tagged = []
    # for i, sp in enumerate(injection_spectra):
    #     sp = dict(sp)                        # shallow copy
    #     params = dict(sp.get("params", {}))
    #     orig_title = params.get("title", f"spectrum_{i}")
    #     params["title"] = f"{orig_title}__INJECTED_{i}"
    #     params["injected"] = "true"
    #     params['scans'] = params['scans'] + 'injected'
    #     sp["params"] = params
    #     tagged.append(sp)

    # # Build merged list
    # if mode == "append":
    #     merged = target_spectra + tagged
    # elif mode == "prepend":
    #     merged = tagged + target_spectra
    # elif mode == "random":
    #     merged = target_spectra[:]
    #     positions = sorted(random.sample(range(len(merged) + n_inject), n_inject))
    #     for pos, sp in zip(positions, tagged):
    #         merged.insert(pos, sp)
    # else:
    #     raise ValueError(f"Unknown mode: {mode!r}. Choose append | prepend | random.")

    # Output path
    stem = target_path.stem
    out_path = output_dir / f"{stem}_with_anomaly.mgf"

    # mgf.write(merged, str(out_path))
    # print(f"      {target_path.name}  ({n_target} + {n_inject} injected)  →  {out_path.name}")

    anomaly_only_path = output_dir / f"anomalies_only{n_target}.mgf"
    mgf.write(injection_spectra, str(anomaly_only_path))
    print(f"      Anomaly-only file written → {anomaly_only_path.name}")
    return out_path


# ─────────────────────────────────────────────
# 3. Process entire target folder
# ─────────────────────────────────────────────
def process_folder(target_folder: str, injection_spectra: list,
                   output_folder: str, mode: str, seed: int) -> List[Path]:
    target_dir = Path(target_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    mgf_files = sorted(target_dir.glob("*.mgf"))
    if not mgf_files:
        print(f"  ⚠  No .mgf files found in {target_dir}")
        return []

    print(f"\n[2/3] Processing {len(mgf_files)} target MGF file(s) in: {target_dir}")
    outputs = []
    for f in mgf_files:
        out = inject_into_file(f, injection_spectra, output_dir, mode, seed)
        outputs.append(out)

    return outputs


# ─────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Inject anomaly spectra into Group-1 MGF files.")
    p.add_argument("--source",  required=True,  help="Path to source (anomaly) MGF file")
    p.add_argument("--target",  required=True,  help="Folder containing Group-1 MGF files")
    p.add_argument("--output",  required=True,  help="Output folder for new MGF files")
    p.add_argument("--n",       type=int,   default=None, help="Number of spectra to inject")
    p.add_argument("--pct",     type=float, default=None, help="Percentage of source spectra to inject (e.g. 10)")
    p.add_argument("--mode",    default="append", choices=["append", "prepend", "random"],
                   help="Where to insert spectra: append (default) | prepend | random")
    p.add_argument("--seed",    type=int,   default=42, help="Random seed (default 42)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.n is None and args.pct is None:
        print("Error: provide --n (count) or --pct (percentage).", file=sys.stderr)
        sys.exit(1)

    injection_spectra = load_source_spectra(
        args.source, n=args.n, pct=args.pct, seed=args.seed
    )

    outputs = process_folder(
        target_folder=args.target,
        injection_spectra=injection_spectra,
        output_folder=args.output,
        mode=args.mode,
        seed=args.seed,
    )

    print(f"\n[3/3] Done. {len(outputs)} file(s) written to: {args.output}")
    for p in outputs:
        print(f"      ✓ {p}")


if __name__ == "__main__":
    main()


#python inject_anomalies.py --source '../hdd/data/sumukh/raw-ms-dataset/PXD001468/b1941_293T_proteinID_05B_QE3_122212.mgf' --target '../hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_split_mgf' --output '../hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_split_mgf_anomaly_10' --n 10 --mode 'random'