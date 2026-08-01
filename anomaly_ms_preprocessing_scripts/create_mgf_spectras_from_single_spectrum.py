"""
create_mgf_spectras_from_single_spectrum.py
===================
This script preprocesses the pigments TOF-SIMS dataset to create "mgf" like files.
The pigments files are in txt files with each containing one single broad spectrum with two columns representing
mz values and intensities respectively. The mgf files are created by sampling spectras from spectrum mz range.
This script can work for any TOF-SIMS data represented in txt files where there is only one broad spectrum
"""
import pandas as pd
import numpy as np

pigment_names = {
    "01931": "Alba_Albula",
    "01932": "Bianco_San_Giovanni",
    "01933": "Eggshell_White",
    "01934": "Dolomite",
    "01935": "Anhydrite_plaster",
    "01936": "Natural_alabaster",
}

polarity_map = {
    "01": "positive",
    "02": "negative",
}

def split_asc_balanced(path, k=2000, min_intensity=1, seed=0):
    rng = np.random.default_rng(seed)

    df = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        names=["Channel", "mz", "Intensity"],
        skiprows=1
    )

    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
    df = df[df["Intensity"] >= min_intensity].copy()

    # m/z windows for balanced sampling
    df["mz_bin"] = pd.cut(df["mz"], bins=np.arange(0, df["mz"].max() + 50, 50))

    splits = [[] for _ in range(k)]

    for _, group in df.groupby("mz_bin", observed=True):
        group = group.sample(frac=1, random_state=int(rng.integers(1e9)))
        chunks = np.array_split(group, k)

        for i, chunk in enumerate(chunks):
            splits[i].append(chunk)

    return [
        pd.concat(parts)
          .sort_values("mz")[["mz", "Intensity"]]
        for parts in splits
    ]


def write_mgf(spectra, title_prefix, out_path):
    with open(out_path, "w") as f:
        for i, spec in enumerate(spectra):
            max_idx =  spec['Intensity'].argmax() if len(spec['Intensity']) > 0 else 0
            intensities = spec['Intensity'].values
            mzs = spec['mz'].values
            f.write("BEGIN IONS\n")
            f.write(f"TITLE={title_prefix}_split{i}\n")
            f.write(f"PEPMASS={mzs[max_idx]} {intensities[max_idx]}\n")
            f.write("CHARGE=1+\n")
            for mz, inten in spec[["mz", "Intensity"]].values:
                f.write(f"{mz:.6f} {inten:.0f}\n")
            f.write("END IONS\n\n")


def write_mgf_full_spectrum(input_path, title_prefix, output_path):

    df = pd.read_csv(
        input_path,
        comment="#",
        sep=r"\s+",
        names=["Channel", "mz", "Intensity"],
        skiprows=1
    )
    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
    max_idx =  df['Intensity'].argmax() if len(df['Intensity']) > 0 else 0
    intensities = df['Intensity'].values
    mzs = df['mz'].values
    with open(output_path, "w") as f:
        f.write("BEGIN IONS\n")
        f.write(f"TITLE={title_prefix}\n")
        f.write(f"PEPMASS={mzs[max_idx]} {intensities[max_idx]}\n")
        f.write("CHARGE=1+\n")
        for mz, inten in df[["mz", "Intensity"]].values:
            f.write(f"{mz:.6f} {inten:.0f}\n")
        f.write("END IONS\n\n")

# print(len(split_asc_balanced("../hdd/data/fahmed/pigments_datsaset/N0193101.asc")))

# print(len(split_asc_balanced("../hdd/data/fahmed/battery_txt_files/cycled_graphite_9014_de_2048_250_after se cleaning_1.txt (5.07 MB).txt")))

# spectras = split_asc_balanced("../hdd/data/fahmed/battery_txt_files/gr_hc_si_one layer_negative_de_2048pixels_rotated_after fib_2.txt (5.26 MB).txt")
# write_mgf(spectras, "graphite_cycled_9014_cathode", "../hdd/data/fahmed/battery_mgf_from_txt/GR_HC_SI.mgf")


# #write_mgf_full_spectrum("../hdd/data/fahmed/pigments_dataset/N0193101.asc", "N0193101","../hdd/data/fahmed/pigments_mgf_full_spectrum/N0193101.mgf")


file_names = [
    "N0193101.asc", "N0193201.asc", "N0193301.asc", "N0193401.asc", "N0193501.asc", "N0193601.asc",
    "N0193102.asc", "N0193202.asc", "N0193302.asc", "N0193402.asc", "N0193502.asc", "N0193602.asc"
]

for file_name in file_names:
    input_path  = f"/hdd/data/fahmed/pigments_dataset/{file_name}"
    output_path = f"/hdd/data/fahmed/pigments_mgf/{file_name.replace('.asc', '.mgf')}"
    title       = file_name.replace('.asc', '')   # e.g. "N0193101"

    spectras = split_asc_balanced(input_path)
    write_mgf(spectras, title, output_path)
    print(f"Done: {output_path}")