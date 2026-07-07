import pandas as pd
import os

ASC_DIR = "."  # run from the folder

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

def asc_to_mgf_entry(asc_path, pigment_name, polarity):
    df = pd.read_csv(asc_path, sep='\s+', comment='#', skiprows=1,
                     header=None, names=['mz', 'intensity'])
    df = df[(df['mz'] > 0) & (df['intensity'] > 0)]

    max_idx = df['intensity'].idxmax()
    charge = "1+" if polarity == "positive" else "1-"

    lines = []
    lines.append("BEGIN IONS")
    lines.append(f"TITLE={pigment_name}_{polarity}")
    lines.append("SCANS=1")
    lines.append("RTINSECONDS=0")
    lines.append(f"PEPMASS={df.loc[max_idx, 'mz']} {df.loc[max_idx, 'intensity']}")
    lines.append(f"CHARGE={charge}")
    for _, row in df.iterrows():
        lines.append(f"{row['mz']} {row['intensity']}")
    lines.append("END IONS\n")
    return "\n".join(lines), len(df)

# Convert all files
with open("pigments_all.mgf", "w") as out_f:
    for fname in sorted(os.listdir(ASC_DIR)):
        if not fname.endswith(".asc"):
            continue

        # Parse accession and polarity from filename e.g. N0193101.asc
        accession = fname[1:6]   # 01931
        polarity_code = fname[6:8]  # 01 or 02

        pigment = pigment_names.get(accession)
        polarity = polarity_map.get(polarity_code)

        if not pigment or not polarity:
            print(f"Skipping unknown file: {fname}")
            continue

        asc_path = os.path.join(ASC_DIR, fname)
        entry, n_peaks = asc_to_mgf_entry(asc_path, pigment, polarity)
        out_f.write(entry)
        print(f"{fname} -> {pigment} ({polarity}), {n_peaks} peaks")

print("\nDone! Written to pigments_all.mgf")