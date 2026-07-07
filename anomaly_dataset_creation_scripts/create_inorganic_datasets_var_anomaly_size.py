"""
make_datasets.py
================
For each cluster_results CSV produced by run_anomalies.sh, split the data
into train / test sets using split_size=2, with anomaly spectra injected
into the test set.

Input  : ../anomaly_size_outputs/*.csv
Output : anomaly_var_size_files/anomaly_spectra_s{pct}.csv
         anomaly_var_size_files/train_data_s{pct}.csv
         anomaly_var_size_files/test_data_mixed_s{pct}.csv
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
INPUT_DIR   = Path("/hdd/data/fahmed/anomaly_inorganic_correct_pc_outputs_Gr_HC_Si_trunc")
OUTPUT_DIR  = Path("/hdd/data/fahmed/anomaly_var_size_files_Gr_HC_Si_inorganic_correct_pc")
SPLIT_SIZE  = 2
RANDOM_SEED = 42


def extract_pct(filename: str) -> str:
    """Pull the pct value from a filename like output_battery_anomalies_n3023_pct1.0.csv"""
    match = re.search(r"pct([\d.]+)", filename)
    return match.group(1) if match else filename


# ──────────────────────────────────────────────────────────────
# CORE
# ──────────────────────────────────────────────────────────────
def make_dataset(df: pd.DataFrame, pct: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Separate anomaly rows ──────────────────────────────
    df_anomalies     = df[df["identifier"].str.contains("anomalies", na=False)]
    anomaly_clusters = set(df_anomalies["cluster"].values)

    print(f"  Anomaly spectra    : {len(df_anomalies)}")
    df_anomalies.to_csv(OUTPUT_DIR / f"anomaly_spectra_s{pct}.csv", index=False)

    # ── 2. Non-anomaly rows ───────────────────────────────────
    df_non_anomaly = df[~df["cluster"].isin(anomaly_clusters)]

    # ── 3. Singleton vs multi clusters ───────────────────────
    cluster_counts     = df_non_anomaly["cluster"].value_counts()
    singleton_clusters = cluster_counts[cluster_counts <= SPLIT_SIZE].index
    multi_clusters     = cluster_counts[cluster_counts >  SPLIT_SIZE].index

    df_singletons = df_non_anomaly[df_non_anomaly["cluster"].isin(singleton_clusters)]
    df_multi      = df_non_anomaly[df_non_anomaly["cluster"].isin(multi_clusters)]

    print(f"  Singleton clusters : {len(singleton_clusters)}  ({len(df_singletons)} rows → train)")
    print(f"  Multi clusters     : {len(multi_clusters)}  ({len(df_multi)} rows → split 60/40)")

    # ── 4. Split multi-clusters 60 / 40 per bucket ───────────
    df_train = df_singletons.copy()
    df_test  = pd.DataFrame()

    for cluster in tqdm(df_multi["cluster"].unique(),
                        desc=f"  Splitting (k={SPLIT_SIZE})"):
        cluster_df = df_multi[df_multi["cluster"] == cluster]

        for bucket in cluster_df["bucket"].unique():
            cb_df = cluster_df[cluster_df["bucket"] == bucket]

            if len(cb_df) <= 1:
                df_train = pd.concat([df_train, cb_df], axis=0)
                continue

            df_60, df_40 = train_test_split(
                cb_df,
                test_size=0.4,
                stratify=cb_df["bucket"],
                random_state=RANDOM_SEED,
            )
            df_train = pd.concat([df_train, df_60], axis=0)
            df_test  = pd.concat([df_test,  df_40], axis=0)

    # ── 5. Add shuffled anomalies to test set ─────────────────
    df_test = pd.concat(
        [df_test, df_anomalies.sample(frac=1, random_state=RANDOM_SEED)], axis=0
    ).reset_index(drop=True)

    n_anom = len(df_test[df_test["cluster"].isin(anomaly_clusters)])
    print(f"  Train rows         : {len(df_train)}")
    print(f"  Test rows          : {len(df_test)}  (of which {n_anom} anomalies)")

    # ── 6. Write outputs ──────────────────────────────────────
    df_train.to_csv(OUTPUT_DIR / f"train_data_s{pct}.csv",      index=False)
    df_test.to_csv( OUTPUT_DIR / f"test_data_mixed_s{pct}.csv", index=False)


# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────
csv_files = sorted(INPUT_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSVs found in {INPUT_DIR}")

print(f"Found {len(csv_files)} cluster-results file(s) in {INPUT_DIR}\n")

for csv_path in csv_files:
    pct = extract_pct(csv_path.name)

    print(f"{'='*60}")
    print(f"Input  : {csv_path.name}  (pct={pct})")
    print(f"Output : {OUTPUT_DIR}/{{anomaly,train,test}}_*_s{pct}.csv")

    df = pd.read_csv(csv_path)
    make_dataset(df, pct)
    print()

print("All datasets written.")