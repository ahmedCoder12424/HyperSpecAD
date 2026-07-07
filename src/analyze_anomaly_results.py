import pandas as pd
import argparse
import os
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument("file1", help="First CSV file, e.g. anomaly_results_s4.csv")
parser.add_argument("file2", help="Second CSV file, e.g. excluded_spectra_smaller.csv")
parser.add_argument("file3", help="Second CSV file, e.g. excluded_spectra_smaller.csv")
parser.add_argument("file4", help="Second CSV file, e.g. excluded_spectra_smaller.csv")
parser.add_argument("param", help="Parameter/label for this run, e.g. threshold value")
args = parser.parse_args()

df1 = pd.read_csv(args.file1)
df2 = pd.read_csv(args.file2)

df1_unique = df1[df1.columns].drop_duplicates()
df2_unique = df2[df2.columns].drop_duplicates()

param = args.param

print("size of anomaly dataframe", len(df1), "size of excluded spectra dataframe", len(df2))

excluded_proteins = df2


def include_only_proteins(spectra_meta_df, spectra_hvs):
    print("nexcluded", len(excluded_proteins))

    blength = len(spectra_meta_df)
    match_cols = ["identifier", "scan"]

    spectra_meta_df["identifier"] = spectra_meta_df["identifier"].astype(str).str.strip()
    excluded_proteins["identifier"] = excluded_proteins["identifier"].astype(str).str.strip()

    spectra_meta_df["scan"] = spectra_meta_df["scan"].astype(int)
    excluded_proteins["scan"] = excluded_proteins["scan"].astype(int)

    spectra_meta_df["retention_time"] = spectra_meta_df["retention_time"].astype(float).round(4)
    excluded_proteins["retention_time"] = excluded_proteins["retention_time"].astype(float).round(4)

    tmp = spectra_meta_df.merge(
        excluded_proteins[match_cols].drop_duplicates(),
        on=match_cols,
        how="left",
        indicator=True
    )
    excluded_mask = tmp["_merge"].eq("both").to_numpy()

    spectra_meta_df_excluded = spectra_meta_df.loc[excluded_mask].copy()
    return spectra_meta_df_excluded


expected_total_anomalies = len(df2)

true_anomalies = len(include_only_proteins(df1, None))
false_anomalies = len(df1) - true_anomalies
# Anomalies that exist in df2 (ground truth) but were never flagged in df1
false_negatives = expected_total_anomalies - true_anomalies

print("total number of true anomalies", expected_total_anomalies)

detection_rate = round(true_anomalies / expected_total_anomalies, 2) if expected_total_anomalies > 0 else 0
print("detection_rate", detection_rate)
print("number of true anomalies in results", true_anomalies)

true_pos_rate = 0
false_pos_rate = 0
if (true_anomalies + false_anomalies) > 0:
    true_pos_rate = round(true_anomalies / (true_anomalies + false_anomalies), 2)
    false_pos_rate = round(false_anomalies / (true_anomalies + false_anomalies), 2)
    print("true positive rate", true_pos_rate)

print("number of false anomalies in results", false_anomalies)

if (true_anomalies + false_anomalies) > 0:
    print("false positive rate", false_pos_rate)

# --- Precision / Recall / F1 ---
# Precision: of everything flagged as anomalous, how much was correct
precision = round(true_anomalies / (true_anomalies + false_anomalies), 4) if (true_anomalies + false_anomalies) > 0 else 0.0
# Recall: of all true anomalies, how many were found
recall = round(true_anomalies / expected_total_anomalies, 4) if expected_total_anomalies > 0 else 0.0
# F1: harmonic mean of precision and recall
f1_score = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

print("precision", precision)
print("recall", recall)
print("f1_score", f1_score)



cluster_df =  pd.read_csv(args.file3)
gt_df = pd.read_csv(args.file2)

match_cols = ["identifier", "scan"]

cluster_df["identifier"] = cluster_df["identifier"].astype(str).str.strip()
gt_df["identifier"] = gt_df["identifier"].astype(str).str.strip()

cluster_df["scan"] = cluster_df["scan"].astype(int)
gt_df["scan"] = gt_df["scan"].astype(int)

tmp = cluster_df.merge(
    gt_df[match_cols].drop_duplicates(),
    on=match_cols,
    how="left",
    indicator=True
)
y_true = tmp["_merge"].eq("both").astype(int)
print(cluster_df.columns)
y_score = cluster_df["anomaly"]

try:
    auroc = roc_auc_score(y_true, y_score)
except ValueError:
    auroc = None

try:
    auprc = average_precision_score(y_true, y_score)
except ValueError:
    auprc = None
    
print(
    true_pos_rate,
    false_pos_rate,
    detection_rate,
    true_anomalies,
    false_anomalies,
    expected_total_anomalies,
    precision,
    recall,
    f1_score,
    auroc,
    auprc,
)

#  "anomaly_validation_results/sweep_anomaly_threshold_results_Gr_HC_Si_trunc_new_encoding.txt"
output_path = (args.file4)

# Write a header row once, only if the file doesn't already exist (or is empty)
needs_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0

with open(output_path, "a") as f:
    if needs_header:
        print(
            "param",
            "true_pos_rate",
            "false_pos_rate",
            "detection_rate",
            "true_anomalies",
            "false_anomalies",
            "expected_total_anomalies",
            "precision",
            "recall",
            "f1_score",
            "auroc",
            "auprc",
            sep=", ",
            file=f
        )
    print(
        param,
        true_pos_rate,
        false_pos_rate,
        detection_rate,
        true_anomalies,
        false_anomalies,
        expected_total_anomalies,
        precision,
        recall,
        f1_score,
        auroc,
        auprc,
        sep=", ",
        file=f
    )