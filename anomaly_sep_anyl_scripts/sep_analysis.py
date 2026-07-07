import sys
import pandas as pd
import numpy as np
import src.hd_cluster as hd_cluster
from tqdm import tqdm
cluster_results = pd.read_csv("cluster_results_main.csv")
metadata_df = pd.read_csv("1468_dataset_meta.csv")
spectra_hvs = np.load("spectra_hvs.npy")

print("spectra_hvs shape:", spectra_hvs.shape)
print(metadata_df["retention_time"][0:5])
print(cluster_results["retention_time"][0:5])

# Use hv_idx if available, otherwise assume dataframe index aligns with spectra_hvs
if "hv_idx" not in cluster_results.columns:
    cluster_results = cluster_results.reset_index(drop=True)
    cluster_results["hv_idx"] = cluster_results.index

cluster_stats = []

unique_clusters = cluster_results["cluster"].unique()

# First collect one representative HV per cluster
cluster_reps = {}
print("finding representatives")
for cluster_id in tqdm(unique_clusters, desc="Finding representatives"):
    cluster_rows = cluster_results[cluster_results["cluster"] == cluster_id]

    rep_rows = cluster_rows[cluster_rows["is_representative"] == True]

    # If no representative is marked, fallback to first member
    if len(rep_rows) == 0:
        rep_row = cluster_rows.iloc[0]
    else:
        rep_row = rep_rows.iloc[0]

    rep_hv_idx = int(rep_row["hv_idx"])
    rep_hv = spectra_hvs[rep_hv_idx]
    rep_mz = float(rep_row["precursor_mz"])

    cluster_reps[cluster_id] = {
        "hv_idx": rep_hv_idx,
        "hv": rep_hv,
        "mz": rep_mz,
    }
print("finding separation")
# Now compute intra-cluster and inter-cluster separation
for cluster_id in tqdm(unique_clusters, desc="Finding separation"):
    
    
    cluster_rows = cluster_results[cluster_results["cluster"] == cluster_id]

    rep_hv = cluster_reps[cluster_id]["hv"]
    rep_mz = cluster_reps[cluster_id]["mz"]

    intra_dists = []

    for _, row in cluster_rows.iterrows():
        member_hv_idx = int(row["hv_idx"])
        member_hv = spectra_hvs[member_hv_idx]
        member_mz = float(row["precursor_mz"])

        both = np.stack([rep_hv, member_hv])
        mzs = np.array([rep_mz, member_mz], dtype=np.float32).reshape(-1, 1)

        dist_mat = hd_cluster.fast_nb_cosine_dist_mask(both, mzs, 20, "numpy")
        d = dist_mat[0, 1]

        intra_dists.append(d)

    avg_intra = float(np.mean(intra_dists))
    max_intra = float(np.max(intra_dists))
   #print(avg_intra, max_intra)
    nearest_cluster_dist = np.inf
    nearest_cluster_id = None

    for other_cluster_id in unique_clusters:
        if other_cluster_id == cluster_id:
            continue

        other_rep_hv = cluster_reps[other_cluster_id]["hv"]
        other_rep_mz = cluster_reps[other_cluster_id]["mz"]

        both = np.stack([rep_hv, other_rep_hv])
        mzs = np.array([rep_mz, other_rep_mz], dtype=np.float32).reshape(-1, 1)

        dist_mat = hd_cluster.fast_nb_cosine_dist_mask(both, mzs, 20, "numpy")
        d = dist_mat[0, 1]

        if d < nearest_cluster_dist:
            nearest_cluster_dist = float(d)
            nearest_cluster_id = other_cluster_id

    separation_margin = nearest_cluster_dist - max_intra
    separation_ratio = avg_intra/nearest_cluster_dist
    print(separation_margin)
    cluster_stats.append({
        "cluster": cluster_id,
        "size": len(cluster_rows),
        "rep_hv_idx": cluster_reps[cluster_id]["hv_idx"],
        "avg_intra_dist": avg_intra,
        "max_intra_dist": max_intra,
        "nearest_cluster": nearest_cluster_id,
        "nearest_cluster_dist": nearest_cluster_dist,
        "separation_margin": separation_margin,
        "separation ratio": separation_ratio
    })

cluster_stats_df = pd.DataFrame(cluster_stats)

print(cluster_stats_df.head())
cluster_stats_df.to_csv("cluster_separation_stats.csv", index=False)

print("Saved cluster_separation_stats.csv")
