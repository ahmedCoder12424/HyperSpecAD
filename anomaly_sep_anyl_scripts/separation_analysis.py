#!/usr/bin/env python3
"""separation_analysis.py

Computes per-cluster intra-cluster cohesion and inter-cluster separation
stats (cosine distance in hyperdimensional space) for a clustered
mass-spec dataset, and writes the result to CSV.

Parameterized by dataset so the same script can be pointed at proteomics,
battery/inorganic, or any other Hyper-Spec dataset without editing code.
"""

import argparse
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import src.hd_cluster as hd_cluster

KEY_COLS = ["bucket", "precursor_charge", "identifier", "scan", "retention_time"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cluster separation/cohesion stats for a Hyper-Spec dataset."
    )
    parser.add_argument(
        "--cluster-results",
        required=True,
        help="Path to cluster_results CSV (e.g. cluster_results_main.csv)",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to dataset metadata CSV (e.g. 1468_dataset_meta.csv)",
    )
    parser.add_argument(
        "--spectra-hvs",
        required=True,
        help="Path to spectra hypervectors .npy file (e.g. spectra_hvs.npy)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV (default: <outdir>/<prefix>cluster_separation_stats_by_bucket_avg.csv)",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory to save output CSV if --output not given (default: current directory)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix for the output CSV, e.g. 'proteomics_' (default: none)",
    )
    parser.add_argument(
        "--mask-window",
        type=int,
        default=20,
        help="m/z mask window passed to fast_nb_cosine_dist_mask (default: 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cluster_results = pd.read_csv(args.cluster_results)
    metadata_df = pd.read_csv(args.metadata)
    spectra_hvs = np.load(args.spectra_hvs)

    cluster_results = metadata_df[KEY_COLS].merge(
        cluster_results, on=KEY_COLS, how="left"
    )

    if "hv_idx" not in cluster_results.columns:
        cluster_results = cluster_results.reset_index(drop=True)
        cluster_results["hv_idx"] = cluster_results.index

    cluster_stats = []
    unique_clusters = cluster_results["cluster"].unique()

    # collect one rep per cluster
    cluster_reps = {}

    for cluster_id in tqdm(unique_clusters, desc="Finding representatives"):
        cluster_rows = cluster_results[cluster_results["cluster"] == cluster_id]

        rep_rows = cluster_rows[cluster_rows["is_representative"] == True]
        rep_row = rep_rows.iloc[0] if len(rep_rows) > 0 else cluster_rows.iloc[0]

        rep_hv_idx = int(rep_row["hv_idx"])

        cluster_reps[cluster_id] = {
            "hv_idx": rep_hv_idx,
            "hv": spectra_hvs[rep_hv_idx],
            "mz": float(rep_row["precursor_mz"]),
            "bucket": int(rep_row["bucket"]),
        }

    # group cluster ids by bucket
    bucket_to_clusters = {}
    for cluster_id, rep_info in cluster_reps.items():
        b = rep_info["bucket"]
        bucket_to_clusters.setdefault(b, []).append(cluster_id)

    for cluster_id in tqdm(unique_clusters, desc="Computing separation"):
        cluster_rows = cluster_results[cluster_results["cluster"] == cluster_id]

        rep_hv = cluster_reps[cluster_id]["hv"]
        rep_mz = cluster_reps[cluster_id]["mz"]
        rep_bucket = cluster_reps[cluster_id]["bucket"]

        # intra distances: rep -> members
        member_indices = cluster_rows["hv_idx"].astype(int).to_numpy()
        member_hvs = spectra_hvs[member_indices]
        member_mzs = cluster_rows["precursor_mz"].astype(float).to_numpy()

        all_hvs = np.vstack([rep_hv[None, :], member_hvs])
        all_mzs = np.concatenate([[rep_mz], member_mzs]).astype(np.float32).reshape(-1, 1)

        dist_mat = hd_cluster.fast_nb_cosine_dist_mask(
            all_hvs, all_mzs, args.mask_window, "numpy"
        )

        intra_dists = dist_mat[0, 1:]

        avg_intra = float(np.mean(intra_dists))
        max_intra = float(np.max(intra_dists))

        # inter distances: rep -> other reps in same bucket
        candidate_clusters = [
            c for c in bucket_to_clusters[rep_bucket] if c != cluster_id
        ]

        if len(candidate_clusters) == 0:
            nearest_cluster_dist = np.inf
            nearest_cluster_id = None
            cluster_min_sep = np.nan
            cluster_avg_sep = np.nan
            cluster_p25_sep = np.nan
        else:
            other_hvs = np.vstack([cluster_reps[c]["hv"] for c in candidate_clusters])
            other_mzs = np.array(
                [cluster_reps[c]["mz"] for c in candidate_clusters], dtype=np.float32
            )

            all_hvs = np.vstack([member_hvs, other_hvs])
            all_mzs = np.concatenate([member_mzs, other_mzs]).astype(np.float32).reshape(-1, 1)

            dist_mat = hd_cluster.fast_nb_cosine_dist_mask(
                all_hvs, all_mzs, args.mask_window, "numpy"
            )

            n_members = len(member_hvs)

            # rows: members, columns: other cluster reps
            member_to_other_rep = dist_mat[:n_members, n_members:]

            nearest_per_member = member_to_other_rep.min(axis=1)

            cluster_min_sep = nearest_per_member.min()
            cluster_avg_sep = nearest_per_member.mean()
            cluster_p25_sep = np.percentile(nearest_per_member, 50)

            # use 25th percentile as nearest_cluster_dist
            nearest_cluster_dist = cluster_p25_sep

            # nearest cluster id based on average distance across members
            avg_dist_per_candidate = member_to_other_rep.mean(axis=0)
            best_i = int(np.argmin(avg_dist_per_candidate))
            nearest_cluster_id = candidate_clusters[best_i]

        if nearest_cluster_id is None:
            separation_margin = np.nan
            separation_score = np.nan
        else:
            separation_margin = nearest_cluster_dist - max_intra
            separation_score = nearest_cluster_dist / avg_intra if avg_intra > 0 else np.inf

        cluster_stats.append({
            "cluster": cluster_id,
            "bucket": rep_bucket,
            "size": len(cluster_rows),
            "avg_intra": avg_intra,
            "max_intra": max_intra,
            "nearest_cluster": nearest_cluster_id,
            "nearest_inter": nearest_cluster_dist,
            "separation_margin": separation_margin,
            "separation_score": separation_score,
            "cluster_min_sep": cluster_min_sep,
            "cluster_avg_sep": cluster_avg_sep,
            "cluster_p50_sep": cluster_p25_sep,
        })

    cluster_stats_df = pd.DataFrame(cluster_stats)

    output_path = args.output
    if output_path is None:
        os.makedirs(args.outdir, exist_ok=True)
        output_path = os.path.join(
            args.outdir, f"{args.prefix}cluster_separation_stats_by_bucket_avg.csv"
        )

    cluster_stats_df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()