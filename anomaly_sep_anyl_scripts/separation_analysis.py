import pandas as pd
import numpy as np
import src.hd_cluster 

cluster_results = pd.read_csv("cluster_results_main.csv")
metadata_df = pd.read_csv("1468_dataset_meta.csv")
spectra_hvs = np.load("spectra_hvs.npy")
print(spectra_hvs)

print(metadata_df['retention_time'][0:5])
print(cluster_results['retention_time'][0:5])

cluster_stats = []
#ist = hd_cluster.fast_nb_cosine_dist_mask(both, mzs, 20, "numpy")

unique_clusters = cluster_results['cluster'].unique()


for cluster_id in unique_clusters:

    member_indices = cluster_results[cluster_results['cluster']==cluster_id].indices
    print(member_indices)
    rep_idx = cluster_results[(cluster_results['cluster']==cluster_id) & (cluster_results['is_representative']==True)].index
    print(rep_idx)
    sys.exit(0)
    rep_hv = ...

    intra_dists = []

    for member in cluster:
        d = dist(rep_hv, member_hv)
        intra_dists.append(d)

    avg_intra = np.mean(intra_dists)
    max_intra = np.max(intra_dists)

    nearest_cluster_dist = inf

    for other_cluster in clusters:

        if other_cluster == cluster_id:
            continue

        d = dist(rep_hv, other_rep_hv)

        nearest_cluster_dist = min(nearest_cluster_dist, d)

    separation_score = nearest_cluster_dist / avg_intra

    cluster_stats.append({
        "cluster": cluster_id,
        "size": len(member_indices),
        "avg_intra": avg_intra,
        "max_intra": max_intra,
        "nearest_inter": nearest_cluster_dist,
        "separation_score": separation_score
    })

