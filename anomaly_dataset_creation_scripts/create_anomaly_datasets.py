import pandas as pd
import numpy as np
import hd_cluster 
from tqdm import tqdm
df = pd.read_csv("cluster_results_main.csv")

separation_results = pd.read_csv("cluster_separation_stats_by_bucket_avg.csv")
print(len(separation_results))




# separation_results = pd.read_csv("cluster_results_main.csv")

# best separated clusters first
best_clusters = separation_results.sort_values(
    "separation_margin",
    ascending=False
)
# print(best_clusters.head())



threshold = .72
sizes = [1,2,3,4,5,6,15,23]
for size in sizes:
    print(size, best_clusters[best_clusters["size"] == size].iloc[0]['separation_margin'], best_clusters[best_clusters["size"] == size].iloc[0]['cluster'])


max_cluster_size = int(best_clusters["size"].max())

sizes = list(range(1, max_cluster_size + 1, 1))

print("max_cluster_size:", max_cluster_size)
print("sizes:", sizes)

for size in sizes:

  # print(size, best_clusters[best_clusters['size']==size]['cluster'].iloc[0].tolist())
  # anomaly_cluster = best_clusters[best_clusters['size']==size]['cluster'].iloc[0]

  candidates = best_clusters[
    (best_clusters["size"] == size) &
    (best_clusters["separation_margin"] >= threshold)
]

  if len(candidates) > 0:
      anomaly_cluster = candidates["cluster"].iloc[0]
  else:
      anomaly_cluster = None
      print(f"No cluster found for size={size} with separation_score > {threshold}")
      continue 


  df_anomalies = df[df["cluster"]== anomaly_cluster ]

  df_anomalies.to_csv(
      f"anomaly_size_files/anomaly_spectra_s{size}.csv",
      index=False
  )


  df_non_anomaly = df[df["cluster"]!= anomaly_cluster ]
  from sklearn.model_selection import train_test_split

  # count rows per cluster
  cluster_counts = df_non_anomaly['cluster'].value_counts()

  # singleton clusters -> must go to 60%
  singleton_clusters = cluster_counts[cluster_counts <=1].index
  multi_clusters = cluster_counts[cluster_counts > 1].index

  df_singletons = df_non_anomaly[df_non_anomaly['cluster'].isin(singleton_clusters)]
  df_multi =  df_non_anomaly[df_non_anomaly['cluster'].isin(multi_clusters)]

  df_train = df_singletons
  df_test = pd.DataFrame()
  for cluster in tqdm( df_multi['cluster'].unique(), desc=f"making dataset split for size {size}"):

      cluster_df = df_multi[df_multi['cluster'] == cluster]

      buckets = cluster_df['bucket'].unique()
      for bucket in buckets:

          cluster_bucket_df = cluster_df[cluster_df['bucket']==bucket]
          n = len(cluster_bucket_df['bucket'].tolist())
          if (n <=1):
            df_train = pd.concat([df_train,cluster_bucket_df], axis=0)
            continue


          df_cluster_60, df_cluster_40 = train_test_split(
          cluster_bucket_df,
          test_size=0.4,
          stratify=cluster_bucket_df['bucket'],
          random_state=42
        )
          # print(len(df_cluster_60))
          # print(len(df_cluster_40))
          if(cluster == 897 or cluster == 893):
            print("train",df_cluster_60['bucket'].tolist())
            print("test" ,df_cluster_40['bucket'].tolist())

          df_train = pd.concat([df_train, df_cluster_60], axis=0)
          df_test = pd.concat([df_test, df_cluster_40], axis=0)

  df_train.to_csv(f"anomaly_size_files/train_datas{size}.csv", index=False)

  excluded_protein = pd.read_csv(f"anomaly_size_files/anomaly_spectra_s{size}.csv")

  # shuffle rows randomly
  excluded_shuffled = excluded_protein.sample(frac=1, random_state=42)
  print("number of expected anomlaies", len(excluded_protein))

  # add to test set
  df_test = pd.concat([df_test, excluded_shuffled], axis=0).reset_index(drop=True)
  print("number of anomalies in test data", len(df_test[df_test["cluster"]==(anomaly_cluster)]))

  df_test.to_csv(f"anomaly_size_files/test_data-mixeds{size}.csv", index=False)

